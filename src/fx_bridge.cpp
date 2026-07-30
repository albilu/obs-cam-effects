#include "fx_bridge.h"

#include "fx/models_dl/downloader.h"
#include "fx/pipeline/segmentation_pipeline.h"
#include "fx/worker.h"

#include <obs-module.h>
#include <util/platform.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

struct ManifestEntry {
	std::string id;
	std::string kind;
	std::string url;
	std::string sha256;
	uint64_t size = 0;
	std::string file;
	std::string notice;
	std::vector<std::string> extract;
};

int64_t nowMs()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(
		       std::chrono::steady_clock::now().time_since_epoch())
		.count();
}

bool fileExists(const std::string &p)
{
	struct stat st;
	return stat(p.c_str(), &st) == 0;
}

std::string cacheDir(const char *sub)
{
	const char *home = getenv("HOME");
	return std::string(home ? home : ".") +
	       "/.config/obs-cam-effects/" + sub;
}

/* libobs' obs_data JSON parser drops primitive array items, so the
 * "extract" string list is collected from the raw manifest text
 * (controlled, hand-written JSON). */
std::vector<std::string> extractListForId(const std::string &json,
					  const std::string &id)
{
	std::vector<std::string> out;
	size_t pos = json.find("\"id\": \"" + id + "\"");
	if (pos == std::string::npos)
		return out;
	size_t ex = json.find("\"extract\"", pos);
	if (ex == std::string::npos)
		return out;
	/* The "extract" key must belong to THIS entry: no other "id" may
	 * appear between the entry start and the key. */
	size_t idBetween = json.find("\"id\"", pos + 1);
	if (idBetween != std::string::npos && idBetween < ex)
		return out;
	size_t open = json.find('[', ex);
	size_t close = open == std::string::npos ? std::string::npos
						 : json.find(']', open);
	/* Bail if another entry starts before the array closes. */
	size_t nextId = json.find("\"id\"", ex + 9);
	if (close == std::string::npos ||
	    (nextId != std::string::npos && nextId < close))
		return out;
	size_t p = open + 1;
	while (p < close) {
		size_t q1 = json.find('"', p);
		if (q1 == std::string::npos || q1 >= close)
			break;
		size_t q2 = json.find('"', q1 + 1);
		if (q2 == std::string::npos || q2 > close)
			break;
		out.push_back(json.substr(q1 + 1, q2 - q1 - 1));
		p = q2 + 1;
	}
	return out;
}

} // namespace

struct cam_fx {
	std::unique_ptr<fx::Worker> worker;
	std::shared_ptr<fx::SegmentationPipeline> pipeline;
	std::unique_ptr<fx::models_dl::Downloader> downloader;
	std::vector<ManifestEntry> manifest;

	std::string litePath;
	std::string standardPath;
	std::string qualityPath;
	int threads = 2;
	int requestedTier = 0; // 0=auto, 1=lite, 2=standard, 3=quality
	int tierInEffect = 0;  // (int)fx::SegTier + 1
	fx::MaskParams params;

	uint64_t seenSeq = 0;
	std::vector<uint8_t> u8;
	bool loggedFirstMask = false;

	int64_t fpsWinStart = 0;
	uint64_t fpsCount = 0;
	uint64_t fpsLast = 0;
};

namespace {

/* 0=auto / 3=quality resolve to Quality when the model file exists,
 * else Standard. Explicit lite/standard map directly. */
fx::SegTier resolveTier(int requested, const std::string &qualityPath)
{
	switch (requested) {
	case 1:
		return fx::SegTier::Lite;
	case 2:
		return fx::SegTier::Standard;
	case 3:
	case 0:
	default:
		if (!qualityPath.empty() && fileExists(qualityPath))
			return fx::SegTier::Quality;
		return fx::SegTier::Standard;
	}
}

/* Builds a pipeline on the calling thread (~100ms model load), then
 * hot-swaps it into the worker. Falls back to Standard when Quality
 * construction fails. */
bool buildAndSwap(cam_fx *fx, fx::SegTier tier)
{
	try {
		auto p = std::make_shared<fx::SegmentationPipeline>(
			tier, fx->litePath, fx->standardPath, fx->qualityPath,
			fx->threads);
		p->setMaskParams(fx->params);
		fx->pipeline = p;
		fx->tierInEffect = (int)tier + 1;
		static const char *names[] = {"lite", "standard", "quality"};
		blog(LOG_INFO, "obs-cam-effects: pipeline tier in effect: %s",
		     names[(int)tier]);
		if (fx->worker) {
			fx::Worker::Processor proc =
				[p](const fx::Frame &frame) {
					return p->process(frame);
				};
			fx->worker->setProcessor(std::move(proc));
		}
		return true;
	} catch (const std::exception &e) {
		blog(LOG_WARNING,
		     "obs-cam-effects: pipeline build failed (tier %d): %s",
		     (int)tier, e.what());
		if (tier != fx::SegTier::Standard)
			return buildAndSwap(fx, fx::SegTier::Standard);
		return false;
	} catch (...) {
		blog(LOG_WARNING,
		     "obs-cam-effects: pipeline build failed (tier %d)",
		     (int)tier);
		if (tier != fx::SegTier::Standard)
			return buildAndSwap(fx, fx::SegTier::Standard);
		return false;
	}
}

void parseManifest(cam_fx *fx)
{
	char *path = obs_module_file("models/manifest.json");
	if (!path) {
		blog(LOG_WARNING, "obs-cam-effects: manifest.json not found");
		return;
	}
	std::string raw;
	{
		std::ifstream f(path);
		std::ostringstream ss;
		ss << f.rdbuf();
		raw = ss.str();
	}
	obs_data_t *data = obs_data_create_from_json_file(path);
	bfree(path);
	if (!data) {
		blog(LOG_WARNING,
		     "obs-cam-effects: manifest.json failed to parse");
		return;
	}
	obs_data_array_t *models = obs_data_get_array(data, "models");
	if (models) {
		size_t count = obs_data_array_count(models);
		for (size_t i = 0; i < count; i++) {
			obs_data_t *item = obs_data_array_item(models, i);
			if (!item)
				continue;
			ManifestEntry e;
			e.id = obs_data_get_string(item, "id");
			e.kind = obs_data_get_string(item, "kind");
			e.url = obs_data_get_string(item, "url");
			e.sha256 = obs_data_get_string(item, "sha256");
			e.size = (uint64_t)obs_data_get_int(item, "size");
			e.file = obs_data_get_string(item, "file");
			e.notice = obs_data_get_string(item, "notice");
			e.extract = extractListForId(raw, e.id);
			if (!e.id.empty())
				fx->manifest.push_back(std::move(e));
			obs_data_release(item);
		}
		obs_data_array_release(models);
	}
	obs_data_release(data);
	blog(LOG_INFO, "obs-cam-effects: manifest parsed, %zu entries",
	     fx->manifest.size());
}

} // namespace

extern "C" {

cam_fx_t *cam_fx_create(const char *lite_path, const char *standard_path,
			const char *quality_path, int threads)
{
	try {
		auto fx = std::make_unique<cam_fx>();
		fx->litePath = lite_path ? lite_path : "";
		fx->standardPath = standard_path ? standard_path : "";
		fx->qualityPath = quality_path ? quality_path : "";
		fx->threads = threads;
		fx->downloader = std::make_unique<fx::models_dl::Downloader>();
		os_mkdirs(cacheDir("models").c_str());
		os_mkdirs(cacheDir("providers").c_str());
		parseManifest(fx.get());
		if (!buildAndSwap(fx.get(),
				  resolveTier(0, fx->qualityPath)))
			return nullptr;
		std::shared_ptr<fx::SegmentationPipeline> p = fx->pipeline;
		fx->worker = std::make_unique<fx::Worker>(
			[p](const fx::Frame &f) { return p->process(f); });
		fx->worker->start();
		return fx.release();
	} catch (...) {
		return nullptr;
	}
}

void cam_fx_destroy(cam_fx_t *fx)
{
	delete fx;
}

void cam_fx_submit(cam_fx_t *fx, const uint8_t *bgra, int w, int h,
		   int linesize)
{
	auto frame = std::make_shared<fx::Frame>();
	frame->width = w;
	frame->height = h;
	frame->bgra.resize((size_t)w * h * 4);
	for (int y = 0; y < h; y++)
		std::memcpy(frame->bgra.data() + (size_t)y * w * 4,
			    bgra + (size_t)y * linesize, (size_t)w * 4);
	fx->worker->submit(std::move(frame));
}

int cam_fx_try_get_mask(cam_fx_t *fx, const uint8_t **px, int *w, int *h,
			uint64_t *seq)
{
	uint64_t s = 0;
	auto m = fx->worker->tryGetLatest(s);
	if (!m)
		return 0;
	if (s != fx->seenSeq) {
		fx->u8.resize(m->px.size());
		for (size_t i = 0; i < m->px.size(); i++) {
			float v = m->px[i];
			v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
			fx->u8[i] = (uint8_t)(v * 255.0f + 0.5f);
		}
		fx->seenSeq = s;
		/* Rolling fps: masks seen per completed 1s window. */
		int64_t now = nowMs();
		if (fx->fpsWinStart == 0)
			fx->fpsWinStart = now;
		fx->fpsCount++;
		int64_t elapsed = now - fx->fpsWinStart;
		if (elapsed >= 1000) {
			fx->fpsLast = fx->fpsCount * 1000 / (uint64_t)elapsed;
			fx->fpsCount = 0;
			fx->fpsWinStart = now;
		}
	}
	if (!fx->loggedFirstMask) {
		fx->loggedFirstMask = true;
		static const char *names[] = {"none", "lite", "standard",
					      "quality"};
		blog(LOG_INFO,
		     "obs-cam-effects: first mask published (%dx%d, tier=%s)",
		     m->width, m->height,
		     fx->tierInEffect >= 0 && fx->tierInEffect <= 3
			     ? names[fx->tierInEffect]
			     : "?");
	}
	*px = fx->u8.data();
	*w = m->width;
	*h = m->height;
	*seq = s;
	return 1;
}

int cam_fx_is_fresh(cam_fx_t *fx, uint64_t max_age_ms)
{
	return fx->worker->isFresh(max_age_ms) ? 1 : 0;
}

void cam_fx_set_tier(cam_fx_t *fx, int tier)
{
	fx->requestedTier = tier;
	fx::SegTier resolved = resolveTier(tier, fx->qualityPath);
	if (fx->pipeline && fx->tierInEffect == (int)resolved + 1)
		return; // idempotent: effective tier unchanged
	buildAndSwap(fx, resolved);
}

int cam_fx_tier_in_effect(cam_fx_t *fx)
{
	return fx->tierInEffect;
}

int cam_fx_quality_available(cam_fx_t *fx)
{
	return (!fx->qualityPath.empty() && fileExists(fx->qualityPath))
		       ? 1
		       : 0;
}

void cam_fx_set_mask_params(cam_fx_t *fx, float threshold, float contour,
			    float feather, float beta)
{
	fx->params.threshold = threshold;
	fx->params.contourFrac = contour;
	fx->params.featherRadius = feather;
	fx->params.beta = beta;
	if (fx->pipeline)
		fx->pipeline->setMaskParams(fx->params);
}

int cam_fx_start_download(cam_fx_t *fx, const char *id)
{
	if (!fx || !id)
		return -1;
	const ManifestEntry *entry = nullptr;
	for (const auto &m : fx->manifest) {
		if (m.id == id) {
			entry = &m;
			break;
		}
	}
	if (!entry)
		return -1;
	fx::models_dl::DownloadRequest req;
	req.url = entry->url;
	req.sha256 = entry->sha256;
	req.expectedSize = entry->size;
	if (entry->kind == "provider") {
		req.destPath = cacheDir("providers") + "/" + entry->id + ".tgz";
		req.extractMembers = entry->extract;
		req.extractDestDir = cacheDir("providers");
	} else {
		req.destPath = cacheDir("models") + "/" + entry->file;
	}
	try {
		fx->downloader->start(req);
		blog(LOG_INFO, "obs-cam-effects: download started: %s",
		     entry->id.c_str());
		return 0;
	} catch (...) {
		return -1;
	}
}

int cam_fx_download_state(cam_fx_t *fx, char *buf, int buf_len,
			  double *progress)
{
	if (!fx)
		return -1;
	if (buf && buf_len > 0)
		snprintf(buf, (size_t)buf_len, "%s",
			 fx::models_dl::stateName(fx->downloader->state()));
	if (progress)
		*progress = fx->downloader->progress();
	return 0;
}

int cam_fx_notice(cam_fx_t *fx, const char *id, char *buf, int buf_len)
{
	if (!fx || !id)
		return -1;
	for (const auto &m : fx->manifest) {
		if (m.id == id) {
			if (buf && buf_len > 0)
				snprintf(buf, (size_t)buf_len, "%s",
					 m.notice.c_str());
			return 0;
		}
	}
	if (buf && buf_len > 0)
		buf[0] = '\0';
	return -1;
}

uint64_t cam_fx_fps(cam_fx_t *fx)
{
	return fx->fpsLast;
}

} // extern "C"
