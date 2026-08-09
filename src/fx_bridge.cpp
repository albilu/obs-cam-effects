#include "fx_bridge.h"

#include "fx/engine/ep_probe.h"
#include "fx/image/align.h"
#include "fx/models/face_embedder.h"
#include "fx/models/yunet.h"
#include "fx/models_dl/downloader.h"
#include "fx/pipeline/face_swap_pipeline.h"
#include "fx/pipeline/segmentation_pipeline.h"
#include "fx/worker.h"

#include <obs-module.h>
#include <util/platform.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
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

/* Directory containing the running plugin .so (e.g.
 * <plugin-dir>/bin/64bit). The GPU ORT build must land here: the
 * plugin's $ORIGIN RPATH loads libonnxruntime from this dir, and the
 * GPU build's provider libs resolve via their own $ORIGIN.
 * obs_get_module_binary_path returns the full path of the module FILE
 * (obs-module.c: mod.bin_path = bstrdup(path) of the .so), so the last
 * component is stripped. Empty when the module is not registered. */
std::string pluginBinDir()
{
	const char *p = obs_get_module_binary_path(obs_current_module());
	std::string path = p ? p : "";
	size_t slash = path.find_last_of('/');
	return slash != std::string::npos ? path.substr(0, slash)
					  : std::string();
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
	int tierInEffect = 0; // (int)fx::SegTier + 1
	fx::MaskParams params;

	uint64_t seenSeq = 0;
	std::vector<uint8_t> u8;
	bool loggedFirstMask = false;

	int64_t fpsWinStart = 0;
	uint64_t fpsCount = 0;
	uint64_t fpsLast = 0;

	/* Face swap. swapM serializes UI-thread setSource/params against
	 * the worker thread's swapPipeline->process() (captured by the
	 * processor lambda as a shared_ptr so it outlives the bridge). */
	std::shared_ptr<fx::FaceSwapPipeline> swapPipeline;
	std::shared_ptr<std::mutex> swapM = std::make_shared<std::mutex>();
	std::unique_ptr<fx::FaceEmbedder> embedder; // lazy, cached
	std::unique_ptr<fx::YuNet> embedDetector;   // lazy, cached
	fx::FaceSwapParams swapParams;
	std::vector<float> pendingLatent; // source set before pipeline built
	bool faceswapEnabled = false;
	/* Background mode != off, mirrored for the worker's swap-then-seg
	 * processor (shared_ptr like swapM: the capturing lambda must
	 * outlive the bridge). When false the fs processor skips the
	 * segmentation run and publishes a null-mask bundle. */
	std::shared_ptr<std::atomic<bool>> bgActive =
		std::make_shared<std::atomic<bool>>(true);

	/* 2-stage face-swap download chain (inswapper_128_fp16 ->
	 * w600k_r50): fsId is the current stage ("" when idle). fsChainM
	 * guards both fields: pumpFaceswapDownload runs on the render
	 * thread (cam_fx_faceswap_available is polled every frame while
	 * face swap is on) AND on the UI thread (status getters), and
	 * fsId is a heap-allocated string — an unsynchronized read/write
	 * pair is a use-after-free. */
	std::string fsId;
	bool fsChain = false;
	std::mutex fsChainM;

	/* Kind of the most recent download started via this bridge: the
	 * provider payload only takes effect after an OBS restart, so the
	 * Done status for it carries a restart hint. */
	bool dlLastProvider = false;

	uint64_t seenFrameSeq = 0;
	std::vector<uint8_t> u8frame;
	int frameW = 0;
	int frameH = 0;
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

/* Cache path of a "model"-kind manifest entry. Falls back to the
 * conventional filename when the manifest lacks the id. */
std::string modelCachePath(cam_fx *fx, const char *id, const char *fallback)
{
	for (const auto &m : fx->manifest)
		if (m.id == id && m.kind == "model")
			return cacheDir("models") + "/" + m.file;
	return cacheDir("models") + "/" + fallback;
}

std::string inswapperFp16CachePath(cam_fx *fx)
{
	return modelCachePath(fx, "inswapper_128_fp16",
			      "inswapper_128_fp16.onnx");
}

std::string inswapperFp32CachePath(cam_fx *fx)
{
	return modelCachePath(fx, "inswapper_128", "inswapper_128.onnx");
}

/* fp16 is the default download (1.7x faster on CUDA, fp32 graph IO and
 * the same embedded emap = zero-change drop-in); the fp32 model remains
 * as fallback for users who already downloaded it. */
std::string inswapperCachePath(cam_fx *fx)
{
	const std::string fp16 = inswapperFp16CachePath(fx);
	if (fileExists(fp16))
		return fp16;
	return inswapperFp32CachePath(fx);
}

std::string arcfaceCachePath(cam_fx *fx)
{
	return modelCachePath(fx, "w600k_r50", "w600k_r50.onnx");
}

bool faceswapModelsPresent(cam_fx *fx)
{
	return (fileExists(inswapperFp16CachePath(fx)) ||
		fileExists(inswapperFp32CachePath(fx))) &&
	       fileExists(arcfaceCachePath(fx));
}

std::string yunetBundledPath()
{
	char *p = obs_module_file("models/face_detection_yunet_2023mar.onnx");
	std::string out = p ? p : "";
	if (p)
		bfree(p);
	return out;
}

/* Installs the worker processor matching the current state: swap-then-
 * segment when face swap is enabled and built, segment-only otherwise.
 * Pipelines (and the swap mutex) are captured by shared_ptr so a
 * hot-swap never tears a mid-call pipeline. */
void installProcessor(cam_fx *fx)
{
	if (!fx->worker || !fx->pipeline)
		return;
	std::shared_ptr<fx::SegmentationPipeline> seg = fx->pipeline;
	if (fx->faceswapEnabled && fx->swapPipeline) {
		std::shared_ptr<fx::FaceSwapPipeline> swap = fx->swapPipeline;
		std::shared_ptr<std::mutex> m = fx->swapM;
		std::shared_ptr<std::atomic<bool>> bg = fx->bgActive;
		fx->worker->setProcessor(
			[seg, swap, m, bg](const fx::Frame &frame) {
				fx::Frame work = frame;
				{
					std::lock_guard<std::mutex> lk(*m);
					swap->process(work);
				}
				fx::WorkerResult r;
				/* The mask only feeds the background
				 * composite — skip the segmentation run
				 * (2-6ms/frame) when background is off; the
				 * fs-only composite draws the swapped frame
				 * directly and tolerates the null mask. */
				if (bg->load(std::memory_order_relaxed))
					r.mask = seg->process(work);
				r.frame = std::make_shared<const fx::Frame>(
					std::move(work));
				return r;
			});
	} else {
		fx->worker->setProcessor([seg](const fx::Frame &frame) {
			fx::WorkerResult r;
			r.mask = seg->process(frame);
			return r;
		});
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
			fx->threads, /*tryCuda=*/true);
		p->setMaskParams(fx->params);
		fx->pipeline = p;
		fx->tierInEffect = (int)tier + 1;
		static const char *names[] = {"lite", "standard", "quality"};
		blog(LOG_INFO,
		     "obs-cam-effects: pipeline tier in effect: %s (backend: %s)",
		     names[(int)tier],
		     fx::EpProbe::backendName(p->usesCuda()));
		installProcessor(fx);
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

/* Starts a background download for the given manifest entry id.
 * Returns 0 on start, -1 if busy/invalid. */
int startDownloadById(cam_fx *fx, const char *id)
{
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
		/* A provider without an extract list would download a
		 * useless 240MB tgz and report Done (manifest formatting
		 * drift) — refuse. */
		if (entry->extract.empty())
			return -1;
		/* The enabling payload is the GPU build's main
		 * libonnxruntime.so.1.28.0 — it must land in the plugin
		 * BIN DIR (the $ORIGIN the plugin .so resolves
		 * libonnxruntime.so.1 from), replacing the bundled CPU
		 * build. Overlay mode: the bin dir also holds the plugin
		 * .so itself, so the downloader's whole-dir swap would
		 * wipe it; per-file rename is also safe against the
		 * running process's mapped lib. */
		std::string binDir = pluginBinDir();
		if (binDir.empty()) {
			blog(LOG_WARNING,
			     "obs-cam-effects: provider download refused: plugin bin dir unknown");
			return -1;
		}
		blog(LOG_INFO,
		     "obs-cam-effects: provider extracts into plugin bin dir %s",
		     binDir.c_str());
		req.destPath = cacheDir("providers") + "/" + entry->id + ".tgz";
		req.extractMembers = entry->extract;
		req.extractDestDir = binDir;
		req.extractOverlay = true;
		/* The provider tar nests the .so files under
		 * <pkg>/lib/: strip 2 components so they land flat. */
		req.stripComponents = 2;
	} else {
		req.destPath = cacheDir("models") + "/" + entry->file;
	}
	try {
		fx->downloader->start(req);
		fx->dlLastProvider = entry->kind == "provider";
		blog(LOG_INFO, "obs-cam-effects: download started: %s",
		     entry->id.c_str());
		return 0;
	} catch (...) {
		return -1;
	}
}

/* Drives the 2-stage face-swap chain: when inswapper_128_fp16
 * completes, w600k_r50 starts (skipped when its file already exists).
 * Called from the status getters, so every UI poll advances the chain.
 * On error the chain stops with fsId on the failed stage.
 * Threading: fsId/fsChain are copied/updated under fsChainM; the lock
 * is never held across Downloader calls. */
void pumpFaceswapDownload(cam_fx *fx)
{
	std::string stage;
	{
		std::lock_guard<std::mutex> lk(fx->fsChainM);
		if (!fx->fsChain)
			return;
		stage = fx->fsId;
	}
	fx::models_dl::State st = fx->downloader->state();
	if (st == fx::models_dl::State::Error) {
		std::lock_guard<std::mutex> lk(fx->fsChainM);
		fx->fsChain = false;
		return;
	}
	if (st != fx::models_dl::State::Done)
		return;
	if (stage == "inswapper_128_fp16") {
		if (!fileExists(arcfaceCachePath(fx)) &&
		    startDownloadById(fx, "w600k_r50") == 0) {
			std::lock_guard<std::mutex> lk(fx->fsChainM);
			fx->fsId = "w600k_r50";
			return;
		}
		std::lock_guard<std::mutex> lk(fx->fsChainM);
		fx->fsChain = false;
		fx->fsId.clear();
	} else if (stage == "w600k_r50") {
		std::lock_guard<std::mutex> lk(fx->fsChainM);
		fx->fsChain = false;
		fx->fsId.clear();
	}
}

/* Packs the (possibly padded) BGRA rows into an fx::Frame and hands it
 * to the worker. Shared by both submit entry points. */
void submitFrame(cam_fx_t *fx, const uint8_t *bgra, int w, int h,
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
			[p](const fx::Frame &f) {
				fx::WorkerResult r;
				r.mask = p->process(f);
				return r;
			});
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
	submitFrame(fx, bgra, w, h, linesize);
}

void cam_fx_submit_full(cam_fx_t *fx, const uint8_t *bgra, int w, int h,
			int linesize)
{
	submitFrame(fx, bgra, w, h, linesize);
}

int cam_fx_try_get_mask(cam_fx_t *fx, const uint8_t **px, int *w, int *h,
			uint64_t *seq)
{
	uint64_t s = 0;
	auto m = fx->worker->tryGetLatestMask(s);
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

int cam_fx_has_mask(cam_fx_t *fx)
{
	if (!fx)
		return 0;
	uint64_t s = 0;
	return fx->worker->tryGetLatestMask(s) ? 1 : 0;
}

void cam_fx_set_tier(cam_fx_t *fx, int tier)
{
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

int cam_fx_gpu_build_present(cam_fx_t *fx)
{
	if (!fx)
		return 0;
	try {
		std::string binDir = pluginBinDir();
		if (binDir.empty())
			return 0;
		/* The bundled CPU build ships libonnxruntime.so.1.28.0
		 * under the SAME filename the GPU build uses, so its
		 * presence proves nothing. The unambiguous marker is the
		 * CUDA provider lib, which only lands in the bin dir via
		 * the provider download's extract list. */
		std::string marker = "libonnxruntime_providers_cuda.so";
		for (const auto &m : fx->manifest)
			if (m.id == "ort_cuda_ep_1.28.0" &&
			    m.kind == "provider" && !m.file.empty())
				marker = m.file;
		return fileExists(binDir + "/" + marker) ? 1 : 0;
	} catch (...) {
		return 0;
	}
}

uint8_t *cam_fx_watermark_badge_rgba(int *w, int *h)
{
	if (!w || !h)
		return nullptr;
	try {
		std::vector<uint8_t> px = fx::renderWatermarkBadgeRGBA(*w, *h);
		if (px.empty())
			return nullptr;
		uint8_t *buf = (uint8_t *)bmalloc(px.size());
		if (!buf)
			return nullptr;
		memcpy(buf, px.data(), px.size());
		return buf;
	} catch (...) {
		return nullptr;
	}
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
	return startDownloadById(fx, id);
}

int cam_fx_download_state(cam_fx_t *fx, char *buf, int buf_len,
			  double *progress)
{
	if (!fx)
		return -1;
	try {
		pumpFaceswapDownload(fx);
	} catch (...) {
	}
	if (buf && buf_len > 0) {
		fx::models_dl::State st = fx->downloader->state();
		/* The provider payload (GPU ORT build) only takes effect
		 * on the next process start — say so explicitly. */
		const bool providerDone =
			st == fx::models_dl::State::Done && fx->dlLastProvider;
		snprintf(buf, (size_t)buf_len, "%s",
			 providerDone
				 ? "done — restart OBS to enable GPU acceleration"
				 : fx::models_dl::stateName(st));
	}
	if (progress)
		*progress = fx->downloader->progress();
	return 0;
}

int cam_fx_download_error(cam_fx_t *fx, char *buf, int buf_len)
{
	if (!fx)
		return -1;
	if (buf && buf_len > 0)
		snprintf(buf, (size_t)buf_len, "%s",
			 fx->downloader->error().c_str());
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

int cam_fx_backend(cam_fx_t *fx, char *buf, int buf_len)
{
	if (!fx)
		return -1;
	if (buf && buf_len > 0)
		snprintf(buf, (size_t)buf_len, "%s",
			 fx::EpProbe::backendName(fx->pipeline &&
						  fx->pipeline->usesCuda()));
	return 0;
}

void cam_fx_set_background_active(cam_fx_t *fx, int active)
{
	if (!fx)
		return;
	fx->bgActive->store(active != 0, std::memory_order_relaxed);
}

int cam_fx_faceswap_available(cam_fx_t *fx)
{
	if (!fx)
		return 0;
	try {
		pumpFaceswapDownload(fx);
		return faceswapModelsPresent(fx) &&
			       fx::EpProbe::cudaAvailable()
			       ? 1
			       : 0;
	} catch (...) {
		return 0;
	}
}

int cam_fx_faceswap_missing(cam_fx_t *fx, char *buf, int buf_len)
{
	if (!fx)
		return 1;
	const char *reason = "";
	try {
		if (!faceswapModelsPresent(fx))
			reason = "models not downloaded";
		else if (!fx::EpProbe::cudaAvailable())
			reason = "no GPU acceleration";
	} catch (...) {
		reason = "unavailable";
	}
	if (buf && buf_len > 0)
		snprintf(buf, (size_t)buf_len, "%s", reason);
	return reason[0] ? 1 : 0;
}

int cam_fx_faceswap_models_present(cam_fx_t *fx)
{
	if (!fx)
		return 0;
	try {
		return faceswapModelsPresent(fx) ? 1 : 0;
	} catch (...) {
		return 0;
	}
}

void cam_fx_faceswap_set_enabled(cam_fx_t *fx, int enabled)
{
	if (!fx || !fx->worker)
		return;
	try {
		if ((enabled != 0) == fx->faceswapEnabled &&
		    (!enabled || fx->swapPipeline))
			return; // idempotent: state unchanged
		if (enabled) {
			if (!fx->swapPipeline) {
				if (cam_fx_faceswap_available(fx) != 1)
					return;
				std::string yunet = yunetBundledPath();
				if (yunet.empty())
					return;
				auto swap =
					std::make_shared<fx::FaceSwapPipeline>(
						yunet, inswapperCachePath(fx),
						arcfaceCachePath(fx),
						fx->threads,
						/*tryCuda=*/true);
				swap->setParams(fx->swapParams);
				/* Missing source is fine: process() no-
				 * swaps until one is set (hasSource). */
				if (!fx->pendingLatent.empty())
					swap->setSourceEmbedding(
						fx->pendingLatent);
				fx->swapPipeline = std::move(swap);
				blog(LOG_INFO,
				     "obs-cam-effects: face swap pipeline built");
			}
			fx->faceswapEnabled = true;
		} else {
			fx->faceswapEnabled = false;
		}
		installProcessor(fx);
	} catch (const std::exception &e) {
		blog(LOG_WARNING,
		     "obs-cam-effects: face swap enable failed: %s", e.what());
	} catch (...) {
		blog(LOG_WARNING, "obs-cam-effects: face swap enable failed");
	}
}

int cam_fx_faceswap_set_source(cam_fx_t *fx, const char *image_path)
{
	if (!fx || !image_path || image_path[0] == '\0')
		return -1;
	try {
		if (!faceswapModelsPresent(fx))
			return -1;
		std::string yunet = yunetBundledPath();
		if (yunet.empty())
			return -1;
		if (!fx->embedder)
			fx->embedder = std::make_unique<fx::FaceEmbedder>(
				arcfaceCachePath(fx), inswapperCachePath(fx),
				fx->threads);
		if (!fx->embedDetector)
			fx->embedDetector =
				std::make_unique<fx::YuNet>(yunet, fx->threads);
		std::vector<float> latent =
			fx->embedder->embedFromImageFile(image_path,
							 *fx->embedDetector);
		if (latent.empty()) {
			blog(LOG_WARNING,
			     "obs-cam-effects: face swap source: no usable face in %s",
			     image_path);
			return -1;
		}
		fx->pendingLatent = latent;
		if (fx->swapPipeline) {
			std::lock_guard<std::mutex> lk(*fx->swapM);
			fx->swapPipeline->setSourceEmbedding(std::move(latent));
		}
		blog(LOG_INFO, "obs-cam-effects: face swap source set: %s",
		     image_path);
		return 0;
	} catch (const std::exception &e) {
		blog(LOG_WARNING,
		     "obs-cam-effects: face swap source failed: %s",
		     e.what());
		return -1;
	} catch (...) {
		return -1;
	}
}

void cam_fx_faceswap_set_params(cam_fx_t *fx, float intensity,
				float sharpness, int preserve_mouth)
{
	if (!fx)
		return;
	try {
		fx->swapParams.intensity = intensity;
		fx->swapParams.sharpness = sharpness;
		fx->swapParams.mouthPreserve = (float)preserve_mouth / 100.0f;
		if (fx->swapPipeline) {
			std::lock_guard<std::mutex> lk(*fx->swapM);
			fx->swapPipeline->setParams(fx->swapParams);
		}
	} catch (...) {
	}
}

int cam_fx_try_get_frame(cam_fx_t *fx, const uint8_t **bgra, int *w, int *h,
			 uint64_t *seq)
{
	if (!fx || !fx->worker || !bgra || !w || !h || !seq)
		return 0;
	try {
		uint64_t s = 0;
		fx::WorkerResult r = fx->worker->tryGetLatest(s);
		if (!r.frame)
			return 0;
		if (s != fx->seenFrameSeq) {
			fx->u8frame = r.frame->bgra;
			fx->frameW = r.frame->width;
			fx->frameH = r.frame->height;
			fx->seenFrameSeq = s;
		}
		*bgra = fx->u8frame.data();
		*w = fx->frameW;
		*h = fx->frameH;
		*seq = s;
		return 1;
	} catch (...) {
		return 0;
	}
}

int cam_fx_start_faceswap_download(cam_fx_t *fx)
{
	if (!fx)
		return -1;
	try {
		fx::models_dl::State st = fx->downloader->state();
		if (st == fx::models_dl::State::Downloading ||
		    st == fx::models_dl::State::Verifying ||
		    st == fx::models_dl::State::Extracting)
			return -1; // another download is running
		{
			std::lock_guard<std::mutex> lk(fx->fsChainM);
			fx->fsChain = false;
			fx->fsId.clear();
		}
		/* No usable inswapper (neither fp16 nor fp32): fetch the
		 * fp16 default. */
		if (!fileExists(inswapperFp16CachePath(fx)) &&
		    !fileExists(inswapperFp32CachePath(fx))) {
			if (startDownloadById(fx, "inswapper_128_fp16") != 0)
				return -1;
			std::lock_guard<std::mutex> lk(fx->fsChainM);
			fx->fsChain = true;
			fx->fsId = "inswapper_128_fp16";
			return 0;
		}
		if (!fileExists(arcfaceCachePath(fx))) {
			if (startDownloadById(fx, "w600k_r50") != 0)
				return -1;
			std::lock_guard<std::mutex> lk(fx->fsChainM);
			fx->fsChain = true;
			fx->fsId = "w600k_r50";
			return 0;
		}
		return 0; // both already present: nothing to do
	} catch (...) {
		return -1;
	}
}

int cam_fx_faceswap_download_state(cam_fx_t *fx, char *id_buf, int id_len,
				   char *state_buf, int state_len,
				   double *progress)
{
	if (!fx)
		return -1;
	try {
		pumpFaceswapDownload(fx);
	} catch (...) {
	}
	std::string stage;
	{
		std::lock_guard<std::mutex> lk(fx->fsChainM);
		stage = fx->fsId;
	}
	if (id_buf && id_len > 0)
		snprintf(id_buf, (size_t)id_len, "%s", stage.c_str());
	if (state_buf && state_len > 0)
		snprintf(state_buf, (size_t)state_len, "%s",
			 fx::models_dl::stateName(fx->downloader->state()));
	if (progress)
		*progress = fx->downloader->progress();
	return 0;
}

} // extern "C"
