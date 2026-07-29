cmake_minimum_required(VERSION 3.28...3.30)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

project(${_name} VERSION ${_version})

option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" OFF)
option(ENABLE_QT "Use Qt functionality" OFF)

include(compilerconfig)
include(defaults)
include(helpers)

add_library(${CMAKE_PROJECT_NAME} MODULE)

find_package(libobs REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs)

if(ENABLE_FRONTEND_API)
  find_package(obs-frontend-api REQUIRED)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
endif()

if(ENABLE_QT)
  find_package(Qt6 COMPONENTS Widgets Core)
  target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets)
  target_compile_options(
    ${CMAKE_PROJECT_NAME}
    PRIVATE $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-quoted-include-in-framework-header -Wno-comma>
  )
  set_target_properties(
    ${CMAKE_PROJECT_NAME}
    PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON
  )
endif()

target_sources(${CMAKE_PROJECT_NAME} PRIVATE src/plugin-main.c src/cam-effects-filter.c src/fx_bridge.cpp)

set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${_name})

# --- ONNX Runtime 1.28.0 (CPU, Linux x86-64, prebuilt) ---
set(ORT_VERSION 1.28.0)
set(ORT_ROOT ${CMAKE_BINARY_DIR}/onnxruntime)
if(NOT EXISTS "${ORT_ROOT}/lib/libonnxruntime.so.${ORT_VERSION}")
  file(DOWNLOAD
    "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
    ${CMAKE_BINARY_DIR}/ort-${ORT_VERSION}.tgz
    EXPECTED_HASH SHA256=a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407
    SHOW_PROGRESS)
  file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/ort-${ORT_VERSION}.tgz
       DESTINATION ${CMAKE_BINARY_DIR})
  file(RENAME ${CMAKE_BINARY_DIR}/onnxruntime-linux-x64-${ORT_VERSION} ${ORT_ROOT})
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ORT_ROOT}/lib/libonnxruntime.so.${ORT_VERSION}
  INTERFACE_INCLUDE_DIRECTORIES ${ORT_ROOT}/include)

# --- Bundled segmentation model: PP-HumanSeg v2 lite (Apache-2.0) ---
set(FX_MODEL_DIR ${CMAKE_BINARY_DIR}/models)
if(NOT EXISTS "${FX_MODEL_DIR}/pphumanseg_fp32.onnx")
  file(DOWNLOAD
    "https://raw.githubusercontent.com/royshil/obs-backgroundremoval/main/models/pphumanseg_fp32.onnx"
    ${FX_MODEL_DIR}/pphumanseg_fp32.onnx
    EXPECTED_HASH SHA256=6913acd125be55fd08e76072ed464146925223392954c437f3aa700e4da84b17)
  file(DOWNLOAD
    "https://raw.githubusercontent.com/royshil/obs-backgroundremoval/main/models/pphumanseg_fp32.onnx.license"
    ${FX_MODEL_DIR}/pphumanseg_fp32.onnx.license
    EXPECTED_HASH SHA256=363552428d64011b9242e0889f90e06cf2c30370908a2a9641dbdd8d37a5f2a6)
endif()

find_package(Threads REQUIRED)

# --- libfx: OBS-free core library (all inference code lives here from Plan 2) ---
add_library(fx STATIC
  src/fx/version.cpp
  src/fx/engine/ort_backend.cpp
  src/fx/image/ops.cpp
  src/fx/models/pp_humanseg.cpp
  src/fx/models_dl/downloader.cpp
  src/fx/pipeline/segmentation_pipeline.cpp
  src/fx/worker.cpp
)
target_include_directories(fx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(fx PUBLIC cxx_std_17)
set_target_properties(fx PROPERTIES POSITION_INDEPENDENT_CODE ON
                                    BUILD_RPATH ${ORT_ROOT}/lib)
target_link_libraries(fx PUBLIC onnxruntime Threads::Threads)
target_compile_definitions(fx PUBLIC FX_MODEL_PATH="${FX_MODEL_DIR}/pphumanseg_fp32.onnx")
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE fx)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
  BUILD_RPATH "\$ORIGIN;${ORT_ROOT}/lib"
  INSTALL_RPATH "$ORIGIN")

# --- Tests ---
include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  include(GoogleTest)
  add_executable(fx_tests
    tests/test_version.cpp
    tests/test_ort_backend.cpp
    tests/test_image_ops.cpp
    tests/test_pp_humanseg.cpp
    tests/test_worker.cpp
    tests/test_golden.cpp
    tests/test_soak.cpp
    tests/test_downloader.cpp
  )
  target_link_libraries(fx_tests PRIVATE fx GTest::gtest_main)
  target_compile_features(fx_tests PRIVATE cxx_std_20)
  set_target_properties(fx_tests PROPERTIES BUILD_RPATH ${ORT_ROOT}/lib)
  gtest_discover_tests(fx_tests)
endif()
