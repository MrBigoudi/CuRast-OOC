#pragma once

#include "kernels/HostDeviceInterface.h"

struct CuRastSettings{
	static inline bool showBoundingBoxes = false;
	static inline bool bruteForceRendering = false;
	static inline int32_t debugLodToRender = -1;
	static inline int32_t voxelsPointsPerAxis = 1;
	static inline float minPixelSpan = 64.;
	static inline bool voxelsDebugColor = false;
	static inline bool freeOldOctreeMemoryOnGPU = false;
	static inline bool autoFreeOldOctreeMemoryOnGPU = true;
	static inline bool storeOctree = false;
	static inline bool loadOctree = false;

	/// The maximum number of batches that should be used per octree update
	static inline int32_t maxBatchesPerUpdate = 10;
	// static inline int32_t maxBatchesPerUpdate = 100;
	/// The maximum number of points in a batch
	static inline int32_t maxBatchSize = 100'000;
	// static inline int32_t maxBatchSize = 1'000'000;

	static inline bool enableEDL = true;
	static inline bool enableFrustumCulling = true;
	static inline bool hideGUI = false;

	static inline bool showKernelInfos = false;
	static inline bool showMemoryInfos = false;
	static inline bool showTimingInfos = false;
	static inline bool showStats = false;
	static inline bool showOverlay = true;
	static inline bool showInset = false;
	static inline bool showBenchmarking = false;
	static inline int supersamplingFactor = 1;

	static inline bool enableLinearInterpolation = true;
	static inline bool enableMipMapping = true;
	static inline float threshold = 0.0f;
	static inline bool freezeFrustum = false;
	static inline bool enableSSAO = false;
	static inline bool enableDiffuseLighting = false;
	static inline bool disableInstancing = false;
	static inline bool enableObjectPicking = false;
	static inline shared_ptr<string> requestScreenshot = nullptr; // Set to path of screenshot, or empty string for auto path
	static inline vec4 background = {1.0f, 1.0f, 1.0f, 1.0f};


	static inline DisplayAttribute displayAttribute = DisplayAttribute::TEXTURE;
	static inline bool showWireframe = false;

	// static inline uint32_t rasterizer = RASTERIZER_VULKAN_INDEXPULLING_INSTANCED;
	// static inline uint32_t rasterizer = RASTERIZER_VISBUFFER_INDEXED;
	static inline uint32_t rasterizer = RASTERIZER_VISBUFFER_INSTANCED;
	// static inline uint32_t rasterizer = RASTERIZER_VULKAN_INDEXPULLING_VISBUFFER;
	// static inline uint32_t rasterizer = RASTERIZER_VISBUFFER_CLUSTERS;

	static inline bool benchmark_load_sponza = false;
	static inline bool benchmark_load_lantern = false;
};

// Enabling this makes CuRast allocate memory for geometry with the Vulkan API instead of CUDA.
// - Needs to be enabled to render things in Vulkan.
// - It's still shared to CUDA because LargeGlbLoader.h uses CUDA for streaming mesh data to GPU. 
// - It's off by default because allocating in Vulkan vs. CUDA has different performance implications.
// - From observations, we assume that the Vulkan buffer implicitly enables compression. 
//   This makes some scenarios faster (e.g. uncompressed geometry) but others slower (resolve).
// - Explicitly enabling compressed CUDA buffers seems to equalize the performance.
// - For benchmarking, we enable it for Vulkan measurements and disable it for CUDA measuerements.
// #define USE_VULKAN_SHARED_MEMORY