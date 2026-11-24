#include "settings.h"

#include "CuRast.h"
#include "kernels/ooc/GpuVersionAllocator.h"
#include <toml/toml.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

/// Fixed element declarations
static std::string settings_file = format("{}/settings.toml", PROJECT_SOURCE_DIR);
static std::string temporary_dir = format("{}/build/tmp", PROJECT_SOURCE_DIR);
const char* OocSimLodSettings::SETTINGS_TOML_FILE = settings_file.c_str();
const char* OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY = temporary_dir.c_str();

/// The parsed toml data
const toml::value SETTINGS_TOML_DATA = toml::parse(OocSimLodSettings::SETTINGS_TOML_FILE);
/// Fill a value from a toml config file
template<typename T>
T init_field(const std::string& toml_entry, const T default_value) {
    return toml::find_or<T>(SETTINGS_TOML_DATA, toml_entry, default_value);
}


void OocSimLodSettings::init_device_properties(){
    /// Device properties
    CUdevice device;
    cuCtxGetDevice(&device);
    int value = 0;
    CURuntime::assertCudaSuccess(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    DEVICE_ATTRIBUTE_NB_SM = (uint32_t)value;
    CURuntime::assertCudaSuccess(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device));
    DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM = (uint32_t)value;
    CURuntime::assertCudaSuccess(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
    DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = (uint32_t)value;
    CURuntime::assertCudaSuccess(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, device));
    DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_SM = (uint32_t)value;
    CURuntime::assertCudaSuccess(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device));
    DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = (uint32_t)value;
    CURuntime::assertCudaSuccess(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device));
    DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = (uint32_t)value;
    CURuntime::assertCudaSuccess(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device));
    DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = (uint32_t)value;
    CURuntime::assertCudaSuccess(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
    DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = (uint32_t)value;
    CURuntime::assertCudaSuccess(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device));
    DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = (uint32_t)value;
    CURuntime::assertCudaSuccess(cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device));
    DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = (uint32_t)value;
    DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE = 
        (DEVICE_ATTRIBUTE_NB_SM * DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK - 1) 
        / DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    ;

    init_memory_properties();
}

void OocSimLodSettings::init_memory_properties(){
    uint64_t free_byte = 0, total_byte = 0;
    CURuntime::assertCudaSuccess(cuMemGetInfo(&free_byte, &total_byte));
    DEVICE_TOTAL_MEMORY = double(total_byte);
    DEVICE_AVAILABLE_MEMORY = double(free_byte);

    // Host physical memory, and what is available of it (free + reclaimable page cache)
#if defined(_WIN32)
    MEMORYSTATUSEX status = {};
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    HOST_TOTAL_MEMORY = double(status.ullTotalPhys);
    HOST_AVAILABLE_MEMORY = double(status.ullAvailPhys);
#else
    HOST_TOTAL_MEMORY = double(sysconf(_SC_PHYS_PAGES)) * double(sysconf(_SC_PAGE_SIZE));
    HOST_AVAILABLE_MEMORY = double(sysconf(_SC_AVPHYS_PAGES)) * double(sysconf(_SC_PAGE_SIZE));
    // `MemAvailable' also counts the page cache the kernel can drop (the free pages alone underestimate it)
    std::ifstream meminfo("/proc/meminfo");
    std::string key, unit;
    uint64_t value_kb = 0;
    while(meminfo >> key >> value_kb >> unit){
        if(key == "MemAvailable:"){
            HOST_AVAILABLE_MEMORY = double(value_kb) * 1024.0;
            break;
        }
    }
#endif
}



void OocSimLodSettings::init_miscellaneous(){
    /// Miscellaneous
    MEASURE_TIMINGS = init_field<bool>("MEASURE_TIMINGS", false);
    DERIVE_AUTOMATIC_PROPERTIES = init_field<bool>("DERIVE_AUTOMATIC_PROPERTIES", false);
    IS_USING_NN = init_field<bool>("IS_USING_NN", false);
}

void OocSimLodSettings::init_ui_params(){
    /// Ui initial parameters
    SHOW_BOUNDING_BOXES_AT_STARTUP = init_field<bool>("SHOW_BOUNDING_BOXES_AT_STARTUP", false);
    CuRastSettings::showBoundingBoxes = SHOW_BOUNDING_BOXES_AT_STARTUP;
    BRUTE_FORCE_RENDERING_AT_STARTUP = init_field<bool>("BRUTE_FORCE_RENDERING_AT_STARTUP", false);
    CuRastSettings::bruteForceRendering = BRUTE_FORCE_RENDERING_AT_STARTUP;
    DEBUG_LOD_TO_RENDER_AT_STARTUP = init_field<int32_t>("DEBUG_LOD_TO_RENDER_AT_STARTUP", -1);
    CuRastSettings::debugLodToRender = DEBUG_LOD_TO_RENDER_AT_STARTUP;
    VOXELS_POINTS_PER_AXIS_AT_STARTUP = init_field<int32_t>("VOXELS_POINTS_PER_AXIS_AT_STARTUP", 1);
    CuRastSettings::voxelsPointsPerAxis = VOXELS_POINTS_PER_AXIS_AT_STARTUP;
    MIN_PIXEL_SPAN_AT_STARTUP = init_field<float>("MIN_PIXEL_SPAN_AT_STARTUP", 64.);
    CuRastSettings::minPixelSpan = MIN_PIXEL_SPAN_AT_STARTUP;
    USE_VOXELS_DEBUG_COLOR_AT_STARTUP = init_field<bool>("USE_VOXELS_DEBUG_COLOR_AT_STARTUP", false);
    CuRastSettings::voxelsDebugColor = USE_VOXELS_DEBUG_COLOR_AT_STARTUP;
    USE_AUTO_FREE_OLD_OCTREE_ON_GPU_AT_STARTUP = init_field<bool>("USE_AUTO_FREE_OLD_OCTREE_ON_GPU_AT_STARTUP", true);
    CuRastSettings::autoFreeOldOctreeMemoryOnGPU = USE_AUTO_FREE_OLD_OCTREE_ON_GPU_AT_STARTUP;
    // The toml entry is `DISPLAY_VISIBLE_NODES_AABB_AT_STARTUP' (it used to be read under another
    // name, so the toml value was ignored; the old name is still read as a fallback)
    DISPLAY_VISIBLE_NODES_AABB_AT_STARTUP = init_field<bool>("DISPLAY_VISIBLE_NODES_AABB_AT_STARTUP",
        init_field<bool>("DIFFERENTIATE_VISIBLE_NODES_AABB_AT_STARTUP", false));
    CuRastSettings::showVisibleNodes = DISPLAY_VISIBLE_NODES_AABB_AT_STARTUP;
}


void OocSimLodSettings::init_batch_sizes(){
    /// Batch sizes
    BATCHES_LIST_SIZE = init_field<uint32_t>("BATCHES_LIST_SIZE", 1'024);
    MAX_POINTS_PER_BATCHES = init_field<uint32_t>("MAX_POINTS_PER_BATCHES", 1'000'000);
    MIN_BATCHES_PER_LOAD = init_field<uint32_t>("MIN_BATCHES_PER_LOAD", 1);
    MAX_BATCHES_PER_LOAD = init_field<uint32_t>( "MAX_BATCHES_PER_LOAD", 1);
    MAX_LOADING_THREADS = init_field<uint32_t>("MAX_LOADING_THREADS", 0);
    MIN_BATCHES_PER_GPU_LOAD = init_field<uint32_t>( "MIN_BATCHES_PER_GPU_LOAD", 1);
    MAX_BATCHES_PER_GPU_LOAD = init_field<uint32_t>("MAX_BATCHES_PER_GPU_LOAD", 1);
    MIN_BATCHES_PER_OCTREE_UPDATE = init_field<uint32_t>("MIN_BATCHES_PER_OCTREE_UPDATE", 1);
    MAX_BATCHES_PER_OCTREE_UPDATE = init_field<uint32_t>("MAX_BATCHES_PER_OCTREE_UPDATE", 1);
    MAX_ATTEMPTS_BEFORE_IGNORING_MIN_VARIABLES = init_field<uint32_t>("MAX_ATTEMPTS_BEFORE_IGNORING_MIN_VARIABLES", 1);
}

void OocSimLodSettings::init_octree_properties(){
    /// Octree properties
    MAX_POINTS_PER_LEAF = init_field<uint32_t>("MAX_POINTS_PER_LEAF", 50'000);
    LRU_UPDATES_CACHE_SIZE = init_field<uint32_t>("LRU_UPDATES_CACHE_SIZE", 128);
    LRU_VISIBILITY_CACHE_SIZE = init_field<uint32_t>("LRU_VISIBILITY_CACHE_SIZE", 512);
    LRU_CPU_CACHE_SIZE = init_field<uint32_t>("LRU_CPU_CACHE_SIZE", 2048);
    NB_ALLOCABLE_CHUNKS = init_field<uint32_t>("NB_ALLOCABLE_CHUNKS", 32'000);
    NB_ALLOCABLE_GRIDS = init_field<uint32_t>("NB_ALLOCABLE_GRIDS", 2048);
    NB_ALLOCABLE_NODES = init_field<uint32_t>("NB_ALLOCABLE_NODES", 2048);
}


void OocSimLodSettings::init_gpu_version_buffers(){
    /// GPU Version buffers
    MAX_NB_NODES = std::bit_ceil(init_field<uint32_t>("MAX_NB_NODES", 1'000'000));
    MAX_NB_NODES_TO_EXCHANGE = init_field<uint32_t>("MAX_NB_NODES_TO_EXCHANGE", 128);
    MAX_NB_SPILLING_POINTS = init_field<uint32_t>("MAX_NB_SPILLING_POINTS", 1'000'000);
    MAX_NB_BACKLOG_VOXELS = init_field<uint32_t>("MAX_NB_BACKLOG_VOXELS", 1'000'000);

    MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE = init_field<uint32_t>("MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE", 256);
    MAX_NB_RENDERED_POINTS = init_field<uint32_t>("MAX_NB_RENDERED_POINTS", 1'000'000);
    MAX_NB_RENDERED_VOXELS = init_field<uint32_t>("MAX_NB_RENDERED_VOXELS", 1'000'000);
    MAX_NB_RENDERED_LARGE_POINTS = init_field<uint32_t>("MAX_NB_RENDERED_LARGE_POINTS", 8'388'608);
    MAX_NB_RENDERED_SIDE_VOXELS = init_field<uint32_t>("MAX_NB_RENDERED_SIDE_VOXELS", 8'388'608);
    MAX_NB_COUNT_SPLIT_ITERATION = init_field<uint32_t>("MAX_NB_COUNT_SPLIT_ITERATION", 8);
}

void OocSimLodSettings::init_automatic_sizing(){
    /// Automatic sizing (inputs of `init_default', see its comments)
    DEVICE_MEMORY_RESERVE_MB = init_field<double>("DEVICE_MEMORY_RESERVE_MB", 512.0);
    DEVICE_MEMORY_RESERVE_FRACTION = init_field<double>("DEVICE_MEMORY_RESERVE_FRACTION", 0.12);
    NN_MEMORY_RESERVE_MB = init_field<double>("NN_MEMORY_RESERVE_MB", 3072.0);
    RENDERING_MEMORY_FRACTION = init_field<double>("RENDERING_MEMORY_FRACTION", 0.25);
    NODE_IDS_MEMORY_FRACTION = init_field<double>("NODE_IDS_MEMORY_FRACTION", 0.04);
    UPDATE_MEMORY_FRACTION = init_field<double>("UPDATE_MEMORY_FRACTION", 0.25);
    EXCHANGE_MEMORY_FRACTION = init_field<double>("EXCHANGE_MEMORY_FRACTION", 0.06);
    SPILLING_POINTS_PER_UPDATE_POINT = init_field<double>("SPILLING_POINTS_PER_UPDATE_POINT", 1.5);
    BACKLOG_VOXELS_PER_UPDATE_POINT = init_field<double>("BACKLOG_VOXELS_PER_UPDATE_POINT", 1.0);
    CHUNKS_FILL_RATIO = init_field<double>("CHUNKS_FILL_RATIO", 0.75);
    GRIDS_PER_RESIDENT_NODE = init_field<double>("GRIDS_PER_RESIDENT_NODE", 0.6);
    NODES_PER_RESIDENT_NODE = init_field<double>("NODES_PER_RESIDENT_NODE", 4.0);
    TRANSIENT_NODES_PER_UPDATE_LEAF = init_field<double>("TRANSIENT_NODES_PER_UPDATE_LEAF", 6.0);
    MIN_LRU_UPDATES_CACHE_SIZE = init_field<uint32_t>("MIN_LRU_UPDATES_CACHE_SIZE", 128);
    HOST_CACHE_MEMORY_FRACTION = init_field<double>("HOST_CACHE_MEMORY_FRACTION", 0.25);
    SIMULATED_DEVICE_MEMORY_GB = init_field<double>("SIMULATED_DEVICE_MEMORY_GB", 0.0);
}

void OocSimLodSettings::init_rendering(){
    /// Rendering
    VOXEL_SPLAT_SCALE = init_field<float>("VOXEL_SPLAT_SCALE", 1.0f);
    VOXEL_SPLAT_MAX_PIXELS = init_field<uint32_t>("VOXEL_SPLAT_MAX_PIXELS", 32);
    DRAW_PENDING_BATCHES = init_field<bool>("DRAW_PENDING_BATCHES", true);
    PENDING_BATCHES_MAX_POINTS = init_field<uint32_t>("PENDING_BATCHES_MAX_POINTS", 0);
    EVICTION_RETAIN_BUDGET_FRACTION = init_field<double>("EVICTION_RETAIN_BUDGET_FRACTION", 0.25);
    EVICTION_RETAIN_NB_PASSES = std::max(1u, init_field<uint32_t>("EVICTION_RETAIN_NB_PASSES", 16));
    MAX_EVICTION_COVERAGE_WAIT_SECONDS = init_field<double>("MAX_EVICTION_COVERAGE_WAIT_SECONDS", 5.0);
}

void OocSimLodSettings::init_visibility(){
    /// Visibility
    VIS_MIN_SHARE_FRACTION = init_field<double>("VIS_MIN_SHARE_FRACTION", 0.0);
    VIS_DISTANCE_EXPONENT = init_field<double>("VIS_DISTANCE_EXPONENT", 3.0);
    VIS_DISTANCE_FLOOR_FRACTION = init_field<double>("VIS_DISTANCE_FLOOR_FRACTION", 0.01);
    VIS_RAM_CACHE_MAX_BYTES = uint64_t(init_field<double>("VIS_RAM_CACHE_MAX_MB", 1024.0) * 1024.0 * 1024.0);
    VIS_RAM_CACHE_MIN_KEEP = init_field<uint32_t>("VIS_RAM_CACHE_MIN_KEEP", 1024);
    VIS_MAX_PAUSE_SECONDS = init_field<double>("VIS_MAX_PAUSE_SECONDS", 0.5);
    DISK_IO_THREADS = init_field<uint32_t>("DISK_IO_THREADS", 0);
}

void OocSimLodSettings::init_loader(){
    /// Loader
    STATUS_READBACK_FALLBACK_SECONDS = init_field<double>("STATUS_READBACK_FALLBACK_SECONDS", 0.25);
    MAX_OUT_OF_VIEW_DEVICE_SLOTS = init_field<uint32_t>("MAX_OUT_OF_VIEW_DEVICE_SLOTS", 8);
    LOAD_AHEAD_FACTOR = init_field<uint32_t>("LOAD_AHEAD_FACTOR", 1);
    PREEMPTION_FACTOR = init_field<float>("PREEMPTION_FACTOR", 2.0f);
    LOAD_CENTER_WEIGHT = init_field<float>("LOAD_CENTER_WEIGHT", 0.5f);
    MAX_READERS_PER_FILE = init_field<uint32_t>("MAX_READERS_PER_FILE", 0);
    CANCEL_CHECK_INTERVAL = init_field<uint32_t>("CANCEL_CHECK_INTERVAL", 65'536);
    PENDING_SUMMARY_REFRESH_SECONDS = init_field<double>("PENDING_SUMMARY_REFRESH_SECONDS", 0.05);
    MAX_LOAD_ATTEMPTS = init_field<uint32_t>("MAX_LOAD_ATTEMPTS", 3);
    OVERVIEW_FALLBACK_STRIDE = init_field<uint32_t>("OVERVIEW_FALLBACK_STRIDE", 50'000);
    PREVIEW_ENABLED = init_field<bool>("PREVIEW_ENABLED", true);
    PREVIEW_NB_REGIONS = init_field<uint32_t>("PREVIEW_NB_REGIONS", 16);
    PREVIEW_POINTS_PER_REGION = init_field<uint32_t>("PREVIEW_POINTS_PER_REGION", 262'144);
    PREVIEW_PROGRESSIVE = init_field<bool>("PREVIEW_PROGRESSIVE", true);
    PREVIEW_PROGRESSIVE_FIRST_POINTS = init_field<uint32_t>("PREVIEW_PROGRESSIVE_FIRST_POINTS", 16'384);
    PREVIEW_PROGRESSIVE_INTERVAL = init_field<uint32_t>("PREVIEW_PROGRESSIVE_INTERVAL", 131'072);
    READ_AHEAD = init_field<bool>("READ_AHEAD", true);
}



namespace {
/// Breakdown of the last `init_default' (printed by `display')
std::string memoryPlan = {};

uint64_t roundDown(double value, uint64_t multiple){
    if(!(value > 0.0)){ return 0; }
    return (uint64_t(value) / multiple) * multiple;
}
uint64_t roundUp(double value, uint64_t multiple){
    if(!(value > 0.0)){ return 0; }
    return ((uint64_t(std::ceil(value)) + multiple - 1) / multiple) * multiple;
}
uint32_t toU32(uint64_t value){
    return uint32_t(std::min<uint64_t>(value, 0xFFFF'FFFFull - 1));
}
std::string mb(double bytes){
    return format("{:>9.1f} MB", bytes / (1024.0 * 1024.0));
}
}

/// Derives the buffer sizes from the memory available at startup
///
/// Principle: every device buffer allocated by `GpuVersion::initBuffers' / `initAllocators' and
/// `LoaderGpuVersion::initPreview' is accounted for explicitly, with its exact per-element size
/// (sizeof of the structures). The buffers whose size is a quality / throughput choice (rendering
/// budgets, octree update size, exchange size) get at most a share of the memory; the node ids get a
/// share too; everything left goes to the octree pools (chunks, grids, nodes) and the updates cache.
/// The pools are sized from a model of the peak number of resident nodes during an update, so that
/// they run out together, as late as possible, instead of one of them first.
///
/// The memory model only uses quantities known at startup. The values of the toml file act as:
///  - targets that are reduced when they don't fit: rendering budgets, MAX_BATCHES_PER_OCTREE_UPDATE,
///    MAX_NB_NODES_TO_EXCHANGE, VIS_RAM_CACHE_MAX_MB, PREVIEW_POINTS_PER_REGION
///  - a floor: MAX_NB_NODES (it depends on the dataset size, that can't be known here)
///  - inputs: MAX_POINTS_PER_LEAF, MAX_POINTS_PER_BATCHES, MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE, the
///    preview size, and the AUTOMATIC SIZING section
///  - ignored (derived): LRU_UPDATES_CACHE_SIZE, NB_ALLOCABLE_*, MAX_NB_SPILLING_POINTS,
///    MAX_NB_BACKLOG_VOXELS, LRU_CPU_CACHE_SIZE
void OocSimLodSettings::init_default(){
    // ===================================================================================================
    // 0. The values requested by the toml file
    // ===================================================================================================
    // `init_default' runs twice: from `init' (so that the settings are valid as soon as possible), then
    // from `GpuVersion::init' once the kernels are loaded, with the memory measured again (the code of
    // the modules and their static data are then already taken). The requested values are captured at
    // the first call: the second one starts from them, not from the values derived by the first one
    struct Requested {
        uint32_t rendered_points;
        uint32_t rendered_large_points;
        uint32_t rendered_voxels;
        uint32_t rendered_side_voxels;
        uint32_t batches_per_update;
        uint32_t nodes_to_exchange;
        uint32_t max_nb_nodes;
        uint64_t vis_ram_cache;
        uint32_t preview_points_per_region;
    };
    static const Requested requested = {
        MAX_NB_RENDERED_POINTS, MAX_NB_RENDERED_LARGE_POINTS,
        MAX_NB_RENDERED_VOXELS, MAX_NB_RENDERED_SIDE_VOXELS,
        MAX_BATCHES_PER_OCTREE_UPDATE, MAX_NB_NODES_TO_EXCHANGE, MAX_NB_NODES,
        VIS_RAM_CACHE_MAX_BYTES, PREVIEW_POINTS_PER_REGION
    };

    constexpr double MB = 1024.0 * 1024.0;

    // ===================================================================================================
    // Simulation of a smaller card (testing only)
    // ===================================================================================================
    // SIMULATED_DEVICE_MEMORY_GB > 0 (toml file): everything below is computed as on a card with that
    // much memory (e.g. 4.0 for a 4 GB card); 0 = the real card. The simulated card has the same memory already in
    // use as the real one (desktop, Vulkan, CuRast and, at the second call, the kernels):
    //   total = SIMULATED_DEVICE_MEMORY_GB, available = total - (real total - real available)
    // Only the sizes are simulated: the rest of the real card stays free, so an allocation made later
    // beyond the simulated memory would not fail as it would on the small card
    bool is_simulated = false;
    if(SIMULATED_DEVICE_MEMORY_GB > 0.0){
        const double simulated_total = SIMULATED_DEVICE_MEMORY_GB * 1024.0 * MB;
        const double in_use = DEVICE_TOTAL_MEMORY - DEVICE_AVAILABLE_MEMORY;
        if(simulated_total < DEVICE_TOTAL_MEMORY){
            DEVICE_TOTAL_MEMORY = simulated_total;
            DEVICE_AVAILABLE_MEMORY = std::max(0.0, simulated_total - in_use);
            is_simulated = true;
        }
        static bool is_reported = false;
        if(!is_reported){
            is_reported = true;
            if(is_simulated){
                println("SIMULATION: buffers sized for a {} GB card ({:.1f} GB in use, {:.1f} GB available)",
                    SIMULATED_DEVICE_MEMORY_GB, in_use / (1024.0 * MB), DEVICE_AVAILABLE_MEMORY / (1024.0 * MB));
            } else {
                println("WARNING: SIMULATED_DEVICE_MEMORY_GB ({} GB) is not smaller than the card: not simulated",
                    SIMULATED_DEVICE_MEMORY_GB);
            }
        }
    }

    const double leaf = double(std::max(1u, MAX_POINTS_PER_LEAF));
    const double leaf_chunks = std::ceil(leaf / NB_POINTS_PER_CHUNK);
    const double point_bytes = sizeof(CPoint);

    // ===================================================================================================
    // 1. Device bytes per element of every buffer (as allocated in `GpuVersion::initBuffers' and
    //    `initAllocators': a pool element also has its slot in the free stack of the pool)
    // ===================================================================================================
    // Chunk: the chunk + its free stack slot + its entry in `allocatedChunks'
    const double chunk_bytes = sizeof(CChunk) + sizeof(uint32_t) + sizeof(CChunk*);
    // Occupancy grid: the grid + its free stack slot + `gridsToInit' + `gridsToInitExchangedIndex'
    const double grid_bytes = sizeof(COccupancyGrid) + sizeof(uint32_t) + sizeof(COctreeNode*) + sizeof(uint32_t);
    // Resident node slot: the node + its free stack slot, and the buffers indexed by resident node:
    // `packedNodes', `allocationPlans', `temporaryIdBuffer2', `spillingNodes' (maxNbSpillingNodes is
    // min(spilling points, resident nodes)) and the extra spilling chunk per spilling node
    // (`spilledChunksCounter' + `spillingChunks')
    const double node_bytes = sizeof(COctreeNode) + sizeof(uint32_t) + sizeof(COctreeNode*)
        + sizeof(CAllocationPlan) + sizeof(CIdAABB) + sizeof(COctreeNode*) + sizeof(uint32_t) + sizeof(CChunk*);
    // Node id (`MAX_NB_NODES'): `relationshipMap', `nodesFlags', the visibility lists (points and
    // voxels: list, temporary list, priority), the two child masks and `temporaryIdBuffer'
    const double node_id_bytes = sizeof(CGlobalVariables::Relationship) + sizeof(uint32_t)
        + 2.0 * (2.0 * sizeof(CIdAABB) + sizeof(float)) + 2.0 * sizeof(uint8_t) + sizeof(CIdAABB);
    // Rendering budget element: the two published buffers (double buffering) + the retained copies of
    // the evicted nodes (a fraction of the budgets, see `EVICTION_RETAIN_BUDGET_FRACTION')
    const double rendered_bytes = 2.0 * point_bytes + EVICTION_RETAIN_BUDGET_FRACTION * point_bytes;
    // Exchanged node: its points (whole chunks), voxels and grid cells (the voxel chunks are capped
    // like in `GpuVersion::initConstraints'), the small per-slot arrays (ids, parents, AABBs, children,
    // counters, pointers, eviction snapshot, `temporaryNodeBuffer') and its retained entries
    const double exchanged_voxel_chunks = std::min<double>(MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE, leaf_chunks);
    const double exchanged_node_bytes = leaf_chunks * NB_POINTS_PER_CHUNK * point_bytes
        + exchanged_voxel_chunks * NB_POINTS_PER_CHUNK * (point_bytes + sizeof(uint64_t))
        + 2.0 * sizeof(CIdAABB) + sizeof(CAABB) + 3.0 * sizeof(uint32_t) + 3.0 * sizeof(void*)
        + sizeof(CIdAABB) + 2.0 * sizeof(uint32_t) + sizeof(COctreeNode*)
        + EVICTION_RETAIN_NB_PASSES * sizeof(CEvictedRetainEntry);
    // Point of an octree update: its slot in the batches + its memoised leaf (`memoizedBatchPointsNodes')
    const double batch_point_bytes = point_bytes + sizeof(uint32_t);
    // Spilled point: `spilledPoints' + `memoizedSpilledPointsNodes' (+ a spilling chunk per 1024 points)
    const double spilled_point_bytes = point_bytes + sizeof(uint32_t)
        + (sizeof(uint32_t) + sizeof(CChunk*)) / double(NB_POINTS_PER_CHUNK);
    // Backlog voxel: `backlogVoxels' + `backlogVoxelsNodes'
    const double backlog_voxel_bytes = point_bytes + sizeof(uint32_t);

    // ===================================================================================================
    // 2. Usable device memory
    // ===================================================================================================
    // DEVICE_AVAILABLE_MEMORY is measured after Vulkan / CuRast (and, at the second call, the kernels).
    // What is allocated later and can't be modelled here still needs room: kernels compiled lazily,
    // local memory of the kernels (it grows with the number of SMs), CUB temporary storage, the
    // framebuffers when the window is resized, the driver's own bookkeeping. A run on a 16 GB card
    // showed about 0.9 GB of such growth after the initialisation. Most of it grows with the card:
    // larger cards have more SMs (more local memory for the same kernels) and drive larger screens
    // (framebuffers), so the reserve is mostly proportional to the card: a small fixed part
    // (DEVICE_MEMORY_RESERVE_MB, 512 MB: driver, lazily compiled kernels) + 12% of the card
    // (DEVICE_MEMORY_RESERVE_FRACTION): 2.4 GB on a 16 GB card (2.6x the growth seen), 1 GB on a 4 GB
    // card. A large fixed part (e.g. 1.5 GB + 5%) leaves nothing usable on a 4 GB card.
    // The old fixed 2 GB + 10% of the available memory left too little room: the pools were sized from
    // a fraction of the memory without counting the rendering buffers (2.4 GB), the card ended at
    // 250 MB free. The network (`IS_USING_NN') needs its own reserve (3 GB, as before: 5 GB - 2 GB)
    const double reserve = DEVICE_MEMORY_RESERVE_MB * MB
        + DEVICE_MEMORY_RESERVE_FRACTION * DEVICE_TOTAL_MEMORY
        + (IS_USING_NN ? NN_MEMORY_RESERVE_MB * MB : 0.0)
        + sizeof(CGlobalVariables) + sizeof(CMemoryAllocator)
        + sizeof(CAllocatorPool<CChunk>) + sizeof(CAllocatorPool<COccupancyGrid>) + sizeof(CAllocatorPool<COctreeNode>);
    const double usable = std::max(0.0, DEVICE_AVAILABLE_MEMORY - reserve);

    // ===================================================================================================
    // 3. Rendering budgets (MAX_NB_RENDERED_*): the toml values, scaled down if they don't fit
    // ===================================================================================================
    // The budgets are a quality choice and they cost memory whatever the dataset: 25M + 8M points and
    // as many voxels take 2.4 GB (36 bytes per element). They get at most RENDERING_MEMORY_FRACTION
    // (25%) of the usable memory: on a 16 GB card the toml values fit as they are; on an 8 GB card they
    // are scaled to about 55% of them, which keeps several points per pixel at 1080p while leaving
    // most of the memory to the octree. The four budgets are scaled together (their ratios are a
    // rendering choice) and kept multiples of 1024
    const double requested_rendered = double(requested.rendered_points) + requested.rendered_large_points
        + requested.rendered_voxels + requested.rendered_side_voxels;
    const double rendering_scale = (requested_rendered > 0.0)
        ? std::min(1.0, RENDERING_MEMORY_FRACTION * usable / (requested_rendered * rendered_bytes))
        : 1.0;
    auto scaleBudget = [&](uint32_t budget){
        return toU32(std::max<uint64_t>(budget > 0 ? 1024 : 0, roundDown(budget * rendering_scale, 1024)));
    };
    MAX_NB_RENDERED_POINTS = scaleBudget(requested.rendered_points);
    MAX_NB_RENDERED_LARGE_POINTS = scaleBudget(requested.rendered_large_points);
    MAX_NB_RENDERED_VOXELS = scaleBudget(requested.rendered_voxels);
    MAX_NB_RENDERED_SIDE_VOXELS = scaleBudget(requested.rendered_side_voxels);
    const double rendering_memory = rendered_bytes * (double(MAX_NB_RENDERED_POINTS) + MAX_NB_RENDERED_LARGE_POINTS
        + MAX_NB_RENDERED_VOXELS + MAX_NB_RENDERED_SIDE_VOXELS);

    // ===================================================================================================
    // 4. Node ids (MAX_NB_NODES): a power of 2, NODE_IDS_MEMORY_FRACTION of the usable memory
    // ===================================================================================================
    // Unlike everything else, the ids are never recycled: MAX_NB_NODES bounds the number of nodes of
    // the whole octree, so the size of the dataset (about one node per 20-40K points). Running out
    // is fatal and can't be fixed by eviction, while an id only costs ~100 bytes (and about as much
    // on the host): 4% of the usable memory gives 4M ids on a 16 GB card (~100 billion points),
    // 1M on small cards. The automatic part is capped to MAX_AUTOMATIC_NODE_IDS = 4M: that already
    // covers datasets of ~100 billion points, and larger cards then keep the memory for the pools
    // instead of ids that would never be used (16M ids would take 1.6 GB on the device, as much on
    // the host). The toml value is a floor, also above the cap (the user may know the dataset is
    // huge). Power of 2: the bitonic sort of the visibility lists (render.cu) needs its capacity to
    // be one
    constexpr uint64_t MIN_AUTOMATIC_NODE_IDS = 1ull << 20;   // 1M
    constexpr uint64_t MAX_AUTOMATIC_NODE_IDS = 1ull << 22;   // 4M
    const uint64_t derived_ids = std::bit_floor(uint64_t(std::max(1.0, NODE_IDS_MEMORY_FRACTION * usable / node_id_bytes)));
    MAX_NB_NODES = toU32(std::max<uint64_t>(
        std::bit_ceil(uint64_t(std::max(1u, requested.max_nb_nodes))),
        std::clamp<uint64_t>(derived_ids, MIN_AUTOMATIC_NODE_IDS, MAX_AUTOMATIC_NODE_IDS)
    ));
    const double node_ids_memory = MAX_NB_NODES * node_id_bytes;

    // ===================================================================================================
    // 5. Exchange (MAX_NB_NODES_TO_EXCHANGE): the toml value, reduced to EXCHANGE_MEMORY_FRACTION
    // ===================================================================================================
    // An exchanged node costs ~2.5 MB (65536 points + 64 voxel chunks with their cells). The exchange
    // size only sets how many passes loading / storing take (they loop until done): it is a throughput
    // knob, never a limit. 6% of the usable memory keeps 256 on a 16 GB card (the value used so far)
    // and ~100 on an 8 GB one. Multiple of 8, at least 8 (on a 4 GB card, 32 slots would already take
    // 10% of the usable memory for fewer passes, which only costs throughput)
    const uint64_t exchange_fit = std::max<uint64_t>(8, roundDown(EXCHANGE_MEMORY_FRACTION * usable / exchanged_node_bytes, 8));
    MAX_NB_NODES_TO_EXCHANGE = toU32(std::max<uint64_t>(1, std::min<uint64_t>(requested.nodes_to_exchange, exchange_fit)));
    const double exchange_memory = MAX_NB_NODES_TO_EXCHANGE * exchanged_node_bytes;

    // ===================================================================================================
    // 6. Loader preview: the toml size (16 regions of 256K points = 64 MB), at most 5% of the usable
    //    memory (64 MB is negligible on a 16 GB card but 8% of a 4 GB one): fewer points per region,
    //    the preview is a subsample anyway. Multiple of 1024, at least 1024
    // ===================================================================================================
    constexpr double PREVIEW_MEMORY_FRACTION = 0.05;
    if(PREVIEW_ENABLED && PREVIEW_NB_REGIONS > 0){
        const double regions = double(std::min(PREVIEW_NB_REGIONS, 32u));
        const double points_fit = PREVIEW_MEMORY_FRACTION * usable / (regions * point_bytes);
        PREVIEW_POINTS_PER_REGION = toU32(std::min<uint64_t>(requested.preview_points_per_region,
            std::max<uint64_t>(1024, roundDown(points_fit, 1024))));
    }
    const double preview_memory = PREVIEW_ENABLED
        ? double(std::min(PREVIEW_NB_REGIONS, 32u)) * PREVIEW_POINTS_PER_REGION * point_bytes
        : 0.0;

    // ===================================================================================================
    // 7. Octree update size and pools, sized together
    // ===================================================================================================
    // Everything else depends on the number of points of one octree update,
    //   P = MAX_BATCHES_PER_OCTREE_UPDATE x MAX_POINTS_PER_BATCHES:
    //
    //  - spilling buffer: in steady growth a leaf spills every ~0.75 x MAX_POINTS_PER_LEAF new points
    //    (its children start a quarter full, for 4 occupied octants) and then moves all its
    //    MAX_POINTS_PER_LEAF points: P new points spill ~1.33 P points, all of them in the first
    //    count/split iteration. SPILLING_POINTS_PER_UPDATE_POINT = 1.5 (the old automatic value was
    //    0.75 P, which only fits while few leaves are close to full)
    //  - backlog voxels: a split resamples ~one leaf of points into a new grid (~20-40K voxels for a
    //    surface), and the new points add cells to their ancestors: ~0.6-0.8 P voxels per update.
    //    BACKLOG_VOXELS_PER_UPDATE_POINT = 1.0
    //    Both buffers hold at least 4 leaves (an update of a few points can still split full leaves)
    //  - nodes brought in by the update (loaded from the host, or created by the splits):
    //    ~1.33 P / MAX_POINTS_PER_LEAF splits x 4 children, plus the leaves loaded to receive the
    //    points: TRANSIENT_NODES_PER_UPDATE_LEAF = 6 nodes per MAX_POINTS_PER_LEAF points (3072 for
    //    32 batches of 1M points)
    //
    // The pools hold the resident nodes: the updates cache (L nodes kept between updates) plus the
    // transient nodes of the update in progress (T). Per resident node they need:
    //  - chunks: a node has either its points (leaf) or its voxels (inner node), at most one leaf of
    //    them. Leaves fill from ~25% (just split) to 100% and inner nodes hold ~25-60% of a leaf of
    //    voxels: the average is ~50-60%. CHUNKS_FILL_RATIO = 0.75 (+ 1 partially filled last chunk)
    //    leaves a ~1.3x margin: over thousands of nodes the average hardly varies, the risk is a
    //    dataset denser than assumed (raise it then, the UI shows the peak number of chunks)
    //  - occupancy grids: one per inner node. Inner nodes are ~25-35% of the nodes for surfaces (4
    //    children each), more for sparse or linear data, and the cache keeps the ancestors first.
    //    GRIDS_PER_RESIDENT_NODE = 0.6. The old automatic value (2 x L = 8192 grids, 2 GB) was 4 times
    //    more than the inner nodes of L + T nodes can need, while a grid costs as much as 16 chunks
    //  - node slots: NODES_PER_RESIDENT_NODE = 4. A node slot costs ~1/70 of a chunk, so the node
    //    pool is made too large to ever be the first one to run out
    //
    // So the pools cost (L + T) x (c chunks + g grids + n nodes), and the update itself costs
    // P x (batch + spill + backlog). The update size is chosen first (the toml value, capped to
    // UPDATE_MEMORY_FRACTION = 25% of the usable memory), then the cache gets what is left:
    //   L = memory left / (cost per resident node) - T
    // If L ends smaller than T (or than MIN_LRU_UPDATES_CACHE_SIZE), the cache can't even keep what
    // one update touched: every update would evict the nodes the next one needs, and the transient
    // nodes would take most of the pools. The update is then made smaller (one batch less at a time:
    // T and the update buffers shrink, L grows) until L >= max(T, MIN_LRU_UPDATES_CACHE_SIZE)
    const double chunks_per_resident_node = CHUNKS_FILL_RATIO * leaf_chunks + 1.0;
    const double resident_node_bytes = chunks_per_resident_node * chunk_bytes
        + GRIDS_PER_RESIDENT_NODE * grid_bytes
        + NODES_PER_RESIDENT_NODE * node_bytes
        + 2.0 * sizeof(CIdAABB);   // updates cache (x 2 for its prefix scan), per cached node

    const double fixed_memory = rendering_memory + node_ids_memory + exchange_memory + preview_memory;
    const double update_point_bytes = batch_point_bytes
        + SPILLING_POINTS_PER_UPDATE_POINT * spilled_point_bytes
        + BACKLOG_VOXELS_PER_UPDATE_POINT * backlog_voxel_bytes;
    const double batch_points = double(std::max(1u, MAX_POINTS_PER_BATCHES));
    // `memoizedBatchPointsNodes' has one entry per point of an update: P must fit in 32 bits
    const uint32_t max_batches_u32 = uint32_t(std::max(1.0, std::floor(double(0xFFFF'FFFFu) / batch_points)));
    uint32_t nb_batches = std::min({
        std::max(1u, requested.batches_per_update),
        max_batches_u32,
        uint32_t(std::max(1.0, UPDATE_MEMORY_FRACTION * usable / (batch_points * update_point_bytes)))
    });

    uint64_t spilling = 0, backlog = 0, transient = 0, lru = 0;
    double update_memory = 0.0;
    while(true){
        const double update_points = nb_batches * batch_points;
        spilling = std::max<uint64_t>(4 * uint64_t(leaf), roundUp(SPILLING_POINTS_PER_UPDATE_POINT * update_points, 1024));
        backlog = std::max<uint64_t>(4 * uint64_t(leaf), roundUp(BACKLOG_VOXELS_PER_UPDATE_POINT * update_points, 1024));
        update_memory = update_points * batch_point_bytes + spilling * spilled_point_bytes + backlog * backlog_voxel_bytes;
        transient = uint64_t(std::ceil(TRANSIENT_NODES_PER_UPDATE_LEAF * std::ceil(update_points / leaf)));
        const double pools_memory = usable - fixed_memory - update_memory;
        const double resident_nodes = std::max(0.0, pools_memory / resident_node_bytes);
        // Multiple of 64: no power of 2 is needed by the cache, rounding down to one would waste up
        // to half of the pools (the old `bit_floor'); a coarser multiple would waste most of the pools
        // of a small card (256 nodes are ~250 MB)
        lru = roundDown(resident_nodes - double(transient), 64);
        if((lru >= MIN_LRU_UPDATES_CACHE_SIZE && lru >= transient) || nb_batches == 1){ break; }
        nb_batches--;
    }
    if(lru < std::max<uint64_t>(MIN_LRU_UPDATES_CACHE_SIZE, transient)){
        println("WARNING: not enough device memory for the octree: updates cache of {} nodes for {} transient nodes"
            " (usable memory {}); the pools may run out", lru, transient, mb(usable));
        lru = std::max<uint64_t>(lru, 64);
    }

    MAX_BATCHES_PER_OCTREE_UPDATE = nb_batches;
    MIN_BATCHES_PER_OCTREE_UPDATE = std::min(MIN_BATCHES_PER_OCTREE_UPDATE, MAX_BATCHES_PER_OCTREE_UPDATE);
    MAX_NB_SPILLING_POINTS = toU32(spilling);
    MAX_NB_BACKLOG_VOXELS = toU32(backlog);
    LRU_UPDATES_CACHE_SIZE = toU32(lru);

    // Pools of the L + T resident nodes. Rounded up (to whole blocks of the init kernels): at most a
    // few MB more than planned, taken from the reserve
    const double resident_nodes = double(lru + transient);
    NB_ALLOCABLE_CHUNKS = toU32(roundUp(resident_nodes * chunks_per_resident_node, 1024));
    NB_ALLOCABLE_GRIDS = toU32(roundUp(resident_nodes * GRIDS_PER_RESIDENT_NODE, 64));
    NB_ALLOCABLE_NODES = toU32(roundUp(resident_nodes * NODES_PER_RESIDENT_NODE, 256));
    const double pools_memory = NB_ALLOCABLE_CHUNKS * chunk_bytes + NB_ALLOCABLE_GRIDS * grid_bytes
        + NB_ALLOCABLE_NODES * node_bytes + 2.0 * LRU_UPDATES_CACHE_SIZE * sizeof(CIdAABB);

    // ===================================================================================================
    // 8. Host memory
    // ===================================================================================================
    // The host cache (`HostStorageNode::init') pins LRU_CPU_CACHE_SIZE + 2 x MAX_NB_NODES_TO_EXCHANGE
    // nodes of MAX_POINTS_PER_LEAF points + voxels + cell indices (40 bytes per point: 2.6 MB per
    // node). Pinned memory can't be swapped out, so it is limited to HOST_CACHE_MEMORY_FRACTION (25%)
    // of the physical memory, and to half of what is available at startup. The cache is a cache: a
    // smaller one only means more disk reads, never a failure (the pool holds exactly the cache and
    // the two exchange passes). Multiple of 256, at least 256 nodes
    const double host_node_bytes = leaf * (2.0 * point_bytes + sizeof(uint64_t));
    const double host_cache_budget = std::min(HOST_CACHE_MEMORY_FRACTION * HOST_TOTAL_MEMORY, 0.5 * HOST_AVAILABLE_MEMORY);
    const double host_cache_nodes = host_cache_budget / host_node_bytes - 2.0 * MAX_NB_NODES_TO_EXCHANGE;
    LRU_CPU_CACHE_SIZE = toU32(std::max<uint64_t>(256, roundDown(host_cache_nodes, 256)));
    // The RAM cache of the visibility updates (regular memory): the toml value, at most 10% of the
    // physical memory
    VIS_RAM_CACHE_MAX_BYTES = std::min<uint64_t>(requested.vis_ram_cache, uint64_t(0.10 * HOST_TOTAL_MEMORY));
    // Other host memory, for the breakdown: pinned staging of the rendering buffers, pinned preview,
    // per node id arrays (parents, AABBs, mutex, visible lists: ~100 bytes)
    const double host_cache_memory = (double(LRU_CPU_CACHE_SIZE) + 2.0 * MAX_NB_NODES_TO_EXCHANGE) * host_node_bytes;
    const double host_staging_memory = point_bytes * (double(MAX_NB_RENDERED_POINTS) + MAX_NB_RENDERED_LARGE_POINTS
        + MAX_NB_RENDERED_VOXELS + MAX_NB_RENDERED_SIDE_VOXELS);

    // ===================================================================================================
    // Breakdown (printed by `display')
    // ===================================================================================================
    const double planned = fixed_memory + update_memory + pools_memory;
    memoryPlan = format(
        "{}"
        "    device available {} | reserve {} | usable {}\n"
        "    - rendering buffers   {}  (budgets x{:.2f} of the toml values)\n"
        "    - node ids            {}  ({} ids)\n"
        "    - exchange            {}  ({} nodes)\n"
        "    - loader preview      {}\n"
        "    - octree update       {}  ({} batches, {} spilling points, {} backlog voxels)\n"
        "    - pools               {}  (cache {} + transient {} nodes: {} chunks, {} grids, {} nodes)\n"
        "    planned {} | left for the reserve {}\n"
        "    host: pinned cache {} ({} nodes), pinned rendering staging {}, visibility RAM cache {}",
        is_simulated ? format("    SIMULATED {} GB card\n", SIMULATED_DEVICE_MEMORY_GB) : std::string(),
        mb(DEVICE_AVAILABLE_MEMORY), mb(reserve), mb(usable),
        mb(rendering_memory), rendering_scale,
        mb(node_ids_memory), MAX_NB_NODES,
        mb(exchange_memory), MAX_NB_NODES_TO_EXCHANGE,
        mb(preview_memory),
        mb(update_memory), MAX_BATCHES_PER_OCTREE_UPDATE, MAX_NB_SPILLING_POINTS, MAX_NB_BACKLOG_VOXELS,
        mb(pools_memory), LRU_UPDATES_CACHE_SIZE, transient, NB_ALLOCABLE_CHUNKS, NB_ALLOCABLE_GRIDS, NB_ALLOCABLE_NODES,
        mb(planned), mb(DEVICE_AVAILABLE_MEMORY - planned),
        mb(host_cache_memory), LRU_CPU_CACHE_SIZE, mb(host_staging_memory), mb(double(VIS_RAM_CACHE_MAX_BYTES))
    );
}



/// Initialises the settings
void OocSimLodSettings::init(){
    init_device_properties();
    init_miscellaneous();
    init_ui_params();
    init_batch_sizes();
    init_octree_properties();
    init_gpu_version_buffers();
    init_automatic_sizing();
    init_rendering();
    init_visibility();
    init_loader();

    // Called again by `GpuVersion::init' once its kernels are loaded, with the memory measured again
    if(DERIVE_AUTOMATIC_PROPERTIES){
        init_default();
    }
}


/// Display the settings
void OocSimLodSettings::display(){
    println("//////////////////////////////////////////////////////////");
	println("//////////////////////// Settings ////////////////////////");
	println("//////////////////////////////////////////////////////////");
	
    println("");
    println("Fixed:");
    println("    - SETTINGS_TOML_FILE: {}", SETTINGS_TOML_FILE);
    println("    - TEMPORARY_NODE_STORAGE_DIRECTORY: {}", TEMPORARY_NODE_STORAGE_DIRECTORY);
    println("    - NB_POINTS_PER_CHUNK: {}", NB_POINTS_PER_CHUNK);
    println("    - GRID_SIZE_PER_DIMENSION: {}", GRID_SIZE_PER_DIMENSION);

    println("");
    println("Device properties:");
    println("    - DEVICE_ATTRIBUTE_NB_SM: {}", DEVICE_ATTRIBUTE_NB_SM);
    println("    - DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM: {}", DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM);
    println("    - DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: {}", DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    println("    - DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_SM: {}", DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_SM);
    println("    - DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: {}", DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
    println("    - DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: {}", DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
    println("    - DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: {}", DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
    println("    - DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: {}", DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
    println("    - DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: {}", DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
    println("    - DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: {}", DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
    println("    - DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE: {}", DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE);
    println("    - DEVICE_TOTAL_MEMORY: {:.1f} MB", DEVICE_TOTAL_MEMORY / (1024.0 * 1024.0));
    println("    - DEVICE_AVAILABLE_MEMORY: {:.1f} MB", DEVICE_AVAILABLE_MEMORY / (1024.0 * 1024.0));
    println("    - HOST_TOTAL_MEMORY: {:.1f} MB", HOST_TOTAL_MEMORY / (1024.0 * 1024.0));
    println("    - HOST_AVAILABLE_MEMORY: {:.1f} MB", HOST_AVAILABLE_MEMORY / (1024.0 * 1024.0));

    println("");
    println("Miscellaneous:");
    println("    - MEASURE_TIMINGS: {}", MEASURE_TIMINGS);
    println("    - DERIVE_AUTOMATIC_PROPERTIES: {}", DERIVE_AUTOMATIC_PROPERTIES);
    println("    - IS_USING_NN: {}", IS_USING_NN);

    println("");
    println("Batch sizes:");
    println("    - BATCHES_LIST_SIZE: {}", BATCHES_LIST_SIZE);
    println("    - MAX_POINTS_PER_BATCHES: {}", MAX_POINTS_PER_BATCHES);
    println("    - MIN_BATCHES_PER_LOAD: {}", MIN_BATCHES_PER_LOAD);
    println("    - MAX_BATCHES_PER_LOAD: {}", MAX_BATCHES_PER_LOAD);
    println("    - MAX_LOADING_THREADS: {}", MAX_LOADING_THREADS);
    println("    - MIN_BATCHES_PER_GPU_LOAD: {}", MIN_BATCHES_PER_GPU_LOAD);
    println("    - MAX_BATCHES_PER_GPU_LOAD: {}", MAX_BATCHES_PER_GPU_LOAD);
    println("    - MIN_BATCHES_PER_OCTREE_UPDATE: {}", MIN_BATCHES_PER_OCTREE_UPDATE);
    println("    - MAX_BATCHES_PER_OCTREE_UPDATE: {}", MAX_BATCHES_PER_OCTREE_UPDATE);
    println("    - MAX_ATTEMPTS_BEFORE_IGNORING_MIN_VARIABLES: {}", MAX_ATTEMPTS_BEFORE_IGNORING_MIN_VARIABLES);

    println("");
    println("Octree properties:");
    println("    - MAX_POINTS_PER_LEAF: {}", MAX_POINTS_PER_LEAF);
    println("    - LRU_UPDATES_CACHE_SIZE: {}", LRU_UPDATES_CACHE_SIZE);
    println("    - LRU_VISIBILITY_CACHE_SIZE: {}", LRU_VISIBILITY_CACHE_SIZE);
    println("    - LRU_CPU_CACHE_SIZE: {}", LRU_CPU_CACHE_SIZE);
    println("    - NB_ALLOCABLE_CHUNKS: {}", NB_ALLOCABLE_CHUNKS);
    println("    - NB_ALLOCABLE_GRIDS: {}", NB_ALLOCABLE_GRIDS);
    println("    - NB_ALLOCABLE_NODES: {}", NB_ALLOCABLE_NODES);

    println("");
    println("GPU version properties:");
    println("    - MAX_NB_NODES: {}", MAX_NB_NODES);
    println("    - MAX_NB_NODES_TO_EXCHANGE: {}", MAX_NB_NODES_TO_EXCHANGE);
    println("    - MAX_NB_SPILLING_POINTS: {}", MAX_NB_SPILLING_POINTS);
    println("    - MAX_NB_BACKLOG_VOXELS: {}", MAX_NB_BACKLOG_VOXELS);
    println("    - MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE: {}", MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE);
    println("    - MAX_NB_RENDERED_POINTS: {}", MAX_NB_RENDERED_POINTS);
    println("    - MAX_NB_RENDERED_VOXELS: {}", MAX_NB_RENDERED_VOXELS);
    println("    - MAX_NB_RENDERED_LARGE_POINTS: {}", MAX_NB_RENDERED_LARGE_POINTS);
    println("    - MAX_NB_RENDERED_SIDE_VOXELS: {}", MAX_NB_RENDERED_SIDE_VOXELS);
    println("    - MAX_NB_COUNT_SPLIT_ITERATION: {}", MAX_NB_COUNT_SPLIT_ITERATION);

    println("");
    println("Automatic sizing{}:", DERIVE_AUTOMATIC_PROPERTIES ? "" : " (not used: DERIVE_AUTOMATIC_PROPERTIES = false)");
    println("    - DEVICE_MEMORY_RESERVE_MB: {}", DEVICE_MEMORY_RESERVE_MB);
    println("    - DEVICE_MEMORY_RESERVE_FRACTION: {}", DEVICE_MEMORY_RESERVE_FRACTION);
    println("    - NN_MEMORY_RESERVE_MB: {}", NN_MEMORY_RESERVE_MB);
    println("    - RENDERING_MEMORY_FRACTION: {}", RENDERING_MEMORY_FRACTION);
    println("    - NODE_IDS_MEMORY_FRACTION: {}", NODE_IDS_MEMORY_FRACTION);
    println("    - UPDATE_MEMORY_FRACTION: {}", UPDATE_MEMORY_FRACTION);
    println("    - EXCHANGE_MEMORY_FRACTION: {}", EXCHANGE_MEMORY_FRACTION);
    println("    - SPILLING_POINTS_PER_UPDATE_POINT: {}", SPILLING_POINTS_PER_UPDATE_POINT);
    println("    - BACKLOG_VOXELS_PER_UPDATE_POINT: {}", BACKLOG_VOXELS_PER_UPDATE_POINT);
    println("    - CHUNKS_FILL_RATIO: {}", CHUNKS_FILL_RATIO);
    println("    - GRIDS_PER_RESIDENT_NODE: {}", GRIDS_PER_RESIDENT_NODE);
    println("    - NODES_PER_RESIDENT_NODE: {}", NODES_PER_RESIDENT_NODE);
    println("    - TRANSIENT_NODES_PER_UPDATE_LEAF: {}", TRANSIENT_NODES_PER_UPDATE_LEAF);
    println("    - MIN_LRU_UPDATES_CACHE_SIZE: {}", MIN_LRU_UPDATES_CACHE_SIZE);
    println("    - HOST_CACHE_MEMORY_FRACTION: {}", HOST_CACHE_MEMORY_FRACTION);
    println("    - SIMULATED_DEVICE_MEMORY_GB: {}", SIMULATED_DEVICE_MEMORY_GB);
    if(DERIVE_AUTOMATIC_PROPERTIES && !memoryPlan.empty()){
        println("  Memory plan:");
        println("{}", memoryPlan);
    }

    println("");
    println("Rendering:");
    println("    - VOXEL_SPLAT_SCALE: {}", VOXEL_SPLAT_SCALE);
    println("    - VOXEL_SPLAT_MAX_PIXELS: {}", VOXEL_SPLAT_MAX_PIXELS);
    println("    - DRAW_PENDING_BATCHES: {}", DRAW_PENDING_BATCHES);
    println("    - PENDING_BATCHES_MAX_POINTS: {}", PENDING_BATCHES_MAX_POINTS);
    println("    - EVICTION_RETAIN_BUDGET_FRACTION: {}", EVICTION_RETAIN_BUDGET_FRACTION);
    println("    - EVICTION_RETAIN_NB_PASSES: {}", EVICTION_RETAIN_NB_PASSES);
    println("    - MAX_EVICTION_COVERAGE_WAIT_SECONDS: {}", MAX_EVICTION_COVERAGE_WAIT_SECONDS);

    println("");
    println("Visibility:");
    println("    - VIS_MIN_SHARE_FRACTION: {}", VIS_MIN_SHARE_FRACTION);
    println("    - VIS_DISTANCE_EXPONENT: {}", VIS_DISTANCE_EXPONENT);
    println("    - VIS_DISTANCE_FLOOR_FRACTION: {}", VIS_DISTANCE_FLOOR_FRACTION);
    println("    - VIS_RAM_CACHE_MAX_BYTES: {} ({:.1f} MB)", VIS_RAM_CACHE_MAX_BYTES, double(VIS_RAM_CACHE_MAX_BYTES) / (1024.0 * 1024.0));
    println("    - VIS_RAM_CACHE_MIN_KEEP: {}", VIS_RAM_CACHE_MIN_KEEP);
    println("    - VIS_MAX_PAUSE_SECONDS: {}", VIS_MAX_PAUSE_SECONDS);
    println("    - DISK_IO_THREADS: {}", DISK_IO_THREADS);

    println("");
    println("Loader:");
    println("    - STATUS_READBACK_FALLBACK_SECONDS: {}", STATUS_READBACK_FALLBACK_SECONDS);
    println("    - MAX_OUT_OF_VIEW_DEVICE_SLOTS: {}", MAX_OUT_OF_VIEW_DEVICE_SLOTS);
    println("    - LOAD_AHEAD_FACTOR: {}", LOAD_AHEAD_FACTOR);
    println("    - PREEMPTION_FACTOR: {}", PREEMPTION_FACTOR);
    println("    - LOAD_CENTER_WEIGHT: {}", LOAD_CENTER_WEIGHT);
    println("    - MAX_READERS_PER_FILE: {}", MAX_READERS_PER_FILE);
    println("    - CANCEL_CHECK_INTERVAL: {}", CANCEL_CHECK_INTERVAL);
    println("    - PENDING_SUMMARY_REFRESH_SECONDS: {}", PENDING_SUMMARY_REFRESH_SECONDS);
    println("    - MAX_LOAD_ATTEMPTS: {}", MAX_LOAD_ATTEMPTS);
    println("    - OVERVIEW_FALLBACK_STRIDE: {}", OVERVIEW_FALLBACK_STRIDE);
    println("    - PREVIEW_ENABLED: {}", PREVIEW_ENABLED);
    println("    - PREVIEW_NB_REGIONS: {}", PREVIEW_NB_REGIONS);
    println("    - PREVIEW_POINTS_PER_REGION: {}", PREVIEW_POINTS_PER_REGION);
    println("    - PREVIEW_PROGRESSIVE: {}", PREVIEW_PROGRESSIVE);
    println("    - PREVIEW_PROGRESSIVE_FIRST_POINTS: {}", PREVIEW_PROGRESSIVE_FIRST_POINTS);
    println("    - PREVIEW_PROGRESSIVE_INTERVAL: {}", PREVIEW_PROGRESSIVE_INTERVAL);
    println("    - READ_AHEAD: {}", READ_AHEAD);

    println("");
    println("UI initial settings:");
    println("    - SHOW_BOUNDING_BOXES_AT_STARTUP: {}", SHOW_BOUNDING_BOXES_AT_STARTUP);
    println("    - BRUTE_FORCE_RENDERING_AT_STARTUP: {}", BRUTE_FORCE_RENDERING_AT_STARTUP);
    println("    - DEBUG_LOD_TO_RENDER_AT_STARTUP: {}", DEBUG_LOD_TO_RENDER_AT_STARTUP);
    println("    - VOXELS_POINTS_PER_AXIS_AT_STARTUP: {}", VOXELS_POINTS_PER_AXIS_AT_STARTUP);
    println("    - MIN_PIXEL_SPAN_AT_STARTUP: {}", MIN_PIXEL_SPAN_AT_STARTUP);
    println("    - USE_VOXELS_DEBUG_COLOR_AT_STARTUP: {}", USE_VOXELS_DEBUG_COLOR_AT_STARTUP);
    println("    - USE_AUTO_FREE_OLD_OCTREE_ON_GPU_AT_STARTUP: {}", USE_AUTO_FREE_OLD_OCTREE_ON_GPU_AT_STARTUP);
    println("    - DISPLAY_VISIBLE_NODES_AABB_AT_STARTUP: {}", DISPLAY_VISIBLE_NODES_AABB_AT_STARTUP);

	println("\n//////////////////////////////////////////////////////////");
	println("//////////////////////////////////////////////////////////");
	println("//////////////////////////////////////////////////////////\n");
}