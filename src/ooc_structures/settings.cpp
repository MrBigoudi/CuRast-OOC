#include "settings.h"

#include "CuRast.h"
#include "kernels/ooc/GpuVersionAllocator.h"
#include <toml/toml.hpp>

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

    uint64_t free_byte, total_byte = 0;
    CURuntime::assertCudaSuccess(cuMemGetInfo(&free_byte, &total_byte));
    DEVICE_TOTAL_MEMORY = double(total_byte);
    DEVICE_AVAILABLE_MEMORY = double(free_byte);
}



void OocSimLodSettings::init_miscellaneous(){
    /// Miscellaneous
    IS_RUNNING_IN_PARALLEL = init_field<bool>("IS_RUNNING_IN_PARALLEL", false);
    NUMBER_OF_FRAMES_BETWEEN_DATA_EXCHANGE = init_field<uint32_t>("NUMBER_OF_FRAMES_BETWEEN_DATA_EXCHANGE", 60);
    MEASURE_TIMINGS = init_field<bool>("MEASURE_TIMINGS", false);
    IS_USING_GPU_VERSION = init_field<bool>("IS_USING_GPU_VERSION", false);
    DERIVE_AUTOMATIC_PROPERTIES = init_field<bool>("DERIVE_AUTOMATIC_PROPERTIES", false);
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
    DISPLAY_VISIBLE_NODES_AABB_AT_STARTUP = init_field<bool>("DIFFERENTIATE_VISIBLE_NODES_AABB_AT_STARTUP", false);
    CuRastSettings::showVisibleNodes = DISPLAY_VISIBLE_NODES_AABB_AT_STARTUP;
}


void OocSimLodSettings::init_batch_sizes(){
    /// Batch sizes
    BATCHES_LIST_SIZE = init_field<uint32_t>("BATCHES_LIST_SIZE", 1'024);
    MAX_POINTS_PER_BATCHES = init_field<uint32_t>("MAX_POINTS_PER_BATCHES", 1'000'000);
    MIN_BATCHES_PER_LOAD = init_field<uint32_t>("MIN_BATCHES_PER_LOAD", 1);
    MAX_BATCHES_PER_LOAD = init_field<uint32_t>( "MAX_BATCHES_PER_LOAD", 1);
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
    MAX_NB_NODES = init_field<uint32_t>("MAX_NB_NODES", 1'000'000);
    MAX_NB_NODES_TO_EXCHANGE = init_field<uint32_t>("MAX_NB_NODES_TO_EXCHANGE", 128);
    MAX_NB_SPILLING_POINTS = init_field<uint32_t>("MAX_NB_SPILLING_POINTS", 1'000'000);
    MAX_NB_BACKLOG_VOXELS = init_field<uint32_t>("MAX_NB_BACKLOG_VOXELS", 1'000'000);

    MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE = init_field<uint32_t>("MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE", 256);
    MAX_NB_RENDERED_POINTS = init_field<uint32_t>("MAX_NB_RENDERED_POINTS", 1'000'000);
    MAX_NB_RENDERED_VOXELS = init_field<uint32_t>("MAX_NB_RENDERED_VOXELS", 1'000'000);
    MAX_NB_COUNT_SPLIT_ITERATION = init_field<uint32_t>("MAX_NB_COUNT_SPLIT_ITERATION", 8);
}

void OocSimLodSettings::init_default(){
    // 1Gb fixed
    MAX_NB_NODES = 1'048'576;
    MAX_NB_RENDERED_POINTS = 8'388'608;
    MAX_NB_RENDERED_VOXELS = 4'194'304;
    MAX_NB_SPILLING_POINTS = 8'388'608;
    MAX_NB_BACKLOG_VOXELS = 8'388'608;
    MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE = 128;
    MAX_POINTS_PER_LEAF = 65'536;
    MAX_POINTS_PER_BATCHES = 1'048'576;
    double fixed_memory = 1.25 * 1024 * 1024 * 1024;

    double available_memory = (DEVICE_AVAILABLE_MEMORY * 0.9) - fixed_memory - sizeof(CGlobalVariables);
    
    // Allocable memory
    double allocable_memory = 0.75 * available_memory;
    double chunk_memory = 0.8 * allocable_memory;
    double chunk_size = 64 * sizeof(CChunk); // 64 because nbChunks should be 64 * LRU_CACHE
    LRU_UPDATES_CACHE_SIZE = std::bit_floor(uint32_t(chunk_memory / chunk_size));
    NB_ALLOCABLE_CHUNKS = 64 * LRU_UPDATES_CACHE_SIZE;
    NB_ALLOCABLE_GRIDS = LRU_UPDATES_CACHE_SIZE;
    NB_ALLOCABLE_NODES = 4 * LRU_UPDATES_CACHE_SIZE;
    chunk_memory = (NB_ALLOCABLE_CHUNKS * (sizeof(CChunk) + 4)) + sizeof(CAllocatorPool<CChunk>);
    double grids_memory = (NB_ALLOCABLE_GRIDS * (sizeof(COccupancyGrid) + 4)) + sizeof(CAllocatorPool<COccupancyGrid>);
    double nodes_memory = (NB_ALLOCABLE_NODES * (sizeof(COctreeNode) + 4)) + sizeof(CAllocatorPool<COctreeNode>);
    double lru_memory = LRU_UPDATES_CACHE_SIZE * sizeof(CIdAABB);

    // Other properties
    available_memory = (available_memory - chunk_memory - grids_memory - nodes_memory - lru_memory);
    double batches_memory = 0.125 * available_memory;
    double batches_factor = (8 + 9 * MAX_POINTS_PER_BATCHES);
    MAX_BATCHES_PER_OCTREE_UPDATE = std::bit_floor(uint32_t(batches_memory / batches_factor));
    batches_memory = MAX_BATCHES_PER_OCTREE_UPDATE * batches_factor;

    double exchangeable_memory = 0.25 * available_memory;
    double exchangeable_factor =  44 + 16 * (MAX_POINTS_PER_LEAF + NB_POINTS_PER_CHUNK * MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE);
    MAX_NB_NODES_TO_EXCHANGE = std::bit_floor(uint32_t(exchangeable_memory / exchangeable_factor));
    exchangeable_memory = MAX_NB_NODES_TO_EXCHANGE * exchangeable_factor;

    double spilling_memory = available_memory - batches_memory - exchangeable_memory;
    uint32_t new_spilling = 0.5 * std::bit_floor(uint32_t(spilling_memory / 64));
    MAX_NB_SPILLING_POINTS += new_spilling;
    MAX_NB_BACKLOG_VOXELS += new_spilling;
    
    available_memory = (available_memory - batches_memory - exchangeable_memory - spilling_memory);
}



/// Initialises the settings
void OocSimLodSettings::init(){
    init_device_properties();
    init_miscellaneous();
    init_ui_params();
    init_batch_sizes();
    init_octree_properties();
    init_gpu_version_buffers();
    
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

    println("");
    println("Miscellaneous:");
    println("    - IS_RUNNING_IN_PARALLEL: {}", IS_RUNNING_IN_PARALLEL);
    println("    - NUMBER_OF_FRAMES_BETWEEN_DATA_EXCHANGE: {}", NUMBER_OF_FRAMES_BETWEEN_DATA_EXCHANGE);
    println("    - MEASURE_TIMINGS: {}", MEASURE_TIMINGS);
    println("    - IS_USING_GPU_VERSION: {}", IS_USING_GPU_VERSION);

    println("");
    println("Batch sizes:");
    println("    - BATCHES_LIST_SIZE: {}", BATCHES_LIST_SIZE);
    println("    - MAX_POINTS_PER_BATCHES: {}", MAX_POINTS_PER_BATCHES);
    println("    - MIN_BATCHES_PER_LOAD: {}", MIN_BATCHES_PER_LOAD);
    println("    - MAX_BATCHES_PER_LOAD: {}", MAX_BATCHES_PER_LOAD);
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
    println("    - MAX_NB_COUNT_SPLIT_ITERATION: {}", MAX_NB_COUNT_SPLIT_ITERATION);

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
