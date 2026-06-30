#include "settings.h"

#include "CuRast.h"
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

///////////////////////////////////////////////////////////////////////////////
///////////////////////// STATIC ELEMENTS DECLARATION /////////////////////////
///////////////////////////////////////////////////////////////////////////////

bool OocSimLodSettings::IS_RUNNING_IN_PARALLEL;
uint32_t OocSimLodSettings::NUMBER_OF_FRAMES_BETWEEN_DATA_EXCHANGE;
bool OocSimLodSettings::MEASURE_TIMINGS;

uint32_t OocSimLodSettings::BATCHES_LIST_SIZE;
uint32_t OocSimLodSettings::MAX_POINTS_PER_BATCHES;
uint32_t OocSimLodSettings::MIN_BATCHES_PER_LOAD;
uint32_t OocSimLodSettings::MAX_BATCHES_PER_LOAD;
uint32_t OocSimLodSettings::MIN_BATCHES_PER_GPU_LOAD;
uint32_t OocSimLodSettings::MAX_BATCHES_PER_GPU_LOAD;
uint32_t OocSimLodSettings::MIN_BATCHES_PER_OCTREE_UPDATE;
uint32_t OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE;
uint32_t OocSimLodSettings::MAX_ATTEMPTS_BEFORE_IGNORING_MIN_VARIABLES;

uint32_t OocSimLodSettings::MAX_POINTS_PER_LEAF;
uint32_t OocSimLodSettings::LRU_UPDATES_CACHE_SIZE;
uint32_t OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE;

bool OocSimLodSettings::SHOW_BOUNDING_BOXES_AT_STARTUP;
bool OocSimLodSettings::BRUTE_FORCE_RENDERING_AT_STARTUP;
int32_t OocSimLodSettings::DEBUG_LOD_TO_RENDER_AT_STARTUP;
int32_t OocSimLodSettings::VOXELS_POINTS_PER_AXIS_AT_STARTUP;
float OocSimLodSettings::MIN_PIXEL_SPAN_AT_STARTUP;
bool OocSimLodSettings::USE_VOXELS_DEBUG_COLOR_AT_STARTUP;
bool OocSimLodSettings::USE_AUTO_FREE_OLD_OCTREE_ON_GPU_AT_STARTUP;
bool OocSimLodSettings::DIFFERENTIATE_VISIBLE_NODES_AABB_AT_STARTUP;





/// Initialises the settings
void OocSimLodSettings::init(){
    /// Miscellaneous
    IS_RUNNING_IN_PARALLEL = init_field<bool>("IS_RUNNING_IN_PARALLEL", false);
    NUMBER_OF_FRAMES_BETWEEN_DATA_EXCHANGE = init_field<uint32_t>("NUMBER_OF_FRAMES_BETWEEN_DATA_EXCHANGE", 60);
    MEASURE_TIMINGS = init_field<bool>("MEASURE_TIMINGS", false);

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

    /// Octree properties
    MAX_POINTS_PER_LEAF = init_field<uint32_t>("MAX_POINTS_PER_LEAF", 50'000);
    LRU_UPDATES_CACHE_SIZE = init_field<uint32_t>("LRU_UPDATES_CACHE_SIZE", 512);
    LRU_VISIBILITY_CACHE_SIZE = init_field<uint32_t>("LRU_VISIBILITY_CACHE_SIZE", 512);

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
    DIFFERENTIATE_VISIBLE_NODES_AABB_AT_STARTUP = init_field<bool>("DIFFERENTIATE_VISIBLE_NODES_AABB_AT_STARTUP", false);
    CuRastSettings::showVisibleNodes = DIFFERENTIATE_VISIBLE_NODES_AABB_AT_STARTUP;
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
    println("    - PER_NODE_KERNEL_BLOCK_SIZE: {}", PER_NODE_KERNEL_BLOCK_SIZE);

    println("");
    println("Miscellaneous:");
    println("    - IS_RUNNING_IN_PARALLEL: {}", IS_RUNNING_IN_PARALLEL);
    println("    - NUMBER_OF_FRAMES_BETWEEN_DATA_EXCHANGE: {}", NUMBER_OF_FRAMES_BETWEEN_DATA_EXCHANGE);
    println("    - MEASURE_TIMINGS: {}", MEASURE_TIMINGS);

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

    println("");
    println("UI initial settings:");
    println("    - SHOW_BOUNDING_BOXES_AT_STARTUP: {}", SHOW_BOUNDING_BOXES_AT_STARTUP);
    println("    - BRUTE_FORCE_RENDERING_AT_STARTUP: {}", BRUTE_FORCE_RENDERING_AT_STARTUP);
    println("    - DEBUG_LOD_TO_RENDER_AT_STARTUP: {}", DEBUG_LOD_TO_RENDER_AT_STARTUP);
    println("    - VOXELS_POINTS_PER_AXIS_AT_STARTUP: {}", VOXELS_POINTS_PER_AXIS_AT_STARTUP);
    println("    - MIN_PIXEL_SPAN_AT_STARTUP: {}", MIN_PIXEL_SPAN_AT_STARTUP);
    println("    - USE_VOXELS_DEBUG_COLOR_AT_STARTUP: {}", USE_VOXELS_DEBUG_COLOR_AT_STARTUP);
    println("    - USE_AUTO_FREE_OLD_OCTREE_ON_GPU_AT_STARTUP: {}", USE_AUTO_FREE_OLD_OCTREE_ON_GPU_AT_STARTUP);
    println("    - DIFFERENTIATE_VISIBLE_NODES_AABB_AT_STARTUP: {}", DIFFERENTIATE_VISIBLE_NODES_AABB_AT_STARTUP);

	println("\n//////////////////////////////////////////////////////////");
	println("//////////////////////////////////////////////////////////");
	println("//////////////////////////////////////////////////////////\n");
}
