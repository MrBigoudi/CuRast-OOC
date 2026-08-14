#pragma once

#include <cstdint>

/// A structure containing all the settings for the OOC-SimLod
/// To avoid recompile time, change the values in the `SETTINGS_TOML_FILE' rather than here
struct OocSimLodSettings {
    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////////// FIXED /////////////////////////////////
    /////////////////////////////////////////////////////////////////////////

    /// The settings toml file from which to read the configuration
    /// Default to `settings.toml'
    static const char* SETTINGS_TOML_FILE;
    /// The temporary files directory to store nodes in disk
    /// Default to `build/tmp'
    static const char* TEMPORARY_NODE_STORAGE_DIRECTORY;

    /// The number of points in a chunk
    /// This setting is set as a constant to be used in static arrays
    static constexpr uint32_t NB_POINTS_PER_CHUNK = 1024;
    /// The occupancy grid size per dimension
    /// This setting is set as a constant to be used in static arrays
    static constexpr uint32_t GRID_SIZE_PER_DIMENSION = 128;
    /// The occupancy grid complete size
    static constexpr uint32_t GRID_SIZE = GRID_SIZE_PER_DIMENSION 
        * GRID_SIZE_PER_DIMENSION 
        * GRID_SIZE_PER_DIMENSION
    ;



    /////////////////////////////////////////////////////////////////////////
    /////////////////////////// DEVICE PROPERTIES ///////////////////////////
    /////////////////////////////////////////////////////////////////////////
    
    /// The number of multiprocessors
    static inline uint32_t DEVICE_ATTRIBUTE_NB_SM;
    /// The maximum number of threads per sm
    static inline uint32_t DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM;
    /// The maximum number of threads per block
    static inline uint32_t DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
    /// The maximum number of blocks per sm
    static inline uint32_t DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_SM;
    /// The maximum block size
    static inline uint32_t DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X;
    static inline uint32_t DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y;
    static inline uint32_t DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z;
    /// The maximum grid size
    static inline uint32_t DEVICE_ATTRIBUTE_MAX_GRID_DIM_X;
    static inline uint32_t DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y;
    static inline uint32_t DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z;
    static inline uint32_t DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE;



    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// MISCELLANEOUS /////////////////////////////
    /////////////////////////////////////////////////////////////////////////

    /// Tells if the program should be entirely sequential or as parallelised as possible
    static inline bool IS_RUNNING_IN_PARALLEL;
    /// The number of frames between each GPU data transfer and visibility update
    static inline uint32_t NUMBER_OF_FRAMES_BETWEEN_DATA_EXCHANGE;
    /// Measure the timings
    static inline bool MEASURE_TIMINGS;
    /// Run the GPU version
    static inline bool IS_USING_GPU_VERSION;



    /////////////////////////////////////////////////////////////////////////
    ////////////////////////////// BATCH SIZES //////////////////////////////
    /////////////////////////////////////////////////////////////////////////
    
    /// The size of the batches list
    /// The size is not constant to handle big sequential versions
    static inline uint32_t BATCHES_LIST_SIZE;
    /// The maximum number of points per batches
    static inline uint32_t MAX_POINTS_PER_BATCHES;

    /// The minimum number of batches that can be loaded from disk at once
    static inline uint32_t MIN_BATCHES_PER_LOAD;
    /// The maximum number of batches that can be loaded from disk at once
    static inline uint32_t MAX_BATCHES_PER_LOAD;

    /// The minimum number of batches that can be loaded to the GPU at once
    /// A batch is loaded as a collection of points on the GPU only for the brute-force comparison
    static inline uint32_t MIN_BATCHES_PER_GPU_LOAD;
    /// The maximum number of batches that can be loaded to the GPU at once
    static inline uint32_t MAX_BATCHES_PER_GPU_LOAD;

    /// The minimum number of batches that should be added at the same time in the octree
	static inline uint32_t MIN_BATCHES_PER_OCTREE_UPDATE;
    /// The maximum number of batches that should be added at the same time in the octree
    static inline uint32_t MAX_BATCHES_PER_OCTREE_UPDATE;

    /// The number of attempts after which the minimum variables are ignored
    static inline uint32_t MAX_ATTEMPTS_BEFORE_IGNORING_MIN_VARIABLES;



    /////////////////////////////////////////////////////////////////////////
    /////////////////////////// OCTREE PROPERTIES ///////////////////////////
    /////////////////////////////////////////////////////////////////////////

    /// The maximum number of points in a leaf node
    static inline uint32_t MAX_POINTS_PER_LEAF;

    /// The size of the LRU updates cache
    static inline uint32_t LRU_UPDATES_CACHE_SIZE;
    /// The size of the LRU visibility cache
    static inline uint32_t LRU_VISIBILITY_CACHE_SIZE;
    /// The size of the LRU cpu cache
    static inline uint32_t LRU_CPU_CACHE_SIZE;

    /// The maximum number of chunks allowed in memory
    static inline uint32_t NB_ALLOCABLE_CHUNKS;
    /// The maximum number of occupancy grids allowed in memory
    static inline uint32_t NB_ALLOCABLE_GRIDS;
    /// The maximum number of nodes allowed in memory
    static inline uint32_t NB_ALLOCABLE_NODES;



    /////////////////////////////////////////////////////////////////////////
    ////////////////////////////// GPU VERSION //////////////////////////////
    /////////////////////////////////////////////////////////////////////////

    /// The initial maximum number of nodes allowed to be created during runtime
    static inline uint32_t MAX_NB_NODES; 
    /// The maximum number of nodes that can be exchanged at once between the host and the device
    static inline uint32_t MAX_NB_NODES_TO_EXCHANGE;
    /// The maximum number of spilling points
    static inline uint32_t MAX_NB_SPILLING_POINTS;
    /// The maximum number of backlog voxels
    static inline uint32_t MAX_NB_BACKLOG_VOXELS;
    /// The maximum number of voxels chunk that can be exchanged at once
    static inline uint32_t MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE;
    /// The points rendering budget for the visibility cache
    static inline uint32_t MAX_NB_RENDERED_POINTS;
    /// The voxels rendering budget for the visibility cache
    static inline uint32_t MAX_NB_RENDERED_VOXELS;




    /////////////////////////////////////////////////////////////////////////
    ///////////////////////// UI INITIAL PARAMETERS /////////////////////////
    /////////////////////////////////////////////////////////////////////////

    /// Display the bounding boxes 
    static inline bool SHOW_BOUNDING_BOXES_AT_STARTUP;
    /// Send points from batches to the GPU as well and display the complete point cloud
	static inline bool BRUTE_FORCE_RENDERING_AT_STARTUP;
    /// The lod to render, -1 for the dynamic selection
	static inline int32_t DEBUG_LOD_TO_RENDER_AT_STARTUP;
    /// The number of points per axis to draw voxels as a cube of points
	static inline int32_t VOXELS_POINTS_PER_AXIS_AT_STARTUP;
    /// The minimum number of pixels to condider a node large
	static inline float MIN_PIXEL_SPAN_AT_STARTUP;
    /// Use a debug color for the voxels; the color will correspond to the level of the node
	static inline bool USE_VOXELS_DEBUG_COLOR_AT_STARTUP;
    /// Activate the automatic free of unused GPU memory
    static inline bool USE_AUTO_FREE_OLD_OCTREE_ON_GPU_AT_STARTUP;
    /// Use a white color for the visible nodes' bounding boxes
	static inline bool DISPLAY_VISIBLE_NODES_AABB_AT_STARTUP;
    


    /////////////////////////////////////////////////////////////////////////
    /////////////////////////////// FUNCTIONS ///////////////////////////////
    /////////////////////////////////////////////////////////////////////////

    /// Display the settings
    static void display();
    
    /// Constructor from the toml `SETTINGS_TOML_FILE' file
    static void init();
};