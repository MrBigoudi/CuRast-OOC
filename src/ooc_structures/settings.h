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
    ///////////////////////////// MISCELLANEOUS /////////////////////////////
    /////////////////////////////////////////////////////////////////////////

    /// Tells if the program should be entirely sequential or as parallelised as possible
    static bool IS_RUNNING_IN_PARALLEL;
    /// The number of frames between each GPU data transfer and visibility update
    static uint32_t NUMBER_OF_FRAMES_BETWEEN_DATA_EXCHANGE;



    /////////////////////////////////////////////////////////////////////////
    ////////////////////////////// BATCH SIZES //////////////////////////////
    /////////////////////////////////////////////////////////////////////////
    
    /// The size of the batches list
    /// The size is not constant to handle big sequential versions
    static uint32_t BATCHES_LIST_SIZE;
    /// The maximum number of points per batches
    static uint32_t MAX_POINTS_PER_BATCHES;

    /// The minimum number of batches that can be loaded from disk at once
    static uint32_t MIN_BATCHES_PER_LOAD;
    /// The maximum number of batches that can be loaded from disk at once
    static uint32_t MAX_BATCHES_PER_LOAD;

    /// The minimum number of batches that can be loaded to the GPU at once
    /// A batch is loaded as a collection of points on the GPU only for the brute-force comparison
    static uint32_t MIN_BATCHES_PER_GPU_LOAD;
    /// The maximum number of batches that can be loaded to the GPU at once
    static uint32_t MAX_BATCHES_PER_GPU_LOAD;

    /// The minimum number of batches that should be added at the same time in the octree
	static uint32_t MIN_BATCHES_PER_OCTREE_UPDATE;
    /// The maximum number of batches that should be added at the same time in the octree
    static uint32_t MAX_BATCHES_PER_OCTREE_UPDATE;

    /// The number of attempts after which the minimum variables are ignored
    static uint32_t MAX_ATTEMPTS_BEFORE_IGNORING_MIN_VARIABLES;



    /////////////////////////////////////////////////////////////////////////
    /////////////////////////// OCTREE PROPERTIES ///////////////////////////
    /////////////////////////////////////////////////////////////////////////

    /// The maximum number of points in a leaf node
    static uint32_t MAX_POINTS_PER_LEAF;

    /// The size of the LRU updates cache
    static uint32_t LRU_UPDATES_CACHE_SIZE;
    /// The size of the LRU visibility cache
    static uint32_t LRU_VISIBILITY_CACHE_SIZE;
    


    /////////////////////////////////////////////////////////////////////////
    /////////////////////////////// FUNCTIONS ///////////////////////////////
    /////////////////////////////////////////////////////////////////////////

    /// Display the settings
    static void display();
    
    /// Constructor from the toml `SETTINGS_TOML_FILE' file
    static void init();
};