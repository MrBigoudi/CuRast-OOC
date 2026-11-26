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
    /// The number of voxels in a chunk: a voxel (`CVoxel', 8 bytes) is half a point (`CPoint', 16
    /// bytes), so a voxel chunk holds twice as many elements in the same memory. Both kinds of chunks
    /// come from the same pool (`CChunk'), so the memory is shared between points and voxels as needed
    static constexpr uint32_t NB_VOXELS_PER_CHUNK = 2 * NB_POINTS_PER_CHUNK;
    /// The occupancy grid size per dimension
    /// This setting is set as a constant to be used in static arrays
    /// At most 256: a voxel stores its cell coordinates on 8 bits per axis (`CVoxel')
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
    /// The maximum available vram
    static inline double DEVICE_TOTAL_MEMORY;
    static inline double DEVICE_AVAILABLE_MEMORY;

    /// The host physical memory, and the part of it available at startup (Linux: `MemAvailable')
    static inline double HOST_TOTAL_MEMORY;
    static inline double HOST_AVAILABLE_MEMORY;



    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////// MISCELLANEOUS /////////////////////////////
    /////////////////////////////////////////////////////////////////////////

    /// Measure the timings
    static inline bool MEASURE_TIMINGS;
    /// Tells if need to find automatic properties
    static inline bool DERIVE_AUTOMATIC_PROPERTIES;
    /// Tells if the point cloud filling network can be used
    static inline bool IS_USING_NN;



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
    /// The maximum number of threads decoding point cloud files at the same time
    /// 0 = automatic (a quarter of the CPU cores)
    static inline uint32_t MAX_LOADING_THREADS;

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
    /// The maximum number of voxels chunk that can be exchanged at once (NB_VOXELS_PER_CHUNK voxels each)
    static inline uint32_t MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE;
    /// The points rendering budget for the visibility cache: points of the cut nodes
    static inline uint32_t MAX_NB_RENDERED_POINTS;
    /// The voxels rendering budget for the visibility cache: voxels of the cut nodes
    static inline uint32_t MAX_NB_RENDERED_VOXELS;
    /// The points rendering budget for the visibility cache: points of the large nodes (above the cut)
    static inline uint32_t MAX_NB_RENDERED_LARGE_POINTS;
    /// The voxels rendering budget for the visibility cache: "side" voxels of the out-of-view fallback
    /// (voxels of the large nodes in the octants of their children outside the frustum)
    static inline uint32_t MAX_NB_RENDERED_SIDE_VOXELS;
    /// The maximum number of Count/Split operations per update
    static inline uint32_t MAX_NB_COUNT_SPLIT_ITERATION;



    /////////////////////////////////////////////////////////////////////////
    //////////////////////////// AUTOMATIC SIZING ///////////////////////////
    /////////////////////////////////////////////////////////////////////////
    // Inputs of `init_default' (only used when `DERIVE_AUTOMATIC_PROPERTIES' is set); see the
    // comments of `init_default' for the meaning and the justification of each of them

    /// Device memory left to everything `init_default' can't account for: allocations made after it
    /// (kernels compiled later, local memory of the kernels, CUB temporary storage, framebuffers
    /// resized with the window, ...): RESERVE_MB + RESERVE_FRACTION x total device memory
    static inline double DEVICE_MEMORY_RESERVE_MB;
    static inline double DEVICE_MEMORY_RESERVE_FRACTION;
    /// Extra reserve when the point cloud filling network is used (`IS_USING_NN')
    static inline double NN_MEMORY_RESERVE_MB;
    /// Largest share of the usable device memory given to each family of buffers
    static inline double RENDERING_MEMORY_FRACTION;
    static inline double NODE_IDS_MEMORY_FRACTION;
    static inline double UPDATE_MEMORY_FRACTION;
    static inline double EXCHANGE_MEMORY_FRACTION;
    /// Size of the spilling / backlog buffers, per point of an octree update
    static inline double SPILLING_POINTS_PER_UPDATE_POINT;
    static inline double BACKLOG_VOXELS_PER_UPDATE_POINT;
    /// Composition of the pools, per resident node: average fill of a node's chunk list (relative to
    /// MAX_POINTS_PER_LEAF), occupancy grids per resident node, node slots per resident node
    static inline double CHUNKS_FILL_RATIO;
    static inline double GRIDS_PER_RESIDENT_NODE;
    /// Share of the resident nodes counted as inner nodes (holding voxels, half the chunks of a leaf)
    static inline double INNER_NODES_FRACTION;
    static inline double NODES_PER_RESIDENT_NODE;
    /// Nodes an update brings in (loaded or created), per MAX_POINTS_PER_LEAF points of the update
    static inline double TRANSIENT_NODES_PER_UPDATE_LEAF;
    /// Smallest updates cache accepted before the update size is reduced
    static inline uint32_t MIN_LRU_UPDATES_CACHE_SIZE;
    /// Share of the host physical memory given to the pinned host cache (`LRU_CPU_CACHE_SIZE')
    static inline double HOST_CACHE_MEMORY_FRACTION;
    /// Testing: > 0 sizes the buffers as on a card with this much memory (GB); 0 = the real card
    static inline double SIMULATED_DEVICE_MEMORY_GB;



    /////////////////////////////////////////////////////////////////////////
    //////////////////////////////// RENDERING //////////////////////////////
    /////////////////////////////////////////////////////////////////////////

    /// Voxel splats: every voxel is drawn as a camera-facing square of side VOXEL_SPLAT_SCALE voxel
    /// cells, at most VOXEL_SPLAT_MAX_PIXELS pixels wide. A scale <= 0 draws one point per voxel
    static inline float VOXEL_SPLAT_SCALE;
    static inline uint32_t VOXEL_SPLAT_MAX_PIXELS;
    /// Draw the batches already on the device but not inserted in the octree yet
    static inline bool DRAW_PENDING_BATCHES;
    /// At most about this many pending points drawn per frame (strided); 0 = all of them
    static inline uint32_t PENDING_BATCHES_MAX_POINTS;
    /// Capacity of the device copies of the evicted nodes kept drawable until the host rendering
    /// buffers show them: a fraction of the rendering budgets, and this many eviction passes
    static inline double EVICTION_RETAIN_BUDGET_FRACTION;
    static inline uint32_t EVICTION_RETAIN_NB_PASSES;
    /// Longest wait for the host rendering buffers to cover an eviction before evicting anyway
    static inline double MAX_EVICTION_COVERAGE_WAIT_SECONDS;



    /////////////////////////////////////////////////////////////////////////
    /////////////////////////////// VISIBILITY //////////////////////////////
    /////////////////////////////////////////////////////////////////////////

    /// Fraction of the rendering budget shared equally between every visible node
    static inline double VIS_MIN_SHARE_FRACTION;
    /// Distance weighting of the budget: weight = (d_closest / d_node)^VIS_DISTANCE_EXPONENT
    static inline double VIS_DISTANCE_EXPONENT;
    /// The node distance is floored to this fraction of the node diagonal
    static inline double VIS_DISTANCE_FLOOR_FRACTION;
    /// Host RAM cache of the node files read by the visibility updates (bytes; MB in the toml file)
    static inline uint64_t VIS_RAM_CACHE_MAX_BYTES;
    /// Smallest prefix of a node file kept in the RAM cache
    static inline uint32_t VIS_RAM_CACHE_MIN_KEEP;
    /// Longest pause of the visibility updates while the octree update does disk work
    static inline double VIS_MAX_PAUSE_SECONDS;
    /// Threads reading / writing the node files of the octree update; 0 = automatic
    static inline uint32_t DISK_IO_THREADS;



    /////////////////////////////////////////////////////////////////////////
    ///////////////////////////////// LOADER ////////////////////////////////
    /////////////////////////////////////////////////////////////////////////

    /// Fallback period of the device slots status readback
    static inline double STATUS_READBACK_FALLBACK_SECONDS;
    /// At most this many device slots hold out-of-view batches; 0 = no limit
    static inline uint32_t MAX_OUT_OF_VIEW_DEVICE_SLOTS;
    /// The loader stays at most this many device updates ahead
    static inline uint32_t LOAD_AHEAD_FACTOR;
    /// A pending batch replaces a loaded (not yet sent) one if it is this many times better
    static inline float PREEMPTION_FACTOR;
    /// Centre bias of the loading priority (0 = none)
    static inline float LOAD_CENTER_WEIGHT;
    /// At most this many workers decode the same file at once; 0 = no limit
    static inline uint32_t MAX_READERS_PER_FILE;
    /// A load in progress is re-ranked (and possibly cancelled) every this many points
    static inline uint32_t CANCEL_CHECK_INTERVAL;
    /// Age after which the summary of the pending batches is computed again
    static inline double PENDING_SUMMARY_REFRESH_SECONDS;
    /// A batch whose loading fails this many times is skipped
    static inline uint32_t MAX_LOAD_ATTEMPTS;
    /// Stride of the overview sample when the LAZ chunk points can't be used
    static inline uint32_t OVERVIEW_FALLBACK_STRIDE;
    /// Preview of the loaded batches not sent yet: regions (at most 32) of points each
    static inline bool PREVIEW_ENABLED;
    static inline uint32_t PREVIEW_NB_REGIONS;
    static inline uint32_t PREVIEW_POINTS_PER_REGION;
    /// Progressive preview of the batches being decoded: first refresh after this many points, then
    /// every PREVIEW_PROGRESSIVE_INTERVAL points
    static inline bool PREVIEW_PROGRESSIVE;
    static inline uint32_t PREVIEW_PROGRESSIVE_FIRST_POINTS;
    static inline uint32_t PREVIEW_PROGRESSIVE_INTERVAL;
    /// Ask the OS to read a batch's part of its file ahead before decoding it
    static inline bool READ_AHEAD;




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
    static void init_device_properties();
    /// Measure the device / host memory (also called again once the kernels are loaded)
    static void init_memory_properties();
    static void init_miscellaneous();
    static void init_ui_params();
    static void init_batch_sizes();
    static void init_octree_properties();
    static void init_gpu_version_buffers();
    static void init_automatic_sizing();
    static void init_rendering();
    static void init_visibility();
    static void init_loader();
    /// Derive the buffer sizes from the available memory (see the comments in settings.cpp)
    /// Can be called several times: it always starts from the values of the toml file
    static void init_default();

    static void init();
};