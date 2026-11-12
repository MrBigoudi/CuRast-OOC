#pragma once

#include "CuRast.h"
#include "globals.h"

/// Initialized empty batches of points from las/laz files
/// Updates the global `batchesToLoad` variable
void initLoadPointBatches(string file,
    std::deque<std::shared_ptr<PointBatch>>& batches_queue = GlobalVariables::batchesQueue,
    std::deque<std::mutex>& batches_queue_mutexes = GlobalVariables::batchesQueueMutexes
);

/// Load points from the global `batchesToLoad` variable
void loadPointsInBatches(
    std::deque<std::shared_ptr<PointBatch>>& batches_queue = GlobalVariables::batchesQueue,
    std::deque<std::mutex>& batches_queue_mutexes = GlobalVariables::batchesQueueMutexes
);

/// Asynchronously load the point clouds into queues of batches
void loadPointcloudRoutine(
    std::deque<std::shared_ptr<PointBatch>>& batches_queue = GlobalVariables::batchesQueue,
    std::deque<std::mutex>& batches_queue_mutexes = GlobalVariables::batchesQueueMutexes
);

/// TODO: temporary function
/// Send points to CUDA memory
/// Used to get baseline timings but consumes a lot of memory
void loadBatchesOnGPU(CuRast* editor, CUcontext* ctx = nullptr,
    std::deque<std::shared_ptr<PointBatch>>& batches_queue = GlobalVariables::batchesQueue,
    std::deque<std::mutex>& batches_queue_mutexes = GlobalVariables::batchesQueueMutexes
);

/// Clear the unused batches
void clearUnusedBatches(
    std::deque<std::shared_ptr<PointBatch>>& batches_queue = GlobalVariables::batchesQueue,
    std::deque<std::mutex>& batches_queue_mutexes = GlobalVariables::batchesQueueMutexes
);
/// Asynchronously clear the unused batches
void clearUnusedBatchesRoutine(
    std::deque<std::shared_ptr<PointBatch>>& batches_queue = GlobalVariables::batchesQueue,
    std::deque<std::mutex>& batches_queue_mutexes = GlobalVariables::batchesQueueMutexes
);



// Helper types of `LoaderGpuVersion', at namespace scope: a static member of a nested type can't use
// its default member initializers inside the enclosing class

/// Approximate spatial extent of a batch: estimated from the chunk points it contains (overview
/// sample) before it is loaded, exact once loaded
struct LoaderBatchBounds {
    glm::vec3 mins = glm::vec3( INFINITY);
    glm::vec3 maxs = glm::vec3(-INFINITY);
    bool isValid() const { return mins.x <= maxs.x && mins.y <= maxs.y && mins.z <= maxs.z; }
    void extend(const glm::vec3& p){ mins = glm::min(mins, p); maxs = glm::max(maxs, p); }
};

/// A batch not loaded yet, with its estimated bounds
struct LoaderPendingBatch {
    std::shared_ptr<PointBatch> batch = nullptr;
    LoaderBatchBounds bounds = {};
    uint32_t nb_failures = 0;   // failed load attempts (open / seek / read errors)
};

/// Camera used to rank the batches (snapshot taken on the main thread every frame)
struct LoaderCameraState {
    glm::mat4 view_proj = glm::mat4(1.0f);
    glm::vec3 position = glm::vec3(0.0f);
    float width = 1.0f;
    float height = 1.0f;
};

/// Loading priority (smaller = loaded first): batches in view first, by decreasing screen-space size
/// at their nearest point (1 / distance^2, centre-weighted like the visibility update), then the
/// others by increasing distance to the camera, then the batches without bounds
struct LoaderLoadPriority {
    uint32_t tier = 2;      // 0: in view, 1: out of view, 2: unknown bounds
    float key = 0.0f;       // tier 0: -centre-weighted 1/distance^2 (-inf: camera inside), tier 1: distance
    bool operator<(const LoaderLoadPriority& rhs) const {
        return tier != rhs.tier ? tier < rhs.tier : key < rhs.key;
    }
};


struct LoaderGpuVersion {
    using BatchBounds = LoaderBatchBounds;
    using PendingBatch = LoaderPendingBatch;
    using CameraState = LoaderCameraState;
    using LoadPriority = LoaderLoadPriority;

    static inline std::deque<std::shared_ptr<PointBatch>> batchesQueue = {};
	static inline std::deque<std::mutex> batchesQueueMutexes = {};

    static inline std::mutex laszipReaderMtx;
    static inline std::unordered_map<std::string, std::mutex> perFileMutexes;
    static inline std::mutex perFileMutexesMtx;

    static inline std::vector<uint32_t> batchesOnGpu = {};
    static inline void* batchesOnGpuStatus = nullptr;

    static inline CUstream stream;
    static inline CUevent eventLoadComplete;

    static inline std::vector<CUmemcpyAttributes> loadingAttributes = {
        CUmemcpyAttributes {
            .srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
            .srcLocHint = {CU_MEM_LOCATION_TYPE_HOST},
            .dstLocHint = {CU_MEM_LOCATION_TYPE_DEVICE},
            .flags = CU_MEMCPY_FLAG_DEFAULT
        }
    };
    static inline std::vector<uint64_t> loadingAttributesIndices = {0};

    /// Every batch not loaded yet: ranked again with the current camera at each loading round
    /// (`loadPendingBatches'); a queue slot is only taken when a batch is picked for loading
    static inline std::vector<PendingBatch> pendingBatches = {};
    static inline std::mutex pendingBatchesMtx;
    /// Bounds of the batch in each slot of `batchesQueue' (guarded like the slot: written when the
    /// slot is taken, then when the batch is loaded, before it is published in `loadedBatches')
    static inline std::vector<BatchBounds> slotsBounds = {};
    /// The loader stays at most this many device updates ahead (loaded but not yet sent batches, plus
    /// one being loaded per worker): loading decisions are taken late, with the current camera
    static inline uint32_t LOAD_AHEAD_FACTOR = 2;
    /// When the loader is that far ahead, a pending batch replaces the worst loaded (not yet sent) one
    /// if it is this many times better (screen-space size, or distance): the loaded batches of a
    /// region the camera left don't hold back the region it flies to
    static inline float PREEMPTION_FACTOR = 2.0f;
    /// Centre bias of the loading priority, same formula as `VISIBILITY_CENTER_WEIGHT' in render.cu:
    /// score *= 1 - weight * (distance of the box's on-screen centre to the screen centre / half
    /// diagonal). 0 = no bias: boxes at the same distance score the same wherever they are on screen
    static inline float LOAD_CENTER_WEIGHT = 0.5f;

    static inline std::mutex cameraStateMtx;
    static inline CameraState cameraState = {};
    static CameraState getCameraState();

    static LoadPriority loadPriority(const BatchBounds& bounds, const CameraState& camera);
    static bool isClearlyBetter(const LoadPriority& a, const LoadPriority& b);
    /// Load the highest priority pending batches (replaces `enqueueBatches' + `loadPointsInBatches'):
    /// in parallel, starts the persistent loading workers on the first call
    static void loadPendingBatches();
    /// Load the current best pending batch (one worker step); false if nothing was loaded
    static bool loadOneBatch();
    /// Stop and join the loading workers (also registered with `std::atexit' when they start, so
    /// that they never outlive the static data they use, whatever the way the program exits)
    static void stopLoading();
    /// A batch whose loading fails this many times is skipped (its points are never sent: after a
    /// LAZ decoding error, the points decoded just before it can't be trusted)
    static inline uint32_t MAX_LOAD_ATTEMPTS = 3;

    /// Persistent loading workers (parallel mode)
    static inline std::mutex loadingWorkersMtx;
    static inline std::vector<std::thread> loadingWorkers = {};
    static inline std::atomic<bool> stopLoadingWorkers{false};
    /// Calls of `loadingRoutine' in progress (`stopLoading' waits for them: main.cpp's loading
    /// thread is not joined when the program exits on an error)
    static inline std::atomic<uint32_t> nbLoadingRoutinesRunning{0};
    static inline uint32_t nbLoadingWorkers = 1;
    /// Batches being loaded, and number of workers reading each file (guarded by `pendingBatchesMtx')
    static inline std::atomic<uint32_t> nbLoadsInFlight{0};
    static inline std::unordered_map<std::string, uint32_t> filesBeingRead = {};

    static inline std::mutex loadedBatchesMtx;
    static inline std::vector<std::pair<uint32_t, std::shared_ptr<PointBatch>>> loadedBatches;
    /// Overview batches (one point per LAZ chunk of every dropped file, see
    /// `createNewBatchesFromFiles'): already in memory, sent to the device before any other batch
    /// Guarded by `loadedBatchesMtx'
    static inline std::vector<std::pair<uint32_t, std::shared_ptr<PointBatch>>> loadedOverviewBatches;

    /// Stride of the overview sample when the chunk points of a file can't be used: uncompressed LAS
    /// files (any point can be read directly) or LAZ files with variable-size chunks
    /// (50000 is the LASzip default chunk size)
    static inline uint32_t OVERVIEW_FALLBACK_STRIDE = 50000;

    struct FileInfo {
        std::shared_ptr<laszip_header> header = nullptr;
        uint32_t nb_batches = 0;
        std::atomic<uint32_t> nb_batches_loaded = 0;
        std::atomic<uint32_t> nb_batches_on_device = 0;
        std::atomic<uint32_t> nb_batches_inserted = 0;
    };

    static inline std::unordered_map<std::string, FileInfo> filesRecord = {};
    static inline std::mutex filesRecordMtx;

    static void createNewBatches(string file);
    /// Create the batches of several files dropped together (LidarScout-like first stage):
    ///  1. the first point of every LAZ chunk (stored uncompressed: one seek and one point read
    ///     each) of every file is read, and these points are mixed into "overview" batches sent
    ///     before anything else: the first octree update sees the whole data set (overview of the
    ///     entire cloud, tight octree root)
    ///  2. then the regular batches of every file are created as before (the overview points are
    ///     inserted a second time with them)
    static void createNewBatchesFromFiles(std::vector<std::string> files);
    static bool run(CuRast* editor, CUcontext* context);

    static void init();
    static void destroy();
    static void fetchFromDevice();
    static bool sendToDevice();

    static void loadingRoutine();


    static void filesRecordUi();
    static glm::vec3 batchHeaderCenter(const laszip_header& h);
    static glm::vec3 batchPointsCentroid(const std::shared_ptr<PointBatch>& batch);

    static inline std::atomic<float> cameraPosX{0.0f};
    static inline std::atomic<float> cameraPosY{0.0f};
    static inline std::atomic<float> cameraPosZ{0.0f};
    static void updateCameraSnapshot();
    static vec3 getCameraSnapshot();
};