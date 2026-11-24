#pragma once

#include "gpuVersion.h"
#include "laszip/laszip_api.h"

/// The state of a batch
enum BatchState {
	Empty,
	ToLoad,
	Loading,
	Loaded,
	Sent,
	Inserted,
	ToRemove
};

/// A set of points read from a las / laz file
struct PointBatch {
	uint64_t first = 0;
	uint64_t count = 0;
	shared_ptr<string> file = nullptr;
	shared_ptr<vector<CPoint>> points = nullptr;
	shared_ptr<laszip_header> header = nullptr;
	BatchState state = BatchState::Empty;
	/// Drawn by the preview of the loaded batches (`LoaderGpuVersion::updatePreview'): never demoted,
	/// never held back, until it is sent. Written by the main thread under `loadedBatchesMtx'
	std::atomic<bool> is_previewed{false};
};

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

/// Best batch a worker could pick right now (see `LoaderGpuVersion::getPendingSummary'): used to
/// cancel the loads in progress that became clearly worse, and to hold back the out-of-view batches
/// at send time while batches in view still wait
struct LoaderPendingSummary {
    bool has_any = false;                   // some pending batch can be picked
    LoaderLoadPriority best = {};           // its priority (camera of the scan)
};

/// Preview of the loaded batches not sent to the device yet (see `LoaderGpuVersion::updatePreview'):
/// one region of the device preview buffer per previewed batch
struct LoaderPreviewRegion {
    std::shared_ptr<PointBatch> batch = nullptr;   // previewed batch (nullptr: free region)
    uint32_t nb_points = 0;                        // points of the batch in the region (subsample)
    bool is_uploading = false;                     // copy to the device in flight (`upload_event')
    bool is_visible = false;                       // drawn (its copy is complete, it is still previewed)
    CUevent upload_event = nullptr;
};

/// Number of points to draw in each preview region (kernel parameter: same layout as
/// `CPreviewCounts' in render.cu)
constexpr uint32_t LOADER_PREVIEW_MAX_REGIONS = 32;
struct LoaderPreviewCounts {
    uint32_t counts[LOADER_PREVIEW_MAX_REGIONS] = {};
};

/// One `sendToDevice' upload in flight on the loading stream (see `LoaderGpuVersion::fetchFromDevice')
struct LoaderSendRecord {
    CUevent event = nullptr;                                  // recorded after its copies
    std::vector<uint32_t> device_slots = {};                  // device batch slots it fills
    std::vector<std::shared_ptr<PointBatch>> batches = {};    // kept alive while they are copied
};


struct LoaderGpuVersion {
    using BatchBounds = LoaderBatchBounds;
    using PendingBatch = LoaderPendingBatch;
    using CameraState = LoaderCameraState;
    using LoadPriority = LoaderLoadPriority;
    using PendingSummary = LoaderPendingSummary;

    static inline std::deque<std::shared_ptr<PointBatch>> batchesQueue = {};
	static inline std::deque<std::mutex> batchesQueueMutexes = {};

    static inline std::mutex laszipReaderMtx;
    static inline std::unordered_map<std::string, std::mutex> perFileMutexes;
    static inline std::mutex perFileMutexesMtx;

    static inline std::vector<uint32_t> batchesOnGpu = {};
    /// Pinned copy of the device `batchesAddedMask', read back asynchronously (`fetchFromDevice')
    static inline void* batchesOnGpuStatus = nullptr;

    static inline CUstream stream;
    static inline CUevent eventLoadComplete;

    /// The device batch slots are handed back without blocking the main thread (`fetchFromDevice'):
    ///  - each `sendToDevice' upload records an event on the loading stream; once it is complete, the
    ///    device slots it filled are "confirmed" (points, count and mask `BatchToHandle' written)
    ///  - the mask is read back with an asynchronous copy on the default stream (after the octree
    ///    update kernels queued so far); only the slots confirmed when that copy was queued are
    ///    interpreted (a slot sent after it may still show the `BatchHandled' of its previous batch)
    /// Main thread only
    static inline std::deque<LoaderSendRecord> sendsInFlight = {};
    static inline std::vector<CUevent> freeSendEvents = {};
    static inline std::vector<uint8_t> isDeviceSlotConfirmed = {};
    static inline CUevent eventStatusReadback = nullptr;
    static inline bool isStatusReadbackInFlight = false;
    static inline std::vector<uint8_t> statusReadbackEligible = {};
    /// True if `batch' is part of an upload not complete yet (its device slot is not drawn yet)
    static bool isUploadInFlight(const PointBatch* batch);

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
    /// (1: fewer stale batches to work through after a camera move)
    static inline uint32_t LOAD_AHEAD_FACTOR = 1;
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

    /// At most this many workers decode batches of the same file at once (each one has its own
    /// LASzip reader); 0 = no limit: the region in front of the camera, often one or two tiles, is
    /// loaded by every worker instead of one
    static inline uint32_t MAX_READERS_PER_FILE = 0;
    /// A load in progress re-ranks its batch with the current camera every this many points, and is
    /// cancelled when a pending batch is clearly better (`isClearlyBetter'); it goes
    /// back to the pending batches, nothing of it was published. Not after half of the batch
    static inline uint32_t CANCEL_CHECK_INTERVAL = 65536;
    /// Age after which `getPendingSummary' scans the pending batches again
    static inline double PENDING_SUMMARY_REFRESH_SECONDS = 0.05;

    /// Index of the best pending batch a worker may pick (its file has fewer than
    /// `MAX_READERS_PER_FILE' readers), -1 if none. `pendingBatchesMtx' must be held
    static int64_t selectBestPending(const CameraState& camera, LoadPriority& best_priority);
    /// Best pending batch with the current camera, scanned at most every
    /// `PENDING_SUMMARY_REFRESH_SECONDS', and again after every pick (a pick invalidates it).
    /// `pendingBatchesMtx' must NOT be held
    static PendingSummary getPendingSummary();
    /// True if some batch in view (tier 0 with the current camera) is pending or being loaded: the
    /// out-of-view loaded batches are then not sent (see `sendToDevice')
    static bool hasInViewWork(const CameraState& camera);
    static inline std::mutex pendingSummaryMtx;
    static inline PendingSummary pendingSummary = {};
    static inline std::chrono::steady_clock::time_point pendingSummaryTime = {};
    /// False when a batch was picked since the last scan (the summary may name it: scanned again)
    static inline bool pendingSummaryIsValid = false;
    /// Load the highest priority pending batches (replaces `enqueueBatches' + `loadPointsInBatches'):
    /// in parallel, starts the persistent loading workers on the first call
    static void loadPendingBatches();
    /// Load the current best pending batch (one worker step); false if nothing was loaded
    static bool loadOneBatch();
    /// Number of loading workers decoding point cloud files at the same time: `MAX_LOADING_THREADS',
    /// or a quarter of the CPU cores when it is 0, at most the number of cores and
    /// `MAX_BATCHES_PER_LOAD'. The overview of dropped files is read with every core instead
    /// (see `createNewBatchesFromFiles')
    static uint32_t nbLoadingThreads();
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
    /// Estimated bounds of the batches being loaded (guarded by `pendingBatchesMtx')
    static inline std::unordered_map<const PointBatch*, BatchBounds> loadsInFlightBounds = {};

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

    static inline std::atomic<float> cameraPosX{0.0f};
    static inline std::atomic<float> cameraPosY{0.0f};
    static inline std::atomic<float> cameraPosZ{0.0f};
    static void updateCameraSnapshot();
    static vec3 getCameraSnapshot();

    /// Clear the unused batches
    static void clearUnusedBatches();

    /// Preview of the loaded batches not sent yet ("stage 1"): the batches in view waiting on the host
    /// for a free device slot are drawn from a small device buffer until they are sent (from then on,
    /// `kernel_draw_pending_batches' draws them from their device slot until they are inserted)
    /// The buffer has `PREVIEW_NB_REGIONS' regions of `PREVIEW_POINTS_PER_REGION' points: the best
    /// batches in view get one each, with an even subsample of their points. A batch is only uploaded
    /// when it enters the preview (no copy when nothing changes)
    /// What is drawn once is never lost: a batch keeps its region until it is sent (then its device
    /// slot, then the octree draw it), whatever the camera does; a previewed batch is neither demoted
    /// nor held back, so its region is always freed
    static inline bool PREVIEW_ENABLED = true;
    static inline uint32_t PREVIEW_NB_REGIONS = 16;                // at most LOADER_PREVIEW_MAX_REGIONS
    static inline uint32_t PREVIEW_POINTS_PER_REGION = 262'144;
    static inline std::vector<LoaderPreviewRegion> previewRegions = {};
    static inline uint32_t previewPointsPerRegion = 0;            // as allocated
    static inline CPoint* previewDevicePoints = nullptr;           // nb regions x points per region
    static inline CPoint* previewPinnedPoints = nullptr;           // staging of the uploads (same layout)
    static inline CUstream previewStream;
    /// Recorded after the preview draw of every frame: a region is only overwritten once the draws
    /// launched before have read it
    static inline CUevent previewDrawnEvent;
    static void initPreview();
    static void destroyPreview();
    /// Main thread, every frame before `sendToDevice' (a sent batch stays previewed until
    /// `fetchFromDevice' sees its upload complete: its device slot is drawn from then on)
    static void updatePreview();
    /// Points to draw per region; returns false if nothing is visible (main thread)
    static bool previewDrawCounts(LoaderPreviewCounts& counts);
    /// After the preview draw (main thread)
    static void previewRecordDrawn();
};