#pragma once

#include "CuRast.h"
#include "../kernels/ooc/GpuVersionInterface.h"
#include "../kernels/ooc/GpuVersionAllocator.h"

#include <list>
#include <unordered_set>

struct LoaderGpuVersion;
struct HostStorageNode;

/// The LRU caches for the nodes
/// https://www.geeksforgeeks.org/dsa/lru-cache-implementation-using-double-linked-lists/ + ChatGPT
struct CLRUCache {
	const uint32_t CACHE_SIZE;
    std::list<CIdAABB> cache;
    std::unordered_map<CIdAABB, std::list<CIdAABB>::iterator> cache_map;

    CLRUCache(uint32_t cache_size) : CACHE_SIZE(cache_size) {
        cache_map.reserve(CACHE_SIZE);
    }

    CIdAABB add(const CIdAABB& aabb_index) {
        // Already present: make it most recently used
        auto map_it = cache_map.find(aabb_index);

        if(map_it != cache_map.end()){
            cache.splice(cache.begin(), cache, map_it->second);
            return CINVALID_ID;
        }

        CIdAABB removed = CINVALID_ID;
        // Evict least recently used
        if (cache_map.size() >= CACHE_SIZE) {
            auto lru_it = std::prev(cache.end());
            removed = *lru_it;
            cache_map.erase(removed);
            cache.erase(lru_it);
        }

        // Add as most recently used
        cache.push_front(aabb_index);
        cache_map[aabb_index] = cache.begin();
        return removed;
    }

    bool contains(const CIdAABB& aabb_index) {
        return cache_map.contains(aabb_index);
    }

    uint32_t getSize() const {
        return static_cast<uint32_t>(cache_map.size());
    }

    void remove(const CIdAABB& aabb_index){
        // Already present: remove it
        auto map_it = cache_map.find(aabb_index);
        if(map_it != cache_map.end()){
            cache.erase(map_it->second);
            cache_map.erase(aabb_index);
        }
    }
};


struct GpuVersionUI {
    static inline uint32_t lastNbTotalUpdates = 0;
    static inline uint32_t nbTotalUpdates = 0;

    // Current update values
    static inline uint64_t nbNewPointsThisUpdate = 0;
	static inline uint64_t nbNewVoxelsThisUpdate = 0;
	static inline uint32_t nbNewNodesThisUpdate = 0;
	static inline uint32_t nbLoadedNodesThisUpdate = 0;
	static inline uint32_t nbStoredNodesThisUpdate = 0;
	static inline uint32_t nbSplitNodesThisUpdate = 0;
	static inline uint32_t nbDeletedNodesThisUpdate = 0;
	static inline uint32_t nbDeletedChunksThisUpdate = 0;
	static inline uint32_t nbDeletedGridsThisUpdate = 0;
	static inline uint32_t nbNewChunksThisUpdate = 0;
	static inline uint32_t nbNewGridsThisUpdate = 0;

    // Values
    static inline uint32_t currentNbNodes = 0;
	static inline uint32_t currentNbChunks = 0;
	static inline uint32_t currentNbGrids = 0;
    static inline uint64_t currentNbPoints = 0;
    static inline uint64_t currentNbVoxels = 0;

    static inline uint64_t nbTotalPoints = 0;
    static inline uint64_t nbTotalVoxels = 0;
    static inline uint32_t nbTotalNewNodes = 0;
    static inline uint32_t nbTotalNewGrids = 0;
    static inline uint32_t nbTotalNewChunks = 0;
    static inline uint32_t nbTotalDeletedNodes = 0;
    static inline uint32_t nbTotalDeletedGrids = 0;
    static inline uint32_t nbTotalDeletedChunks = 0;
    static inline uint32_t nbTotalLoadedNodes = 0;
    static inline uint32_t nbTotalSplitNodes = 0;
    static inline uint32_t nbTotalStoredNodes = 0;

    // Flow rates
    static inline uint64_t maxNbNewPointsPerUpdate = 0;
    static inline uint64_t minNbNewPointsPerUpdate = 0;
    static inline uint64_t avgNbNewPointsPerUpdate = 0;

    static inline uint64_t maxNbNewVoxelsPerUpdate = 0;
    static inline uint64_t minNbNewVoxelsPerUpdate = 0;
    static inline uint64_t avgNbNewVoxelsPerUpdate = 0;

    static inline uint32_t maxNbNewNodesPerUpdate = 0;
    static inline uint32_t minNbNewNodesPerUpdate = 0;
    static inline uint32_t avgNbNewNodesPerUpdate = 0;

    static inline uint32_t maxNbNewGridsPerUpdate = 0;
    static inline uint32_t minNbNewGridsPerUpdate = 0;
    static inline uint32_t avgNbNewGridsPerUpdate = 0;

    static inline uint32_t maxNbNewChunksPerUpdate = 0;
    static inline uint32_t minNbNewChunksPerUpdate = 0;
    static inline uint32_t avgNbNewChunksPerUpdate = 0;

    static inline uint32_t maxNbLoadedNodesPerUpdate = 0;
    static inline uint32_t minNbLoadedNodesPerUpdate = 0;
    static inline uint32_t avgNbLoadedNodesPerUpdate = 0;

    static inline uint32_t maxNbStoredNodesPerUpdate = 0;
    static inline uint32_t minNbStoredNodesPerUpdate = 0;
    static inline uint32_t avgNbStoredNodesPerUpdate = 0;

    static inline uint32_t maxNbSplitNodesPerUpdate = 0;
    static inline uint32_t minNbSplitNodesPerUpdate = 0;
    static inline uint32_t avgNbSplitNodesPerUpdate = 0;

    static inline uint32_t maxNbDeletedNodesPerUpdate = 0;
    static inline uint32_t minNbDeletedNodesPerUpdate = 0;
    static inline uint32_t avgNbDeletedNodesPerUpdate = 0;

    static inline uint32_t maxNbDeletedGridsPerUpdate = 0;
    static inline uint32_t minNbDeletedGridsPerUpdate = 0;
    static inline uint32_t avgNbDeletedGridsPerUpdate = 0;

    static inline uint32_t maxNbDeletedChunksPerUpdate = 0;
    static inline uint32_t minNbDeletedChunksPerUpdate = 0;
    static inline uint32_t avgNbDeletedChunksPerUpdate = 0;

    static inline uint32_t maxNbUpdatesPerSecond = 0;
    static inline uint32_t minNbUpdatesPerSecond = 0;
    static inline uint32_t avgNbUpdatesPerSecond = 0;

    // Rendering counters
    static inline uint32_t visNbNodes = 0;
    static inline uint64_t visNbPoints = 0;
    static inline uint64_t visNbVoxels = 0;


    // Timers
    static inline std::chrono::time_point<std::chrono::high_resolution_clock> lastUpdateStart;
    static inline std::chrono::time_point<std::chrono::high_resolution_clock> firstUpdateStart;

    /// Mirror of the UI block of `CGlobalVariables' (same order, same types; checked in gpuVersion.cpp)
    struct UIStats {
        uint64_t nbTotalPoints;
        uint64_t nbTotalVoxels;
        uint64_t currentNbPoints;
        uint64_t currentNbVoxels;
        uint64_t nbNewPointsThisUpdate;
        uint64_t nbNewVoxelsThisUpdate;
        uint32_t nbTotalUpdates;
        uint32_t curNbNodes;
        uint32_t currentNbChunks;
        uint32_t currentNbGrids;
        uint32_t nbTotalNewNodes;
        uint32_t nbTotalNewGrids;
        uint32_t nbTotalNewChunks;
        uint32_t nbTotalDeletedNodes;
        uint32_t nbTotalDeletedGrids;
        uint32_t nbTotalDeletedChunks;
        uint32_t nbTotalLoadedNodes;
        uint32_t nbTotalSplitNodes;
        uint32_t nbTotalStoredNodes;
        uint32_t nbNewNodesThisUpdate;
        uint32_t nbLoadedNodesThisUpdate;
        uint32_t nbStoredNodesThisUpdate;
        uint32_t nbSplitNodesThisUpdate;
        uint32_t nbDeletedNodesThisUpdate;
        uint32_t nbDeletedChunksThisUpdate;
        uint32_t nbDeletedGridsThisUpdate;
        uint32_t nbNewChunksThisUpdate;
        uint32_t nbNewGridsThisUpdate;
    };
    static inline UIStats* pinnedStats = nullptr;
    static inline CUevent eventStatsReady;
    static inline bool statsCopyInFlight = false;
    static inline CUstream uiStream;

    static void update();
    static void init();
    static void destroy();
    static void kickoffStatsCopy();
};






struct GpuVersion {
	static inline CudaModularProgram* prog = nullptr;
    static inline CGlobalVariables hostStaging = {};
    static inline CUdeviceptr deviceStaging = 0;
    static inline CUstream stream;
    static inline CUstream visStream;
    static inline uint64_t totalAllocatedMemory = 0;
    static inline bool isTakingScreenshots = false;
	static inline bool mainLoopIsTerminating = false;
	static std::string getGpuMemoryUsage();
	static std::string formatMemSize(uint64_t size_bytes, uint32_t pad = 0);

    static inline std::vector<CIdAABB> parentsMap = {};
    static inline std::vector<CAABB> aabbsMap = {};

    
    // CPU cache
    /// The host cache is write-back: a node is written to disk only when it leaves `hostCache' (and
    /// only what it gained since it was last written). While it is in the cache, its host RAM copy
    /// (`updateNodes') is the reference: the octree update and the visibility update read it there
    /// (see `finishHostCacheWrite' and `refreshHostNodesReadableCounts' below)
    static inline CLRUCache* hostCache = nullptr;
    static inline CLRUCache* hostVisCache = nullptr;
    static inline std::unordered_map<CIdAABB, std::shared_ptr<HostStorageNode>> updateNodes = {};
    static inline std::unordered_map<CIdAABB, std::shared_ptr<HostStorageNode>> visibilityNodes = {};
    // static inline std::unordered_map<CIdAABB, std::shared_ptr<HostStorageNode>> persistentStoredNodes = {};
    static void updateHostCache();
    /// `hostCache->add', recording the id it evicts (see `hostCacheEvictedIds')
    static void hostCacheAdd(CIdAABB id);
    /// What `updateHostCache' has to look at (main thread only), instead of every host RAM node:
    /// the ids evicted from `hostCache' since its last pass (written and freed if they are not back
    /// in it), and the nodes created or written by `storeNodes' since (only checked to still be in
    /// the cache: write-back, a node is not written while it is cached)
    static inline std::vector<CIdAABB> hostCacheEvictedIds = {};
    static inline std::unordered_set<CIdAABB> hostCacheDirtyIds = {};
    /// Write `hostCacheToSerialise' (the nodes leaving the host cache) and record what was written in
    /// `hostCacheWrittenCounters'. Runs on its own thread; it publishes nothing and frees nothing:
    /// the main thread does it afterwards (`finishHostCacheWrite')
    static void serialiseHostCacheNodes();
    static void updateVisCache();
    static inline std::atomic<bool> isDoneUpdatingHostCache = true;
    static inline std::atomic<bool> isDoneUpdatingVisCache = true;

    // Visibility cache
    static inline std::unordered_set<CIdAABB> storedNodes = {}; 
    static inline std::unordered_set<CIdAABB> currentlyInUpdatesCache = {}; 
    static void visibilityUpdate(CuRast* editor, CUcontext* context);
    static void visibilityUpdateSort(CuRast* editor, CUcontext* context);
    static inline std::vector<std::pair<CIdAABB, float>> visibleNodes = {};
    static inline std::vector<CIdAABB> visibleNodesOrdered = {};

    static inline std::unordered_set<CIdAABB> storedNodesVisibilityCopy = {};
    static inline std::atomic<uint32_t> nbStoredNodesInVisibility = 0;
    static inline std::vector<std::mutex> storedNodesMtx = {};
    static inline std::mutex storedNodesVisibilityMtx;

    /// Number of points / voxels written on disk for each stored node
    /// Only published once the data is entirely written (protected by `storedNodesVisibilityMtx')
    /// The files are append-only, so these many elements can always be read from them afterwards
    struct StoredNodeCounters {
        uint32_t nb_points = 0;
        uint32_t nb_voxels = 0;
    };
    static inline std::unordered_map<CIdAABB, StoredNodeCounters> storedNodesVisibilityCounters = {};

    /// Counters written by the last `serialiseHostCacheNodes', in the order of `hostCacheToSerialise'
    static inline std::vector<StoredNodeCounters> hostCacheWrittenCounters = {};
    /// A write of `hostCacheToSerialise' was started and not finished on the main thread yet
    static inline bool isHostCacheWritePending = false;
    /// Nodes that left the host cache and were written: their buffers are given back to the pinned
    /// allocators once no visibility worker copies a host RAM node anymore (`visPendingUpdateNodeCopies')
    static inline std::vector<std::shared_ptr<HostStorageNode>> hostCacheToFree = {};
    /// Main thread. Once the write thread is done: publish the written nodes for the visibility
    /// update (on-disk counters, `storedNodes'), take them out of `updateNodes' (in that order: a
    /// node is always readable from RAM or from disk), and free their buffers as soon as no
    /// visibility worker can be reading them
    /// Returns true when nothing is left to do (the octree update may create host nodes again)
    static bool finishHostCacheWrite();

    /// Host RAM nodes whose counters may be ahead of their `readable_*' counts (main thread only):
    /// created for a deserialisation, or receiving device data (`storeNodes')
    static inline std::unordered_set<CIdAABB> hostNodesNotReadable = {};
    /// Main thread: when no deserialisation and no device->host copy is in progress
    /// (`visCanReadUpdateNodes'), every host RAM node is complete: its `readable_*' counts catch up
    static void refreshHostNodesReadableCounts();

    /// RAM cache of the visible nodes read from disk by the visibility updates, kept from one
    /// visibility update to the next, so that an update only reads what a node gained since (usually
    /// nothing) instead of every visible node
    /// For each file (points, voxels), an entry keeps the first `keep' elements of a uniform random
    /// permutation of the `seen' first elements of the file (inside-out Fisher-Yates, extended when
    /// the file grows): the first `quota' elements are a uniform random subset, the same from one
    /// update to the next while the node does not change. Only that prefix is stored (about twice the
    /// quota of the node), not the whole file: the elements beyond it are never needed. The random
    /// draws only depend on the node and the element index, so a larger prefix (quota increased) is
    /// rebuilt from the file identically, extended
    /// Owned by the main thread, except the entries of the current tasks, filled by the visibility
    /// workers (one task per node: one worker per entry)
    template <typename TElement>
    struct VisRamCachedFile {
        std::vector<TElement> data = {}; // first min(seen, keep) elements of the permutation
        uint32_t seen = 0;               // elements of the file processed
        uint32_t keep = 0;               // length of the prefix kept
    };
    struct VisRamCacheEntry {
        VisRamCachedFile<CPoint> points = {};
        VisRamCachedFile<CVoxel> voxels = {};
        uint64_t last_used = 0;   // visibility update that used it last (`visRamCacheGeneration')
    };
    static inline std::unordered_map<CIdAABB, std::unique_ptr<VisRamCacheEntry>> visRamCache = {};
    static inline uint64_t visRamCacheGeneration = 0;
    /// Memory of the cache (points and voxels): `OocSimLodSettings::VIS_RAM_CACHE_MAX_BYTES'; beyond
    /// it, the least recently used entries are freed
    /// Prefix kept for a quota q: max(`OocSimLodSettings::VIS_RAM_CACHE_MIN_KEEP', bit_ceil(2 q)), the
    /// whole file at most; shrunk back when it is more than 4 times what the quota needs

    /// Host RAM nodes (`updateNodes') the visibility workers still have to copy into the staging
    /// buffers (they copy them first). The buffers of the nodes that left the host cache are only
    /// freed once it is 0 (`finishHostCacheWrite')
    static inline std::atomic<uint32_t> visPendingUpdateNodeCopies{0};

    /// Threads reading / writing the node files of the octree update (deserialisation for loading and
    /// storing, serialisation of the host cache): `OocSimLodSettings::DISK_IO_THREADS', 0 = automatic
    /// (a quarter of the cores, 1 to 8)
    static uint32_t nbDiskIoThreads();
    /// True while the octree update reads or writes node files: no visibility update is started
    /// meanwhile (their disk reads would slow it down), unless the last one started more than
    /// `OocSimLodSettings::VIS_MAX_PAUSE_SECONDS' ago (the rendering must keep up while the update
    /// loads a lot)
    static bool isOctreeDoingDiskWork();
    static inline std::chrono::steady_clock::time_point lastVisibilityUpdateStart = {};

    /// One visible node streamed into the rendering staging buffers
    /// Built on the main thread, filled by the workers (`*_written'), read back on the main thread
    /// A node's elements come from two places: the first `nb_*_on_disk' from its files, the next
    /// `nb_*_in_ram' from its host RAM window (`update_node', elements written to disk later, when the
    /// node leaves the host cache). Its quota is shared between both parts: the RAM part is written
    /// first, at `*_offset' (first pass of the workers), the disk part right after it (second pass)
    struct VisibilityTask {
        CIdAABB id = CINVALID_ID;
        uint32_t nb_points_on_disk = 0;
        uint32_t nb_points_in_ram = 0;
        uint32_t nb_points_total = 0;       // nb_points_on_disk + nb_points_in_ram (shares the budget)
        uint32_t points_quota = 0;          // number of points this node may write
        uint32_t points_quota_ram = 0;      // part of it taken from the RAM window (<= nb_points_in_ram)
        uint64_t points_offset = 0;         // reserved slot in `visStagingPoints'
        uint32_t points_written_ram = 0;    // written by the RAM part (first pass)
        uint32_t points_written = 0;        // number of points really written (<= quota)
        uint32_t nb_voxels_on_disk = 0;
        uint32_t nb_voxels_in_ram = 0;
        uint32_t nb_voxels_total = 0;
        uint32_t voxels_quota = 0;
        uint32_t voxels_quota_ram = 0;
        uint64_t voxels_offset = 0;
        uint32_t voxels_written_ram = 0;
        uint32_t voxels_written = 0;
        float points_priority = 0.0f;   // device visibility priority (~ screen area covered, in pixels)
        float voxels_priority = 0.0f;   // negative: outside the view (out-of-view fallback, see below)

        /// Out-of-view fallback ("Vox", see part 3 in render.cu): only the voxels lying in these
        /// octants are drawn (bit i: octant of child i; the octant of a voxel comes from its cell,
        /// `CVoxel::getOctant'); 0 = every voxel (cut node)
        uint8_t voxels_child_mask = 0;

        /// The node has a RAM part (`nb_*_in_ram' > 0): its host RAM copy in `updateNodes', kept alive
        /// until the first pass copied it; its buffers are not given back to the pinned allocators
        /// meanwhile, see `visPendingUpdateNodeCopies'
        bool is_from_update_node = false;
        std::shared_ptr<const HostStorageNode> update_node = nullptr;
        /// RAM cache entry of the node (disk part), filled by the worker of the second pass
        VisRamCacheEntry* ram_cache_entry = nullptr;
    };
    static inline std::vector<VisibilityTask> visibilityTasks = {};

    /// Voxel splats (see `drawSplat' in render.cu): every voxel is drawn as a camera-facing square of
    /// side `OocSimLodSettings::VOXEL_SPLAT_SCALE' voxel cells (cell = root size /
    /// (GRID_SIZE_PER_DIMENSION * 2^level of its node)), at most `VOXEL_SPLAT_MAX_PIXELS' pixels wide
    /// The host rendering buffers come with the node of every voxel (`renderedVoxelSegments', see
    /// `CGlobalVariables'): its bounding box gives the position and the size of the voxel

    /// Draw the batches already on the device but not inserted in the octree yet (see
    /// `kernel_draw_pending_batches' in render.cu): the points near the camera show up as soon as
    /// they are sent, instead of when the device update inserting them completes
    /// (`OocSimLodSettings::DRAW_PENDING_BATCHES', at most `PENDING_BATCHES_MAX_POINTS' per frame)

    /// Pinned staging buffers, as large as the device rendering buffers
    /// Written by the visibility workers, read by a single batched copy to the device
    static inline CPoint* visStagingPoints = nullptr;
    static inline CVoxel* visStagingVoxels = nullptr;
    /// Node of every voxel of the staging buffer, once packed (see `CGlobalVariables::renderedVoxelSegments'):
    /// built on the main thread when the content is published, sent with the voxels
    static inline CVoxelSegment* visStagingVoxelSegments = nullptr;
    static inline uint32_t* visStagingVoxelTiles = nullptr;
    static inline uint32_t* visStagingNbVoxelSegments = nullptr;

    /// True between the copy of a new visibility content and its publication to the device
    static inline bool isVisibilityPublishPending = false;

    /// Hand-over of the evicted nodes between the device (`kernel_draw_evicted_nodes', drawing the
    /// last evicted batch from the exchange buffers) and the host rendering buffers (main thread only)
    /// Generation of the last eviction pass (mirrored on the device as `evictionGeneration')
    static inline uint32_t evictionGenerationHost = 0;
    /// Generation whose data `storeNodes' has brought into host RAM, and what each of its nodes must
    /// show in the host rendering buffers: id, points, voxels (counts of its host RAM copy)
    static inline uint32_t evictionGenerationStored = 0;
    struct EvictedNodeRecord {
        CIdAABB id = CINVALID_ID;
        uint32_t nb_points = 0;
        uint32_t nb_voxels = 0;
    };
    static inline std::vector<EvictedNodeRecord> lastEvictedNodes = {};
    /// Every eviction generation not covered by published host buffers yet, in order (the device draws
    /// them: the last one from the exchange buffers, the older ones from their retained copies)
    struct PendingEviction {
        uint32_t generation = 0;
        std::vector<EvictedNodeRecord> records = {};
    };
    static inline std::deque<PendingEviction> pendingEvictions = {};
    /// The last eviction pass did not fit entirely in the retained copies (device
    /// `retainOverflowGeneration'): the next pass must wait for its coverage
    static inline bool hasLastPassOverflowedRetain = false;
    /// Capacity of the retained copies of evicted nodes: `OocSimLodSettings::EVICTION_RETAIN_BUDGET_FRACTION'
    /// of the rendering budgets (points and voxels); entries: `EVICTION_RETAIN_NB_PASSES' eviction
    /// passes of `maxNbNodesExchanged' nodes
    /// Latest eviction generation shown by the visibility update in progress (computed in
    /// `visibilityUpdateSort', published with its buffers in `updateVisCache')
    static inline uint32_t visCoveredEvictionGeneration = 0;
    /// Latest eviction generation covered by the host rendering buffers the device draws (host mirror
    /// of `CGlobalVariables::coveredEvictionGeneration', set when they are published)
    static inline uint32_t publishedCoveredGeneration = 0;
    /// The device draws the evicted batches until published host buffers cover them: the last one from
    /// the exchange buffers (overwritten by the next eviction pass), the older ones from their retained
    /// copies. When the last pass did not fit in the retained copies, a new pass waits until it is
    /// covered. True while the cache update is held back for that reason (nothing else of the update
    /// pipeline may run meanwhile: the update in progress is not finished before its cache update)
    static inline bool isCacheUpdateDeferred = false;
    static inline std::chrono::time_point<std::chrono::steady_clock> cacheUpdateDeferredSince;
    /// Longest wait for the coverage before evicting anyway (the device then stops drawing the
    /// previous batch: holes until the next publication): `OocSimLodSettings::MAX_EVICTION_COVERAGE_WAIT_SECONDS'
    /// True if a new eviction pass would overwrite a batch the device still has to draw
    static bool isWaitingForEvictionCoverage();

    static inline std::vector<CUdeviceptr> exchangedPointsPointers = {};
	static inline std::vector<CUdeviceptr> exchangedVoxelsPointers = {};
	static inline std::vector<CUdeviceptr> exchangedLoadedVoxelsPointers = {};
	static inline void* batchesToAddPointsPointers = nullptr;
    static inline void* nbExchangedNodes = nullptr;
    static inline uint32_t curNbNodes = 0;
    static inline const uint32_t RESET = 0;

    static inline void* isDoneLoading = nullptr;
    static inline void* isDoneStoring = nullptr;
    static inline void* isDoneIterating = nullptr;
    static inline bool firstBatchSent = false;
    static inline void* isInitialised = nullptr;
    static inline void* isUpdating = nullptr;

    /// Atomic: written by the visibility coordinator thread, read by the main thread
    static inline std::atomic<bool> isDoneDeserializingForVisibility{true};
    static inline bool hasStartedVisibilityUpdate = false;
    static inline uint32_t nbVisibleNodesVisibilityUpdate = 0;
    static inline std::atomic<bool> isDoneDeserializingForLoading = true;
    static inline bool hasStartedLoading = false;
    static inline uint32_t nbNodesToLoad = 0;
    static inline std::vector<CIdAABB> loadingToDeserialise = {};
    static inline std::atomic<bool> isDoneDeserializingForStoring = true;
    static inline bool hasStartedStoring = false;
    static inline uint32_t nbNodesToStore = 0;
    static inline std::vector<CIdAABB> storingToDeserialise = {};
    static inline bool isDoneSynchronisingStoringCompleteEvent = true;
    static inline bool hasStartedSynchronisingStoringCompleteEvent = false;
    /// Nodes leaving the host cache, written by `serialiseHostCacheNodes' (they stay in `updateNodes'
    /// until `finishHostCacheWrite')
    static inline std::vector<std::shared_ptr<HostStorageNode>> hostCacheToSerialise = {};
    static inline bool isDoneSynchronisingEventInStoreNodes = true;
    static inline bool hasStartedSynchronisingEventInStoreNodes = false;

    static inline bool hasStartedSortingVisibleNodes = false;
    static inline bool isDoneSortingVisibleNodes = true;

    static inline void* exchangedIds = nullptr;
    static inline void* exchangedParentsIds = nullptr;
    static inline void* exchangedChildrenIds = nullptr;
    static inline void* exchangedAABBs = nullptr;
    static inline void* exchangedPointsCounters = nullptr;
    static inline void* exchangedVoxelsCounters = nullptr;

    static inline CUevent eventVisPreprocess;
    static inline void* exchangedAABBIndicesVisPoints = nullptr;
    static inline void* exchangedAABBIndicesVisVoxels = nullptr;
    /// Priorities of the visible lists (sorted on device together with the ids above)
    static inline void* exchangedPrioritiesVisPoints = nullptr;
    static inline void* exchangedPrioritiesVisVoxels = nullptr;
    /// Child masks of the voxels list (see `CGlobalVariables::exchangedVisVoxelsChildMasks')
    static inline void* exchangedVisVoxelsChildMasks = nullptr;
    static inline void* nbNodesExchangedVisPoints = nullptr;
    static inline void* nbNodesExchangedVisVoxels = nullptr;
    static inline void* curNbVisNode = nullptr;

    static inline CRenderTarget renderingTarget = {};
    static inline CRenderingSettings renderingSettings = {};

    static inline void* nbRenderedPoints = nullptr;
    static inline void* nbRenderedVoxels = nullptr;

    static inline void* nbRenderedPoints2 = nullptr;
    static inline void* nbRenderedVoxels2 = nullptr;
    static inline bool isUsingSecondRenderingBuffer = false;

    static inline uint32_t randomOffset = 0;


    static inline CUevent eventVisibilityUpdateComplete;
    static inline CUevent eventLoadingComplete;
    static inline std::vector<CUmemcpyAttributes> batchLoadingAttributes = {
        CUmemcpyAttributes {
            .srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
            .srcLocHint = {CU_MEM_LOCATION_TYPE_HOST},
            .dstLocHint = {CU_MEM_LOCATION_TYPE_DEVICE},
            .flags = CU_MEMCPY_FLAG_DEFAULT
        }
    };
    static inline std::vector<uint64_t> batchLoadingAttributesIndices = {0};

    static inline CUevent eventStoringComplete;
    static inline std::vector<CUmemcpyAttributes> batchStoringAttributes = {
        CUmemcpyAttributes {
            .srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
            .srcLocHint = {CU_MEM_LOCATION_TYPE_DEVICE},
            .dstLocHint = {CU_MEM_LOCATION_TYPE_HOST},
            .flags = CU_MEMCPY_FLAG_DEFAULT
        }
    };
    static inline std::vector<uint64_t> batchStoringAttributesIndices = {0};
    static void storeNodes(uint32_t nb_nodes_to_store);
    static inline uint32_t MAX_NB_VOXELS = 0;

    static void shuffleForRendering(HostStorageNode* node, CIdAABB id);

    static inline std::thread* updateHostCacheComplete = nullptr;



    /// Initialises everything needed on device memory
    static void init(CuRast* editor, CUcontext* context);
    static void destroy(CuRast* editor, CUcontext* context);
    static void updateOctree(CuRast* editor, CUcontext* context);
    static void renderOctree(RenderTarget& target);

    /// For another project
    static void takeRandomScreenShots();
    static void takeSingleScreenShot();
    static void tryToCenterTarget();

    friend LoaderGpuVersion;

    template<typename T>
    static T* alloc(uint32_t size){
        CUdeviceptr new_ptr = 0;
        uint64_t real_size = size * sizeof(T);
        totalAllocatedMemory += real_size;
        CURuntime::assertCudaSuccess(cuMemAlloc(&new_ptr, real_size));
        pointers.push_back(new_ptr);
        return reinterpret_cast<T*>(new_ptr);
    }

    template<typename T>
    static void* allocHost(uint32_t size){
        void* new_ptr = nullptr;
        uint64_t real_size = size * sizeof(T);
        CURuntime::assertCudaSuccess(cuMemAllocHost(&new_ptr, real_size));
        return new_ptr;
    }

    /// Initial bounding volume of the octree root: the union of the header bounding boxes of the first
    /// dropped files, proposed by the loader before any of their batches is queued. Only the first
    /// proposal is kept; `octreeUpdateInit' uses it instead of measuring the first points
    static inline std::mutex initialAABBMtx;
    static inline bool hasInitialAABB = false;
    static inline glm::vec3 initialAABBMins = glm::vec3(0.0f);
    static inline glm::vec3 initialAABBMaxs = glm::vec3(0.0f);
    static void proposeInitialAABB(const glm::vec3& mins, const glm::vec3& maxs);

    private:
        static void octreeUpdateInit(CuRast* editor, CUcontext* context);
        static void octreeUpdateBottomUp(CuRast* editor, CUcontext* context);
        static void octreeUpdateSimLOD(CuRast* editor, CUcontext* context);
        static void octreeUpdateSimLODLoad(CuRast* editor, CUcontext* context);
        static void octreeUpdateSimLODCountSplit(CuRast* editor, CUcontext* context);
        static void octreeUpdateSimLODVoxelSampling(CuRast* editor, CUcontext* context);
        static void octreeUpdateSimLODInsertion(CuRast* editor, CUcontext* context);
        static void octreeUpdateCacheUpdate(CuRast* editor, CUcontext* context);
        static void octreeUpdateFillNewGrids(CuRast* editor, CUcontext* context);


        static inline std::vector<CUdeviceptr> pointers = {};
        static void initConstraints(CuRast* editor, CUcontext* context);
        static void initHostSide(CuRast* editor, CUcontext* context);
        static void initBuffers(CuRast* editor, CUcontext* context);
        static void initAllocators(CuRast* editor, CUcontext* context, CUstream* stream);


        template<typename T>
        static CAllocatorPool<T>* allocAllocator(uint32_t size, AllocatorId type){
            uint64_t real_size = 0;

            CUdeviceptr allocator_ptr = 0;
            real_size = sizeof(CAllocatorPool<T>);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&allocator_ptr, real_size));
            pointers.push_back(allocator_ptr);

            uint64_t alignment   = alignof(T);
            uint64_t aligned_size = sizeof(T) + ((alignment - (sizeof(T) % alignment)) % alignment);
            CUdeviceptr pool_ptr = 0;
            real_size = (uint64_t)size * aligned_size;
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&pool_ptr, real_size));
            pointers.push_back(pool_ptr);

            CUdeviceptr stack_ptr = 0;
            real_size = (uint64_t)size * sizeof(uint32_t);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&stack_ptr, real_size));
            pointers.push_back(stack_ptr);

            std::vector<uint32_t> stack_host(size);
            for(uint32_t i = 0; i < size; i++) {
                stack_host[i] = i;   // slot i owns &memory[i]
            }
            CURuntime::assertCudaSuccess(
                cuMemcpyHtoD(stack_ptr, stack_host.data(), real_size)
            );

            CAllocatorPool<T> tmp(size, type);
            tmp.allocated_memory     = reinterpret_cast<T*>(pool_ptr);
            tmp.free_stack = reinterpret_cast<uint32_t*>(stack_ptr);
            tmp.free_top   = size;
            tmp.nb_allocated_elements = 0;

            CURuntime::assertCudaSuccess(
                cuMemcpyHtoD(allocator_ptr, &tmp, sizeof(CAllocatorPool<T>))
            );

            return reinterpret_cast<CAllocatorPool<T>*>(allocator_ptr);
        };
};



enum HostStorageNodeOwner {
    Update,
    Visibility,
};
    

template <typename T>
struct PinnedMemoryAllocator {
    std::unordered_set<T*> free_data = {};
    std::unordered_set<T*> used_data = {};
    T* allocate(const HostStorageNodeOwner& owner) {
        for(T* data : free_data){
            used_data.insert(data);
            free_data.erase(data);
            return data;
        }
        switch(owner){
            case Update:
                printf("ERROR: no more %s can be created for the update cache\n", typeid(T).name());
                break;
            case Visibility:
                printf("ERROR: no more %s can be created for the visibility cache\n", typeid(T).name());
                break;
        }
        throw(EXIT_FAILURE);
    }
    void deallocate(T* data) {
        used_data.erase(data);
        free_data.insert(data);
    }
    void init(uint32_t node_count, uint32_t data_count) {
        for(uint32_t i = 0; i < node_count; i++){
            T* allocable = (T*)GpuVersion::allocHost<T>(data_count);
            free_data.insert(allocable);
        }
    }
    void destroy(){
        for(T* data : free_data){
            CURuntime::assertCudaSuccess(cuMemFreeHost(data));
        }
        for(T* data : used_data){
            CURuntime::assertCudaSuccess(cuMemFreeHost(data));
        }
    }
};


struct HostStorageNode {
	COctreeNode node = {};
	CPoint* points = nullptr;
	/// Voxels: their cells are also the occupancy grid of the node (no separate indices)
	CVoxel* voxels = nullptr;

    /// The buffers are indexed like the node's elements (element i at `points[i]'), but only hold the
    /// window [`*_valid_from', `node.*_counter'): the elements before it are on disk only. A node
    /// enters host RAM with empty windows starting at its counts on disk (`attach': nothing is read),
    /// the device data is appended to them (`storeNodes'), and the whole window is written when the
    /// node leaves the host cache. The points are never read back: only the visibility update needs
    /// them, and it reads the disk part from the files. The voxels are read back (into
    /// [0, voxels_valid_from), then `voxels_valid_from' = 0) when the node is loaded to the device
    uint32_t points_valid_from = 0;
    uint32_t voxels_valid_from = 0;

    /// End of the part of the windows known to be complete (main thread only, see
    /// `GpuVersion::refreshHostNodesReadableCounts'): the visibility update may copy the elements
    /// from the counts on disk to these at any time. They are never written again: the device->host
    /// copies of `storeNodes' append after them, the voxels read back go before them
    uint32_t readable_points = 0;
    uint32_t readable_voxels = 0;

    static inline PinnedMemoryAllocator<CPoint> points_allocator_update = {};
    static inline PinnedMemoryAllocator<CVoxel> voxels_allocator_update = {};

    static inline PinnedMemoryAllocator<CPoint> points_allocator_visibility = {};
    static inline PinnedMemoryAllocator<CVoxel> voxels_allocator_visibility = {};

    HostStorageNodeOwner owner;

    static void init();
    static void destroy();
    void deallocate();
    HostStorageNode(const HostStorageNodeOwner& owner);
};