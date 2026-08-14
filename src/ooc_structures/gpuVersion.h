#pragma once

#include "globals.h"
#include "../kernels/ooc/GpuVersionInterface.h"
#include "../kernels/ooc/GpuVersionAllocator.h"



struct GpuVersionUI {
    static inline uint32_t lastNbTotalUpdates = 0;
    static inline uint32_t nbTotalUpdates = 0;

    // Current update values
    static inline uint32_t nbNewPointsThisUpdate = 0;
	static inline uint32_t nbNewVoxelsThisUpdate = 0;
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
    static inline uint32_t currentNbPoints = 0;
    static inline uint32_t currentNbVoxels = 0;

    static inline uint32_t nbTotalPoints = 0;
    static inline uint32_t nbTotalVoxels = 0;
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
    static inline uint32_t maxNbNewPointsPerUpdate = 0;
    static inline uint32_t minNbNewPointsPerUpdate = 0;
    static inline uint32_t avgNbNewPointsPerUpdate = 0;

    static inline uint32_t maxNbNewVoxelsPerUpdate = 0;
    static inline uint32_t minNbNewVoxelsPerUpdate = 0;
    static inline uint32_t avgNbNewVoxelsPerUpdate = 0;

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
    static inline uint32_t visNbPoints = 0;
    static inline uint32_t visNbVoxels = 0;


    // Timers
    static inline std::chrono::time_point<std::chrono::high_resolution_clock> lastUpdateStart;

    static void update();
};






struct GpuVersion {
	static inline CudaModularProgram* prog = nullptr;
    static inline CGlobalVariables hostStaging = {};
    static inline CUdeviceptr deviceStaging = 0;
    static inline CUstream stream;
    static inline uint64_t totalAllocatedMemory = 0;

    static inline std::vector<CIdAABB> relationshipMap = {};

    
    // CPU cache
    // static inline std::unordered_map<CIdAABB, CAABB> storedNodes = {}; 
    // static inline CLRUCache* hostCache = nullptr;
    // static inline std::unordered_set<CIdAABB> recentlyUsedNodesFromUpdates = {};
    // static inline std::unordered_set<CIdAABB> removedNodes = {};
    // static inline std::unordered_map<CIdAABB, std::shared_ptr<HostStorageNode>> persistentStoredNodes = {};
    // static inline std::mutex syncAABBStorageAccessMtx;
    // static inline std::mutex syncHostStorageNodesAccessMtx;
    // static inline std::mutex syncVisibilityUpdateMtx;

    // static inline std::vector<CIdAABB> previouslyVisible = {};
    // static inline std::vector<std::shared_ptr<HostStorageNode>> newlyVisible = {};
    // static inline std::vector<bool> newlyVisibleToDelete = {};

    // static void updateHostCache();
    // static void visibilityUpdate(CuRast* editor, CUcontext* context);





    static inline std::mutex renderSubmissionMutex;
	static inline void* exchangedPointsPointers = nullptr;
	static inline void* exchangedVoxelsPointers = nullptr;
	static inline void* batchesToAddPointsPointers = nullptr;
    static inline void* nbExchangedNodes = nullptr;
    static inline void* isTemporarySwitching = nullptr;
    static inline uint32_t curNbNodes = 0;

    static inline CUevent eventUpdateCompleted;
    static inline CUevent eventSwapCompleted;
    static inline CUevent eventRenderingStreamInformed;

    /// Initialises everything needed on device memory
    static void init(CuRast* editor, CUcontext* context);
    static void destroy(CuRast* editor, CUcontext* context);
    static void updateOctree(CuRast* editor, CUcontext* context);
    static void renderOctree(RenderTarget& target);

    /// For another project
    static void takeRandomScreenShots();

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
        static void initHostSide(CuRast* editor, CUcontext* context);
        static void initBuffers(CuRast* editor, CUcontext* context);
        static void initAllocators(CuRast* editor, CUcontext* context, CUstream* stream);

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
        static CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>* allocAllocatorElements(
            uint32_t size, CUdeviceptr allocated_memory_base, uint64_t aligned_size,
            CUstream* stream, std::vector<CUdeviceptr>& out_entries_it_ptr
        ){
            using PoolEntry     = typename CAllocatorPool<T>::Entry;
            using List          = CDoubleLinkedList<PoolEntry*>;
            using ListIterator  = typename List::Iterator;

            uint64_t real_size = 0;

            CUdeviceptr elements_ptr = 0;
            real_size = sizeof(List);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_ptr, real_size));

            CUdeviceptr elements_first_ptr = 0, elements_last_ptr = 0;
            real_size = sizeof(typename List::FirstEntry);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_first_ptr, real_size));
            real_size = sizeof(typename List::LastEntry);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_last_ptr, real_size));

            // One contiguous block for ALL entries/iterators instead of `size` separate cuMemAlloc calls
            CUdeviceptr entries_base_ptr = 0, entries_it_base_ptr = 0;
            if(size > 0){
                real_size = (uint64_t)size * sizeof(PoolEntry);
                totalAllocatedMemory += real_size;
                CURuntime::assertCudaSuccess(cuMemAlloc(&entries_base_ptr, real_size));
                real_size = (uint64_t)size * sizeof(ListIterator);
                totalAllocatedMemory += real_size;
                CURuntime::assertCudaSuccess(cuMemAlloc(&entries_it_base_ptr, real_size));
            }

            std::vector<PoolEntry> entries_host(size);
            std::vector<ListIterator> entries_it_host(size);
            out_entries_it_ptr.resize(size);

            for(uint32_t i=0; i<size; i++){
                CUdeviceptr entry_ptr    = entries_base_ptr    + (uint64_t)i * sizeof(PoolEntry);
                CUdeviceptr entry_it_ptr = entries_it_base_ptr + (uint64_t)i * sizeof(ListIterator);

                entries_host[i].is_free = true;
                // Pre-link the slot to its address in the pool. `initAllocatorPool` no longer needs to set this itself.
                entries_host[i].value = reinterpret_cast<T*>(allocated_memory_base + (uint64_t)i * aligned_size);

                entries_it_host[i].value = reinterpret_cast<PoolEntry*>(entry_ptr);
                entries_it_host[i].prev  = (i == 0)        ? nullptr : reinterpret_cast<ListIterator*>(entries_it_base_ptr + (uint64_t)(i-1) * sizeof(ListIterator));
                entries_it_host[i].next  = (i+1 < size)    ? reinterpret_cast<ListIterator*>(entries_it_base_ptr + (uint64_t)(i+1) * sizeof(ListIterator)) : nullptr;

                out_entries_it_ptr[i] = entry_it_ptr;
            }

            typename List::FirstEntry elements_first_host = {};
            typename List::LastEntry  elements_last_host  = {};
            if(size > 0){
                elements_first_host.next = reinterpret_cast<ListIterator*>(entries_it_base_ptr);
                elements_last_host.prev  = reinterpret_cast<ListIterator*>(entries_it_base_ptr + (uint64_t)(size-1) * sizeof(ListIterator));
            }
            List elements_host = {};
            elements_host.size = size;
            elements_host.first = reinterpret_cast<typename List::FirstEntry*>(elements_first_ptr);
            elements_host.last  = reinterpret_cast<typename List::LastEntry*>(elements_last_ptr);

            if(size > 0){
                CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(entries_base_ptr,    entries_host.data(),    size * sizeof(PoolEntry),   *stream));
                CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(entries_it_base_ptr, entries_it_host.data(), size * sizeof(ListIterator),*stream));
            }
            CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(elements_last_ptr,  &elements_last_host,  sizeof(elements_last_host),  *stream));
            CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(elements_first_ptr, &elements_first_host, sizeof(elements_first_host), *stream));
            CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(elements_ptr,      &elements_host,       sizeof(elements_host),       *stream));

            if(size > 0){
                pointers.push_back(entries_it_base_ptr);
                pointers.push_back(entries_base_ptr);
            }
            pointers.push_back(elements_last_ptr);
            pointers.push_back(elements_first_ptr);
            pointers.push_back(elements_ptr);

            return reinterpret_cast<List*>(elements_ptr);
        }

        template<typename T>
        static CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>*
        allocAllocatorElementsMap(
            uint32_t size, CUdeviceptr allocated_memory_base, uint64_t aligned_size,
            const std::vector<CUdeviceptr>& entries_it_ptr, CUstream* stream
        ){
            using ElemIterator  = typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator;
            using Map           = CHashMap<T*, ElemIterator*>;
            using MapEntry       = typename Map::Entry;
            using MapList        = CDoubleLinkedList<MapEntry>;
            using MapListIterator= typename MapList::Iterator;

            uint64_t capacity = size; // matches original convention: map capacity == pool size
            uint64_t real_size = 0;

            CUdeviceptr map_ptr = 0;
            real_size = sizeof(Map);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&map_ptr, real_size));

            CUdeviceptr buckets_ptr = 0;
            real_size = capacity * sizeof(MapList);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&buckets_ptr, real_size));

            CUdeviceptr bucket_firsts_ptr = 0, bucket_lasts_ptr = 0;
            real_size = capacity * sizeof(typename MapList::FirstEntry);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&bucket_firsts_ptr, real_size));
            real_size = capacity * sizeof(typename MapList::LastEntry);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&bucket_lasts_ptr, real_size));

            CUdeviceptr nodes_ptr = 0;
            if(size > 0){
                real_size = (uint64_t)size * sizeof(MapListIterator);
                totalAllocatedMemory += real_size;
                CURuntime::assertCudaSuccess(cuMemAlloc(&nodes_ptr, real_size));
            }

            // Same hash the device would compute for each slot's T* key (pure arithmetic,
            // no CUDA-only syntax, so it's callable from host code as-is)
            std::vector<uint64_t> murmurs(size);
            std::vector<std::vector<uint32_t>> bucket_members(capacity);
            for(uint32_t i=0; i<size; i++){
                T* key_ptr = reinterpret_cast<T*>(allocated_memory_base + (uint64_t)i * aligned_size);
                uint64_t murmur = Map::hashMurmur(key_ptr);
                murmurs[i] = murmur;
                bucket_members[murmur % capacity].push_back(i);
            }

            std::vector<MapList> buckets_host(capacity);
            std::vector<typename MapList::FirstEntry> firsts_host(capacity);
            std::vector<typename MapList::LastEntry> lasts_host(capacity);
            std::vector<MapListIterator> nodes_host(size);

            uint32_t cursor = 0;
            for(uint64_t b=0; b<capacity; b++){
                auto* first_dev = reinterpret_cast<typename MapList::FirstEntry*>(bucket_firsts_ptr) + b;
                auto* last_dev  = reinterpret_cast<typename MapList::LastEntry*>(bucket_lasts_ptr) + b;

                buckets_host[b].first = first_dev;
                buckets_host[b].last  = last_dev;
                buckets_host[b].size  = (uint32_t)bucket_members[b].size();
                firsts_host[b].next = nullptr;
                lasts_host[b].prev  = nullptr;

                MapListIterator* prev_dev = nullptr;
                for(uint32_t pool_index : bucket_members[b]){
                    MapListIterator* node_dev = reinterpret_cast<MapListIterator*>(nodes_ptr) + cursor;
                    MapListIterator& node_host = nodes_host[cursor];

                    node_host.value.element  = reinterpret_cast<ElemIterator*>(entries_it_ptr[pool_index]);
                    node_host.value.real_key = reinterpret_cast<T*>(allocated_memory_base + (uint64_t)pool_index * aligned_size);
                    node_host.value.key      = murmurs[pool_index];
                    node_host.prev = prev_dev;
                    node_host.next = nullptr;

                    if(prev_dev){ nodes_host[cursor-1].next = node_dev; }
                    else        { firsts_host[b].next = node_dev; }

                    prev_dev = node_dev;
                    cursor++;
                }
                lasts_host[b].prev = prev_dev;
            }

            Map map_host = {};
            map_host.capacity = capacity;
            map_host.size = size;          // every slot is inserted exactly once, up front
            map_host.elements = reinterpret_cast<MapList*>(buckets_ptr);

            if(size > 0){
                CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(nodes_ptr, nodes_host.data(), size * sizeof(MapListIterator), *stream));
            }
            CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(bucket_firsts_ptr, firsts_host.data(), capacity * sizeof(typename MapList::FirstEntry), *stream));
            CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(bucket_lasts_ptr,  lasts_host.data(),  capacity * sizeof(typename MapList::LastEntry),  *stream));
            CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(buckets_ptr, buckets_host.data(), capacity * sizeof(MapList), *stream));
            CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(map_ptr, &map_host, sizeof(Map), *stream));

            if(size > 0){ pointers.push_back(nodes_ptr); }
            pointers.push_back(bucket_lasts_ptr);
            pointers.push_back(bucket_firsts_ptr);
            pointers.push_back(buckets_ptr);
            pointers.push_back(map_ptr);

            return reinterpret_cast<Map*>(map_ptr);
        }

        template<typename T>
        static CAllocatorPool<T>* allocAllocator(uint32_t size, AllocatorId type, CUstream* stream){
            uint64_t real_size = 0;

            CUdeviceptr allocator_ptr = 0;
            real_size = sizeof(CAllocatorPool<T>);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&allocator_ptr, real_size));
            CAllocatorPool<T> tmp = CAllocatorPool<T>(size, type);

            // Allocate the memory pool FIRST so every slot's address is known up front
            uint64_t alignment = alignof(T);
            uint64_t aligned_size = sizeof(T) + ((alignment - (sizeof(T) % alignment)) % alignment);
            CUdeviceptr allocation_pool_ptr = 0;
            real_size = (uint64_t)size * aligned_size;
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&allocation_pool_ptr, real_size));
            tmp.allocated_memory = reinterpret_cast<T*>(allocation_pool_ptr);

            std::vector<CUdeviceptr> entries_it_ptr;
            tmp.elements = allocAllocatorElements<T>(size, allocation_pool_ptr, aligned_size, stream, entries_it_ptr);

            // Fully pre-built map — no device malloc/hashing needed at init anymore
            tmp.elements_map = allocAllocatorElementsMap<T>(size, allocation_pool_ptr, aligned_size, entries_it_ptr, stream);

            CUdeviceptr deallocation_pool_ptr = 0;
            real_size = size * sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&deallocation_pool_ptr, real_size));
            CURuntime::assertCudaSuccess(cuMemsetD8Async(deallocation_pool_ptr, 0, real_size, *stream));
            tmp.deallocated_memory = reinterpret_cast<typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator**>(deallocation_pool_ptr);

            CURuntime::assertCudaSuccess(cuMemcpyHtoD(allocator_ptr, &tmp, sizeof(CAllocatorPool<T>)));

            pointers.push_back(deallocation_pool_ptr);
            pointers.push_back(allocation_pool_ptr);
            pointers.push_back(allocator_ptr);
            return reinterpret_cast<CAllocatorPool<T>*>(allocator_ptr);
        };
};