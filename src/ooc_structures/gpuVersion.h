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

    
    // CPU cache
    static inline std::unordered_map<CIdAABB, CAABB> storedNodes = {}; 
    static inline CLRUCache* hostCache = nullptr;
    static inline std::unordered_set<CIdAABB> recentlyUsedNodesFromUpdates = {};
    static inline std::unordered_set<CIdAABB> removedNodes = {};
    static inline std::unordered_map<CIdAABB, HostStorageNode*> persistentStoredNodes = {};
    static inline std::mutex syncAABBStorageAccessMtx;
    static inline std::mutex syncHostStorageNodesAccessMtx;
    static inline std::mutex syncVisibilityUpdateMtx;

    static inline std::vector<CIdAABB> previouslyVisible = {};
    static inline std::vector<HostStorageNode*> newlyVisible = {};

    static void updateHostCache();
    static void visibilityUpdate(CuRast* editor, CUcontext* context);





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
        static CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>* allocAllocatorElements(uint32_t size, CUstream* stream){
            uint64_t real_size = 0;

            // Allocate main elements
            CUdeviceptr elements_ptr = 0;
            real_size = sizeof(CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_ptr, real_size));
            
            CUdeviceptr elements_first_ptr = 0;
            CUdeviceptr elements_last_ptr = 0;
            real_size = sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::FirstEntry);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_first_ptr, real_size));
            real_size = sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::LastEntry);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_last_ptr, real_size));

            // Allocate list entries
            std::vector<CUdeviceptr> entries_ptr = {};
            std::vector<CUdeviceptr> entries_it_ptr = {};
            std::vector<typename CAllocatorPool<T>::Entry> entries_host = {};
            std::vector<typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator> entries_it_host = {};
            for(uint32_t i=0; i<size; i++){
                CUdeviceptr new_entry_ptr = 0;
                CUdeviceptr new_entry_it_ptr = 0;
                real_size = sizeof(typename CAllocatorPool<T>::Entry);
                totalAllocatedMemory += real_size;
                CURuntime::assertCudaSuccess(cuMemAlloc(&new_entry_ptr, real_size));
                real_size = sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator);
                totalAllocatedMemory += real_size;
                CURuntime::assertCudaSuccess(cuMemAlloc(&new_entry_it_ptr, real_size));
                typename CAllocatorPool<T>::Entry new_entry_host = {};
                typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator new_entry_it_host = {};
                new_entry_it_host.value = reinterpret_cast<typename CAllocatorPool<T>::Entry*>(new_entry_ptr);

                if(!entries_it_host.empty()){
                    entries_it_host.back().next = 
                        reinterpret_cast<typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>(
                            new_entry_it_ptr
                        );
                    new_entry_it_host.prev = 
                        reinterpret_cast<typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>(
                            entries_it_ptr.back()
                        );
                }
                entries_ptr.push_back(new_entry_ptr);
                entries_host.push_back(new_entry_host);
                entries_it_ptr.push_back(new_entry_it_ptr);
                entries_it_host.push_back(new_entry_it_host);
            }

            // Fill up host side elements
            typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::FirstEntry elements_first_host = {};
            typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::LastEntry elements_last_host = {};
            if(!entries_it_host.empty()){
                elements_first_host.next = 
                    reinterpret_cast<typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>(
                        entries_it_ptr.front()
                    );
                elements_last_host.prev =
                    reinterpret_cast<typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>(
                        entries_it_ptr.back()
                    );
            }
            CDoubleLinkedList<typename CAllocatorPool<T>::Entry*> elements_host = {};
            elements_host.initialised = false;
            elements_host.size = size;
            elements_host.first = 
                reinterpret_cast<typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::FirstEntry*>(
                    elements_first_ptr);
            elements_host.last = 
                reinterpret_cast<typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::LastEntry*>(
                    elements_last_ptr);            

            
            // Copy memory to GPU
            std::vector<CUdeviceptr> srcs = {};
            std::vector<CUdeviceptr> dsts = {};
            std::vector<size_t> sizes = {};

            for(uint32_t i=0; i<size; i++){
                CUdeviceptr it_ptr = entries_it_ptr[i];
                srcs.push_back((CUdeviceptr)&entries_it_host[i]);
                dsts.push_back(it_ptr);
                sizes.push_back(sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator));

                CUdeviceptr ptr = entries_ptr[i];
                srcs.push_back((CUdeviceptr)&entries_host[i]);
                dsts.push_back(ptr);
                sizes.push_back(sizeof(typename CAllocatorPool<T>::Entry));

                pointers.push_back(it_ptr);
                pointers.push_back(ptr);
            }

            srcs.push_back((CUdeviceptr)&elements_last_host);
            dsts.push_back(elements_last_ptr);
            sizes.push_back(sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::LastEntry));
            pointers.push_back(elements_last_ptr);

            srcs.push_back((CUdeviceptr)&elements_first_host);
            dsts.push_back(elements_first_ptr);
            sizes.push_back(sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::FirstEntry));
            pointers.push_back(elements_first_ptr);

            srcs.push_back((CUdeviceptr)&elements_host);
            dsts.push_back(elements_ptr);
            sizes.push_back(sizeof(CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>));
            pointers.push_back(elements_ptr);

            CUmemcpyAttributes attributes = CUmemcpyAttributes();
            attributes.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
            attributes.srcLocHint.type = CU_MEM_LOCATION_TYPE_HOST;
            attributes.dstLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
            size_t attributes_idxs = 0;
            size_t nb_attributes = 1;

            CURuntime::assertCudaSuccess(
                cuMemcpyBatchAsync(
                    (CUdeviceptr*)dsts.data(), (CUdeviceptr*)srcs.data(), (size_t*)sizes.data(), 
                    srcs.size(), &attributes, &attributes_idxs, nb_attributes, *stream
                )
            );

            return reinterpret_cast<CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>*>(elements_ptr);
        }

        template<typename T>
        static CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>* allocAllocatorElementsMap(uint32_t size, CUstream* stream){
            uint64_t real_size = 0;

            // Allocate main elements
            CUdeviceptr elements_map_ptr = 0;
            real_size = sizeof(CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_map_ptr, real_size));
            CUdeviceptr elements_ptr = 0;
            real_size = size * sizeof(CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_ptr, real_size));

            // Fill up host side elements
            CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>
                elements_map_host = {};
            elements_map_host.capacity = size;
            elements_map_host.size = 0;
            elements_map_host.initialised = false;

            elements_map_host.elements = reinterpret_cast<
                CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>*
            >(elements_ptr);

            std::vector<CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>>
                elements_host(size);


            // Copy memory to GPU
            CURuntime::assertCudaSuccess(cuMemcpyHtoD(
                elements_ptr, elements_host.data(), 
                size * sizeof(CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>)
            ));
            pointers.push_back(elements_ptr);

            CURuntime::assertCudaSuccess(cuMemcpyHtoD(
                elements_map_ptr, &elements_map_host, 
                sizeof(CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>)
            ));
            pointers.push_back(elements_map_ptr);


            return reinterpret_cast<
                CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>*  
            >(elements_map_ptr);
        }

        template<typename T>
        static CAllocatorPool<T>* allocAllocator(uint32_t size, AllocatorId type, CUstream* stream){
            uint64_t real_size = 0;
            
            // Allocate the allocator
            CUdeviceptr allocator_ptr = 0;
            real_size = sizeof(CAllocatorPool<T>);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&allocator_ptr, real_size));
            CAllocatorPool<T> tmp = CAllocatorPool<T>(size, type);

            // Allocate the list of elements
            tmp.elements = allocAllocatorElements<T>(size, stream);

            // Allocate the map of iterators
            tmp.elements_map = allocAllocatorElementsMap<T>(size, stream);

            // Allocate the memory pool
            CUdeviceptr allocation_pool_ptr = 0;
            uint64_t alignment = alignof(T);
            uint64_t aligned_size = sizeof(T) + ((alignment - (sizeof(T) % alignment)) % alignment);
            real_size = size * aligned_size;
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&allocation_pool_ptr, real_size));
            tmp.allocated_memory = reinterpret_cast<T*>(allocation_pool_ptr);

            // Allocate the deallocation array
            CUdeviceptr deallocation_pool_ptr = 0;
            real_size = size * sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*);
            totalAllocatedMemory += real_size;
            CURuntime::assertCudaSuccess(cuMemAlloc(&deallocation_pool_ptr, real_size));
            tmp.deallocated_memory = reinterpret_cast<typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator**>(deallocation_pool_ptr);

            CURuntime::assertCudaSuccess(cuMemcpyHtoD(allocator_ptr, &tmp, sizeof(CAllocatorPool<T>)));
            
            pointers.push_back(deallocation_pool_ptr);
            pointers.push_back(allocation_pool_ptr);
            pointers.push_back(allocator_ptr);
            return reinterpret_cast<CAllocatorPool<T>*>(allocator_ptr);
        };
};