#pragma once

#include "globals.h"
#include "../kernels/ooc/GpuVersionInterface.h"
#include "../kernels/ooc/GpuVersionAllocator.h"

struct GpuVersion {
	static inline CudaModularProgram* prog = nullptr;
    static inline CGlobalVariables hostStaging = {};
    static inline CUdeviceptr deviceStaging = 0;
    static inline CUstream stream;

    static inline std::unordered_set<CIdAABB> storedNodes = {}; 

    static inline std::mutex renderSubmissionMutex;
	static inline void* exchangedPointsPointers = nullptr;
	static inline void* exchangedVoxelsPointers = nullptr;
	static inline void* batchesToAddPointsPointers = nullptr;
    static inline void* nbExchangedNodes = nullptr;
    static inline uint32_t curNbNodes = 0;

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
            CURuntime::assertCudaSuccess(cuMemAlloc(&new_ptr, size * sizeof(T)));
            pointers.push_back(new_ptr);
            return reinterpret_cast<T*>(new_ptr);
        }

        template<typename T>
        static CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>* allocAllocatorElements(uint32_t size, CUstream* stream){
            // Allocate main elements
            CUdeviceptr elements_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_ptr, 
                sizeof(CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>)
            ));
            CUdeviceptr elements_first_ptr = 0;
            CUdeviceptr elements_last_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_first_ptr, 
                sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::FirstEntry)
            ));
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_last_ptr, 
                sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::LastEntry)
            ));

            // Allocate list entries
            std::vector<CUdeviceptr> entries_ptr = {};
            std::vector<CUdeviceptr> entries_it_ptr = {};
            std::vector<typename CAllocatorPool<T>::Entry> entries_host = {};
            std::vector<typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator> entries_it_host = {};
            for(uint32_t i=0; i<size; i++){
                CUdeviceptr new_entry_ptr = 0;
                CUdeviceptr new_entry_it_ptr = 0;
                CURuntime::assertCudaSuccess(cuMemAlloc(&new_entry_ptr, 
                    sizeof(typename CAllocatorPool<T>::Entry)
                ));
                CURuntime::assertCudaSuccess(cuMemAlloc(&new_entry_it_ptr, 
                    sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator)
                ));
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
            // Allocate main elements
            CUdeviceptr elements_map_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_map_ptr, 
                sizeof(CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>)
            ));
            CUdeviceptr elements_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_ptr, 
                size * sizeof(CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>)
            ));

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
            // TODO: check if copies to GPU are correct and that pointers are correct
            
            // Allocate the allocator
            CUdeviceptr allocator_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(&allocator_ptr, sizeof(CAllocatorPool<T>)));
            CAllocatorPool<T> tmp = CAllocatorPool<T>(size, type);

            // Allocate the list of elements
            tmp.elements = allocAllocatorElements<T>(size, stream);

            // Allocate the map of iterators
            tmp.elements_map = allocAllocatorElementsMap<T>(size, stream);

            // Allocate the memory pool
            CUdeviceptr allocation_pool_ptr = 0;
            uint64_t alignment = alignof(T);
            uint64_t aligned_size = sizeof(T) + ((alignment - (sizeof(T) % alignment)) % alignment);
            CURuntime::assertCudaSuccess(cuMemAlloc(&allocation_pool_ptr, size * aligned_size));
            tmp.allocated_memory = reinterpret_cast<T*>(allocation_pool_ptr);

            // Allocate the deallocation array
            CUdeviceptr deallocation_pool_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(
                &deallocation_pool_ptr, 
                size * sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*)
            ));
            tmp.deallocated_memory = reinterpret_cast<typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator**>(deallocation_pool_ptr);

            CURuntime::assertCudaSuccess(cuMemcpyHtoD(allocator_ptr, &tmp, sizeof(CAllocatorPool<T>)));
            
            pointers.push_back(deallocation_pool_ptr);
            pointers.push_back(allocation_pool_ptr);
            pointers.push_back(allocator_ptr);
            return reinterpret_cast<CAllocatorPool<T>*>(allocator_ptr);
        };
};