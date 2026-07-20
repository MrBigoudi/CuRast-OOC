#pragma once

#include "globals.h"
#include "../kernels/ooc/GpuVersionInterface.h"
#include "../kernels/ooc/GpuVersionAllocator.h"

struct GpuVersion {
	static inline CudaModularProgram* prog = nullptr;

    /// Initialises everything needed on device memory
    static void init(CuRast* editor, CUcontext* context);
    static void destroy(CuRast* editor, CUcontext* context);

    private:
        static inline std::vector<CUdeviceptr> pointers = {};
        static void initBuffers(CuRast* editor, CUcontext* context);
        static void initAllocators(CuRast* editor, CUcontext* context);

        template<typename T>
        static T* alloc(uint32_t size){
            CUdeviceptr new_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(&new_ptr, size * sizeof(T)));
            pointers.push_back(new_ptr);
            return reinterpret_cast<T*>(new_ptr);
        }

        template<typename T>
        static CAllocatorPool<T>* allocAllocator(uint32_t size, AllocatorId type){
            // TODO: check if copies to GPU are correct and that pointers are correct

            // Allocate the allocator
            CUdeviceptr allocator_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(&allocator_ptr, sizeof(CAllocatorPool<T>)));
            CAllocatorPool<T> tmp = CAllocatorPool<T>(size, type);

            // Allocate all the lists of elements
            CUdeviceptr elements_first_ptr = 0;
            CUdeviceptr elements_last_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_first_ptr, sizeof(void*)));
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_last_ptr, sizeof(void*)));
            
            tmp.elements.first = reinterpret_cast<
                CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::FirstEntry*
            >(elements_first_ptr);
            tmp.elements.last = reinterpret_cast<
                CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::LastEntry*
            >(elements_last_ptr);


            CUdeviceptr map_list_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(
                &map_list_ptr, size * sizeof(
                    CDoubleLinkedList<
                        typename CHashMap<
                            T*, 
                            typename CDoubleLinkedList<
                                typename CAllocatorPool<T>::Entry*
                            >::Iterator*>::Entry>
                )
            ));
            tmp.elements_map.elements = reinterpret_cast<
                CDoubleLinkedList<typename CHashMap<
                    T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*
                >::Iterator*>::Entry>*>(map_list_ptr);
            
            // for(uint32_t i=0; i<size; i++){
            //     CUdeviceptr tmp_first_ptr = 0;
            //     CUdeviceptr tmp_last_ptr = 0;
            //     CURuntime::assertCudaSuccess(cuMemAlloc(&tmp_first_ptr, sizeof(void*)));
            //     CURuntime::assertCudaSuccess(cuMemAlloc(&tmp_last_ptr, sizeof(void*)));
                
            //     tmp.elements_map.elements[i].first = reinterpret_cast<
            //         CDoubleLinkedList<typename CHashMap<
            //             T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*
            //         >::Entry>::FirstEntry*>(tmp_first_ptr);
            //     tmp.elements_map.elements[i].last = reinterpret_cast<
            //         CDoubleLinkedList<typename CHashMap<
            //             T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*
            //         >::Entry>::LastEntry*>(tmp_last_ptr);

            //     pointers.push_back(tmp_first_ptr);
            //     pointers.push_back(tmp_last_ptr);
            // }

            // Allocate the pool
            CUdeviceptr pool_ptr = 0;
            uint32_t alignment = alignof(T);
            uint32_t aligned_size = sizeof(T) + ((alignment - (sizeof(T) % alignment)) % alignment);
            CURuntime::assertCudaSuccess(cuMemAlloc(&pool_ptr, size * aligned_size));
            tmp.allocated_memory = reinterpret_cast<T*>(pool_ptr);

            CURuntime::assertCudaSuccess(cuMemcpyHtoD(allocator_ptr, &tmp, sizeof(CAllocatorPool<T>)));
            
            pointers.push_back(map_list_ptr);
            pointers.push_back(elements_first_ptr);
            pointers.push_back(elements_last_ptr);
            pointers.push_back(allocator_ptr);
            pointers.push_back(pool_ptr);
            return reinterpret_cast<CAllocatorPool<T>*>(allocator_ptr);
        };
};