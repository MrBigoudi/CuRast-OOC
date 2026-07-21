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

            // Allocate the list of elements
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

            

            for(uint32_t i=0; i<entries_it_host.size(); i++){
                CUdeviceptr it_ptr = entries_it_ptr[i];
                typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator& it_host = entries_it_host[i];
                CURuntime::assertCudaSuccess(cuMemcpyHtoD(
                    it_ptr, &it_host, 
                    sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator)
                ));

                CUdeviceptr ptr = entries_ptr[i];
                typename CAllocatorPool<T>::Entry& host = entries_host[i];
                CURuntime::assertCudaSuccess(cuMemcpyHtoD(
                    ptr, &host, 
                    sizeof(typename CAllocatorPool<T>::Entry*)
                ));
            }

            CURuntime::assertCudaSuccess(cuMemcpyHtoD(
                elements_last_ptr, &elements_last_host, 
                sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::LastEntry)
            ));

            CURuntime::assertCudaSuccess(cuMemcpyHtoD(
                elements_first_ptr, &elements_first_host, 
                sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::FirstEntry)
            ));

            CURuntime::assertCudaSuccess(cuMemcpyHtoD(
                elements_ptr, &elements_host, 
                sizeof(CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>)
            ));

            tmp.elements = reinterpret_cast<CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>*>(elements_ptr);




            // Allocate the map of iterators
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

            CUdeviceptr next_entry = 0;

            for(uint32_t i=0; i<size; i++){

            }
            

            // Allocate the memory pool
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