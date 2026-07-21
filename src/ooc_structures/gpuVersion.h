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
        static CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>* allocAllocatorElements(uint32_t size){
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

                pointers.push_back(it_ptr);
                pointers.push_back(ptr);
            }

            CURuntime::assertCudaSuccess(cuMemcpyHtoD(
                elements_last_ptr, &elements_last_host, 
                sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::LastEntry)
            ));
            pointers.push_back(elements_last_ptr);

            CURuntime::assertCudaSuccess(cuMemcpyHtoD(
                elements_first_ptr, &elements_first_host, 
                sizeof(typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::FirstEntry)
            ));
            pointers.push_back(elements_first_ptr);

            CURuntime::assertCudaSuccess(cuMemcpyHtoD(
                elements_ptr, &elements_host, 
                sizeof(CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>)
            ));
            pointers.push_back(elements_ptr);

            return reinterpret_cast<CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>*>(elements_ptr);
        }

        template<typename T>
        static CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>* allocAllocatorElementsMap(uint32_t size){
            // Allocate main elements
            CUdeviceptr elements_map_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_map_ptr, 
                sizeof(CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>)
            ));
            CUdeviceptr elements_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(&elements_ptr, 
                size * sizeof(CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>)
            ));

            // Allocate and initialise first and last for each entry
            std::vector<CUdeviceptr> firsts_ptr = {};
            std::vector<CUdeviceptr> lasts_ptr = {};
            for(uint32_t i=0; i<size; i++){
                CUdeviceptr new_first_ptr = 0;
                CURuntime::assertCudaSuccess(cuMemAlloc(&new_first_ptr, 
                    sizeof(typename CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>::FirstEntry)
                ));
                CUdeviceptr new_last_ptr = 0;
                CURuntime::assertCudaSuccess(cuMemAlloc(&new_last_ptr, 
                    sizeof(typename CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>::LastEntry)
                ));

                typename CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>::FirstEntry
                    new_first_host = {};
                typename CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>::LastEntry
                    new_last_host = {};

                CURuntime::assertCudaSuccess(cuMemcpyHtoD(
                    new_first_ptr, &new_first_host,
                    sizeof(typename CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>::FirstEntry)
                ));
                CURuntime::assertCudaSuccess(cuMemcpyHtoD(
                    new_last_ptr, &new_last_host,
                    sizeof(typename CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>::LastEntry)
                ));

                firsts_ptr.push_back(new_first_ptr);
                lasts_ptr.push_back(new_last_ptr);
                pointers.push_back(new_first_ptr);
                pointers.push_back(new_last_ptr);
            }

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
            for(uint32_t i=0; i<size; i++){
                elements_host[i].first = reinterpret_cast<
                    typename CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>::FirstEntry*
                >(firsts_ptr[i]);

                elements_host[i].last = reinterpret_cast<
                    typename CDoubleLinkedList<typename CHashMap<T*, typename CDoubleLinkedList<typename CAllocatorPool<T>::Entry*>::Iterator*>::Entry>::LastEntry*
                >(lasts_ptr[i]);
            }


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
        static CAllocatorPool<T>* allocAllocator(uint32_t size, AllocatorId type){
            // TODO: check if copies to GPU are correct and that pointers are correct

            // Allocate the allocator
            CUdeviceptr allocator_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(&allocator_ptr, sizeof(CAllocatorPool<T>)));
            CAllocatorPool<T> tmp = CAllocatorPool<T>(size, type);

            // Allocate the list of elements
            tmp.elements = allocAllocatorElements<T>(size);

            // Allocate the map of iterators
            tmp.elements_map = allocAllocatorElementsMap<T>(size);

            // Allocate the memory pool
            CUdeviceptr pool_ptr = 0;
            uint32_t alignment = alignof(T);
            uint32_t aligned_size = sizeof(T) + ((alignment - (sizeof(T) % alignment)) % alignment);
            CURuntime::assertCudaSuccess(cuMemAlloc(&pool_ptr, size * aligned_size));
            tmp.allocated_memory = reinterpret_cast<T*>(pool_ptr);

            CURuntime::assertCudaSuccess(cuMemcpyHtoD(allocator_ptr, &tmp, sizeof(CAllocatorPool<T>)));
            
            pointers.push_back(pool_ptr);
            pointers.push_back(allocator_ptr);
            return reinterpret_cast<CAllocatorPool<T>*>(allocator_ptr);
        };
};