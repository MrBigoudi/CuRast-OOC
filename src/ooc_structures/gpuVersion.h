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
            // Allocate the allocator
            CUdeviceptr allocator_ptr = 0;
            CURuntime::assertCudaSuccess(cuMemAlloc(&allocator_ptr, sizeof(CAllocatorPool<T>)));
            CAllocatorPool<T> tmp = CAllocatorPool<T>(size, type);

            // Allocate the pool
            CUdeviceptr pool_ptr = 0;
            uint32_t alignment = alignof(T);
            uint32_t aligned_size = sizeof(T) + ((alignment - (sizeof(T) % alignment)) % alignment);
            CURuntime::assertCudaSuccess(cuMemAlloc(&pool_ptr, size * aligned_size));
            tmp.allocated_memory = reinterpret_cast<T*>(pool_ptr);

            CURuntime::assertCudaSuccess(cuMemcpyHtoD(allocator_ptr, &tmp, sizeof(CAllocatorPool<T>)));
            
            pointers.push_back(allocator_ptr);
            pointers.push_back(pool_ptr);
            return reinterpret_cast<CAllocatorPool<T>*>(allocator_ptr);
        };
};