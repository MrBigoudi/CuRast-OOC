#pragma once

#include "./GpuVersionInterface.h"
extern __device__ CGlobalVariables globalVariables;

struct CMemoryAllocator;
extern __device__ CMemoryAllocator globalAllocator;