#pragma once

#include "globals.h"
#include "../kernels/ooc/GpuVersionInterface.h"

struct GpuVersion {
	static inline CudaModularProgram* prog = nullptr;


    /// Initialises everything needed on device memory
    static void init(CuRast* editor, CUcontext* context);
};