#pragma once

#include "CuRast.h"

/// Initialized empty batches of points from las/laz files
/// Updates the global `batchesToLoad` variable
void initLoadPointBatches(string file);

/// Load points from the global `batchesToLoad` variable
void loadPointsInBatches();

/// Send points to CUDA memory
void loadBatchesOnGPU(CuRast* editor);

void loadPointcloud(string file, CuRast* editor);