#pragma once

#include "CuRast.h"

/// Initialized empty batches of points from las/laz files
/// Updates the global `batchesToLoad` variable
void initLoadPointBatches(string file);

/// Load points from the global `batchesToLoad` variable
void loadPointsInBatches();

/// Asynchronously load the point clouds into queues of batches
void loadPointcloudRoutine();

/// TODO: temporary function
/// Send points to CUDA memory
/// Used to get baseline timings but consumes a lot of memory
void loadBatchesOnGPU(CuRast* editor);