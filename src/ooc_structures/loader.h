#pragma once

#include "CuRast.h"
#include "globals.h"

/// Initialized empty batches of points from las/laz files
/// Updates the global `batchesToLoad` variable
void initLoadPointBatches(string file,
    std::deque<std::shared_ptr<PointBatch>>& batches_queue = GlobalVariables::batchesQueue,
    std::deque<std::mutex>& batches_queue_mutexes = GlobalVariables::batchesQueueMutexes
);

/// Load points from the global `batchesToLoad` variable
void loadPointsInBatches(
    std::deque<std::shared_ptr<PointBatch>>& batches_queue = GlobalVariables::batchesQueue,
    std::deque<std::mutex>& batches_queue_mutexes = GlobalVariables::batchesQueueMutexes
);

/// Asynchronously load the point clouds into queues of batches
void loadPointcloudRoutine(
    std::deque<std::shared_ptr<PointBatch>>& batches_queue = GlobalVariables::batchesQueue,
    std::deque<std::mutex>& batches_queue_mutexes = GlobalVariables::batchesQueueMutexes
);

/// TODO: temporary function
/// Send points to CUDA memory
/// Used to get baseline timings but consumes a lot of memory
void loadBatchesOnGPU(CuRast* editor, CUcontext* ctx = nullptr,
    std::deque<std::shared_ptr<PointBatch>>& batches_queue = GlobalVariables::batchesQueue,
    std::deque<std::mutex>& batches_queue_mutexes = GlobalVariables::batchesQueueMutexes
);

/// Clear the unused batches
void clearUnusedBatches(
    std::deque<std::shared_ptr<PointBatch>>& batches_queue = GlobalVariables::batchesQueue,
    std::deque<std::mutex>& batches_queue_mutexes = GlobalVariables::batchesQueueMutexes
);
/// Asynchronously clear the unused batches
void clearUnusedBatchesRoutine(
    std::deque<std::shared_ptr<PointBatch>>& batches_queue = GlobalVariables::batchesQueue,
    std::deque<std::mutex>& batches_queue_mutexes = GlobalVariables::batchesQueueMutexes
);



struct LoaderGpuVersion {
    static inline std::deque<std::shared_ptr<PointBatch>> batchesQueue = {};
	static inline std::deque<std::mutex> batchesQueueMutexes = {};

    static inline std::mutex laszipReaderMtx;

    static inline std::vector<uint32_t> batchesOnGpu = {};
    static inline void* batchesOnGpuStatus = nullptr;

    static inline CUstream stream;
    static inline CUevent eventLoadComplete;

    static inline std::vector<CUmemcpyAttributes> loadingAttributes = {
        CUmemcpyAttributes {
            .srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
            .srcLocHint = {CU_MEM_LOCATION_TYPE_HOST},
            .dstLocHint = {CU_MEM_LOCATION_TYPE_DEVICE},
            .flags = CU_MEMCPY_FLAG_DEFAULT
        }
    };
    static inline std::vector<uint64_t> loadingAttributesIndices = {0};


    static void createNewBatches(string file);
    static bool run(CuRast* editor, CUcontext* context);

    static void init();
    static void destroy();
    static void fetchFromDevice();
    static bool sendToDevice();

    static void loadingRoutine();
};