#include "gpuVersion.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <thread>
#include <unordered_map>

#include "loader.h"
#include "outOfCore.h"

void HostStorageNode::init(){
    uint32_t node_count = OocSimLodSettings::LRU_CPU_CACHE_SIZE // Cache size
        + OocSimLodSettings::MAX_NB_NODES_TO_EXCHANGE    // Current update loaded node points
        + OocSimLodSettings::MAX_NB_NODES_TO_EXCHANGE    // Current update stored node points
    ;
    uint32_t data_count = OocSimLodSettings::MAX_POINTS_PER_LEAF;

    points_allocator_update.init(node_count, data_count);
    voxels_allocator_update.init(node_count, data_count);
    indices_allocator_update.init(node_count, data_count);

    // The visibility nodes are no longer kept in pinned memory: they are streamed from disk
    // into regular RAM and only their rendered points / voxels are copied into the pinned
    // staging buffers `GpuVersion::visStaging*' (see `GpuVersion::visibilityUpdate')
    // -> the `*_allocator_visibility' pools stay empty (no Visibility HostStorageNode is created)
}

void HostStorageNode::destroy(){
    points_allocator_update.destroy();
    voxels_allocator_update.destroy();
    indices_allocator_update.destroy();

    points_allocator_visibility.destroy();
    voxels_allocator_visibility.destroy();
    indices_allocator_visibility.destroy();
}

void HostStorageNode::deallocate(){
    switch(owner) {
        case Update:
            points_allocator_update.deallocate(points);
            voxels_allocator_update.deallocate(voxels);
            indices_allocator_update.deallocate(occupancy_indices);
            break;
        case Visibility:
            points_allocator_visibility.deallocate(points);
            voxels_allocator_visibility.deallocate(voxels);
            indices_allocator_visibility.deallocate(occupancy_indices);
            break;
    }

    points = nullptr;
    voxels = nullptr;
    occupancy_indices = nullptr;
}


HostStorageNode::HostStorageNode(const HostStorageNodeOwner& owner): owner(owner){
    switch(owner){
        case Update:
            points = points_allocator_update.allocate(owner);
            voxels = voxels_allocator_update.allocate(owner);
            occupancy_indices = indices_allocator_update.allocate(owner);
            break;
        case Visibility:
            points = points_allocator_visibility.allocate(owner);
            voxels = voxels_allocator_visibility.allocate(owner);
            occupancy_indices = indices_allocator_visibility.allocate(owner);
            break;
    }
}



#ifndef COPY_FROM_GPU
#define COPY_FROM_GPU(member, value, type)                                     \
    {                                                                          \
        const uint64_t pad =                                                   \
            reinterpret_cast<uintptr_t>(&(GpuVersion::hostStaging.member)) -   \
            reinterpret_cast<uintptr_t>(&GpuVersion::hostStaging);             \
        const CUdeviceptr src_device = GpuVersion::deviceStaging + pad;        \
        CURuntime::assertCudaSuccess(                                          \
            cuMemcpyDtoH(value, src_device, sizeof(type))                      \
        );                                                                     \
    }
#endif // COPY_FROM_GPU

#ifndef COPY_FROM_GPU_ASYNC
#define COPY_FROM_GPU_ASYNC(member, value, type)                               \
    {                                                                          \
        const uint64_t pad =                                                   \
            reinterpret_cast<uintptr_t>(&(GpuVersion::hostStaging.member)) -   \
            reinterpret_cast<uintptr_t>(&GpuVersion::hostStaging);             \
        const CUdeviceptr src_device = GpuVersion::deviceStaging + pad;        \
        CURuntime::assertCudaSuccess(                                          \
            cuMemcpyDtoHAsync(value, src_device, sizeof(type), 0)              \
        );                                                                     \
    }
#endif // COPY_FROM_GPU_ASYNC

#ifndef COPY_TO_GPU
#define COPY_TO_GPU(member, value, type)                                       \
    {                                                                          \
        const uint64_t pad =                                                   \
            reinterpret_cast<uintptr_t>(&(GpuVersion::hostStaging.member)) -   \
            reinterpret_cast<uintptr_t>(&GpuVersion::hostStaging);             \
        const CUdeviceptr dst_device = GpuVersion::deviceStaging + pad;        \
        CURuntime::assertCudaSuccess(                                          \
            cuMemcpyHtoD(dst_device, value, sizeof(type))                      \
        );                                                                     \
    }
#endif // COPY_TO_GPU

#ifndef COPY_TO_GPU_ASYNC
#define COPY_TO_GPU_ASYNC(member, value, type)                                 \
    {                                                                          \
        const uint64_t pad =                                                   \
            reinterpret_cast<uintptr_t>(&(GpuVersion::hostStaging.member)) -   \
            reinterpret_cast<uintptr_t>(&GpuVersion::hostStaging);             \
        const CUdeviceptr dst_device = GpuVersion::deviceStaging + pad;        \
        CURuntime::assertCudaSuccess(                                          \
            cuMemcpyHtoDAsync(dst_device, value, sizeof(type), 0)              \
        );                                                                     \
    }
#endif // COPY_TO_GPU_ASYNC

#ifndef COPY_TO_GPU_ASYNC_STREAM
#define COPY_TO_GPU_ASYNC_STREAM(member, value, type, stream)                  \
    {                                                                          \
        const uint64_t pad =                                                   \
            reinterpret_cast<uintptr_t>(&(GpuVersion::hostStaging.member)) -   \
            reinterpret_cast<uintptr_t>(&GpuVersion::hostStaging);             \
        const CUdeviceptr dst_device = GpuVersion::deviceStaging + pad;        \
        CURuntime::assertCudaSuccess(                                          \
            cuMemcpyHtoDAsync(dst_device, value, sizeof(type), stream)         \
        );                                                                     \
    }
#endif // COPY_TO_GPU_ASYNC_STREAM



void GpuVersion::initConstraints(CuRast* editor, CUcontext* context) {
    hostStaging.maxNbConcurrentNodes = OocSimLodSettings::MAX_NB_NODES;
    hostStaging.maxAllocatedChunks = OocSimLodSettings::NB_ALLOCABLE_CHUNKS;
    hostStaging.maxCountSplitIterations = OocSimLodSettings::MAX_NB_COUNT_SPLIT_ITERATION;

    hostStaging.maxNbNodesExchanged = OocSimLodSettings::MAX_NB_NODES_TO_EXCHANGE;
    hostStaging.maxNbPointsChunksPerExchangedNode =
        (OocSimLodSettings::MAX_POINTS_PER_LEAF + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1)
        / OocSimLodSettings::NB_POINTS_PER_CHUNK
    ;
    hostStaging.maxNbVoxelsChunksPerExchangedNode = OocSimLodSettings::MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE;

    hostStaging.maxNbRenderedPoints = OocSimLodSettings::MAX_NB_RENDERED_POINTS;
    hostStaging.maxNbRenderedVoxels = OocSimLodSettings::MAX_NB_RENDERED_VOXELS;

    hostStaging.maxNbBatches = OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE;
    hostStaging.maxBatchSize = OocSimLodSettings::MAX_POINTS_PER_BATCHES;

    hostStaging.updatesCacheSize = OocSimLodSettings::LRU_UPDATES_CACHE_SIZE;
    hostStaging.visibilityCacheSize = OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE;

    hostStaging.maxNbSpilledPoints = OocSimLodSettings::MAX_NB_SPILLING_POINTS;
    hostStaging.maxNbBacklogVoxels = OocSimLodSettings::MAX_NB_BACKLOG_VOXELS;
    hostStaging.maxPointsPerLeaf = OocSimLodSettings::MAX_POINTS_PER_LEAF;
}


void GpuVersion::initHostSide(CuRast* editor, CUcontext* context) {
    // Host side data
    exchangedPointsPointers = std::vector<CUdeviceptr>(hostStaging.maxNbNodesExchanged);
    exchangedVoxelsPointers = std::vector<CUdeviceptr>(hostStaging.maxNbNodesExchanged);
    exchangedGridsPointers = std::vector<CUdeviceptr>(hostStaging.maxNbNodesExchanged);
    batchesToAddPointsPointers = malloc(OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE * sizeof(CUdeviceptr));
    CURuntime::assertCudaSuccess(cuMemAllocHost(&nbExchangedNodes, sizeof(uint32_t)));
    *(uint32_t*)nbExchangedNodes = 0;

    storedNodesMtx = std::vector<std::mutex>(hostStaging.maxNbConcurrentNodes);

    CURuntime::assertCudaSuccess(cuEventCreate(&eventVisPreprocess, CU_EVENT_DISABLE_TIMING));
    CURuntime::assertCudaSuccess(cuEventCreate(&eventLoadingComplete, CU_EVENT_DISABLE_TIMING));
    CURuntime::assertCudaSuccess(cuEventCreate(&eventStoringComplete, CU_EVENT_DISABLE_TIMING));
    CURuntime::assertCudaSuccess(cuEventCreate(&eventVisibilityUpdateComplete, CU_EVENT_DISABLE_TIMING));

    hostCache = new CLRUCache(OocSimLodSettings::LRU_CPU_CACHE_SIZE);
    hostVisCache = new CLRUCache(OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE);
    parentsMap = std::vector<CIdAABB>(OocSimLodSettings::MAX_NB_NODES, CINVALID_ID);
    aabbsMap = std::vector<CAABB>(OocSimLodSettings::MAX_NB_NODES, CAABB());

    CURuntime::assertCudaSuccess(cuMemAllocHost(&isInitialised, sizeof(bool)));
    CURuntime::assertCudaSuccess(cuMemAllocHost(&isUpdating, sizeof(bool)));
    CURuntime::assertCudaSuccess(cuMemAllocHost(&isDoneLoading, sizeof(bool)));
    CURuntime::assertCudaSuccess(cuMemAllocHost(&isDoneStoring, sizeof(bool)));
    CURuntime::assertCudaSuccess(cuMemAllocHost(&isDoneIterating, sizeof(bool)));
    *(bool*)isInitialised = false;
    *(bool*)isUpdating = false;
    *(bool*)isDoneLoading = true;
    *(bool*)isDoneStoring = true;
    *(bool*)isDoneIterating = true;

    exchangedIds = allocHost<CIdAABB>(hostStaging.maxNbNodesExchanged);
    exchangedParentsIds = allocHost<CIdAABB>(hostStaging.maxNbNodesExchanged);
    exchangedAABBs = allocHost<CAABB>(hostStaging.maxNbNodesExchanged);
    exchangedChildrenIds = allocHost<uint32_t>(hostStaging.maxNbNodesExchanged);
    exchangedPointsCounters = allocHost<uint32_t>(hostStaging.maxNbNodesExchanged);
    exchangedVoxelsCounters = allocHost<uint32_t>(hostStaging.maxNbNodesExchanged);

    // visibilityCache = allocHost<CIdAABB>(hostStaging.visibilityCacheSize);
    voxelsNodesToSend = allocHost<CIdAABB>(hostStaging.maxNbRenderedVoxels);
    nbRenderedPoints = allocHost<uint32_t>(1);
    nbRenderedVoxels = allocHost<uint32_t>(1);
    nbRenderedNodes = allocHost<uint32_t>(1);

    // visibilityCache2 = allocHost<CIdAABB>(hostStaging.visibilityCacheSize);
    voxelsNodesToSend2 = allocHost<CIdAABB>(hostStaging.maxNbRenderedVoxels);
    nbRenderedPoints2 = allocHost<uint32_t>(1);
    nbRenderedVoxels2 = allocHost<uint32_t>(1);
    nbRenderedNodes2 = allocHost<uint32_t>(1);

    // The visible lists are not truncated anymore: they can hold every node (same size as on device)
    exchangedAABBIndicesVisPoints = allocHost<CIdAABB>(hostStaging.maxNbConcurrentNodes);
    exchangedAABBIndicesVisVoxels = allocHost<CIdAABB>(hostStaging.maxNbConcurrentNodes);
    exchangedPrioritiesVisPoints = allocHost<float>(hostStaging.maxNbConcurrentNodes);
    exchangedPrioritiesVisVoxels = allocHost<float>(hostStaging.maxNbConcurrentNodes);

    // Staging buffers for the rendered points / voxels (same capacity as the device buffers)
    visStagingPoints = static_cast<CPoint*>(allocHost<CPoint>(hostStaging.maxNbRenderedPoints));
    visStagingVoxels = static_cast<CPoint*>(allocHost<CPoint>(hostStaging.maxNbRenderedVoxels));
    nbNodesExchangedVisPoints = allocHost<uint32_t>(1);
    nbNodesExchangedVisVoxels = allocHost<uint32_t>(1);
    curNbVisNode = allocHost<uint32_t>(1);


    MAX_NB_VOXELS = hostStaging.maxNbVoxelsChunksPerExchangedNode * OocSimLodSettings::NB_POINTS_PER_CHUNK;
    HostStorageNode::init();
    GpuVersionUI::init();

    visibleNodes.resize(hostStaging.maxNbConcurrentNodes);
    visibleNodesOrdered.resize(hostStaging.maxNbConcurrentNodes);
}

void GpuVersion::initBuffers(CuRast* editor, CUcontext* context) {

    // Unbounded data
    hostStaging.relationshipMap = alloc<CGlobalVariables::Relationship>(hostStaging.maxNbConcurrentNodes);
    hostStaging.packedNodes = alloc<COctreeNode*>(hostStaging.maxNbConcurrentNodes);
    hostStaging.nodesFlags = alloc<uint32_t>(hostStaging.maxNbConcurrentNodes);


    // Exchangeable data
    hostStaging.exchangedAABBIndices = alloc<CIdAABB>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedAABBParentsIndices = alloc<CIdAABB>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedAABBs = alloc<CAABB>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedChildrenIds = alloc<uint32_t>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedPointsCounters = alloc<uint32_t>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedVoxelsCounters = alloc<uint32_t>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedPoints = alloc<CPoint*>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedVoxels = alloc<CPoint*>(hostStaging.maxNbNodesExchanged);
    hostStaging.exchangedGrids = alloc<uint64_t*>(hostStaging.maxNbNodesExchanged);
    // Snapshot of the last eviction pass (a few KB), see `kernel_draw_evicted_nodes'
    hostStaging.evictedNodesIds = alloc<CIdAABB>(hostStaging.maxNbNodesExchanged);
    hostStaging.evictedNodesNbPoints = alloc<uint32_t>(hostStaging.maxNbNodesExchanged);
    hostStaging.evictedNodesNbVoxels = alloc<uint32_t>(hostStaging.maxNbNodesExchanged);
    static_assert(sizeof(CIdAABB) == sizeof(uint32_t), "evictedNodesIds is initialised with cuMemsetD32");
    CURuntime::assertCudaSuccess(cuMemsetD32((CUdeviceptr)hostStaging.evictedNodesIds, CINVALID_ID, hostStaging.maxNbNodesExchanged));
    CURuntime::assertCudaSuccess(cuMemsetD32((CUdeviceptr)hostStaging.evictedNodesNbPoints, 0, hostStaging.maxNbNodesExchanged));
    CURuntime::assertCudaSuccess(cuMemsetD32((CUdeviceptr)hostStaging.evictedNodesNbVoxels, 0, hostStaging.maxNbNodesExchanged));
    for(uint32_t i=0; i<hostStaging.maxNbNodesExchanged; i++){
        uint64_t real_size = 0;

        CUdeviceptr new_ptr = 0;
        real_size = OocSimLodSettings::NB_POINTS_PER_CHUNK * hostStaging.maxNbPointsChunksPerExchangedNode * sizeof(CPoint);
        totalAllocatedMemory += real_size;
        CURuntime::assertCudaSuccess(cuMemAlloc(&new_ptr, real_size));
        exchangedPointsPointers[i] = new_ptr;
        pointers.push_back(new_ptr);

        real_size = OocSimLodSettings::NB_POINTS_PER_CHUNK * hostStaging.maxNbVoxelsChunksPerExchangedNode * sizeof(CPoint);
        totalAllocatedMemory += real_size;
        CURuntime::assertCudaSuccess(cuMemAlloc(&new_ptr, real_size));
        exchangedVoxelsPointers[i] = new_ptr;
        pointers.push_back(new_ptr);

        real_size = OocSimLodSettings::NB_POINTS_PER_CHUNK * hostStaging.maxNbVoxelsChunksPerExchangedNode * sizeof(uint64_t);
        totalAllocatedMemory += real_size;
        CURuntime::assertCudaSuccess(cuMemAlloc(&new_ptr, real_size));
        exchangedGridsPointers[i] = new_ptr;
        pointers.push_back(new_ptr);

        CURuntime::assertCudaSuccess(cuMemcpyHtoD(
            (CUdeviceptr)hostStaging.exchangedPoints,
            exchangedPointsPointers.data(),
            hostStaging.maxNbNodesExchanged * sizeof(CUdeviceptr)
        ));
        CURuntime::assertCudaSuccess(cuMemcpyHtoD(
            (CUdeviceptr)hostStaging.exchangedVoxels,
            exchangedVoxelsPointers.data(),
            hostStaging.maxNbNodesExchanged * sizeof(CUdeviceptr)
        ));
        CURuntime::assertCudaSuccess(cuMemcpyHtoD(
            (CUdeviceptr)hostStaging.exchangedGrids,
            exchangedGridsPointers.data(),
            hostStaging.maxNbNodesExchanged * sizeof(CUdeviceptr)
        ));
    }

    hostStaging.gridsToInit = alloc<COctreeNode*>(hostStaging.maxNbConcurrentNodes);
    hostStaging.gridsToInitExchangedIndex = alloc<uint32_t>(hostStaging.maxNbConcurrentNodes);

    hostStaging.batchesAddedMask = alloc<uint32_t>(hostStaging.maxNbBatches);
    hostStaging.batchesToAddCounts = alloc<uint32_t>(hostStaging.maxNbBatches);
    hostStaging.batchesToAddPoints = alloc<CPoint*>(hostStaging.maxNbBatches);
    uint64_t real_size = hostStaging.maxBatchSize * sizeof(CPoint);
    for(uint32_t i=0; i<hostStaging.maxNbBatches; i++){
        totalAllocatedMemory += real_size;
        CUdeviceptr new_ptr = 0;
        CURuntime::assertCudaSuccess(cuMemAlloc(&new_ptr, real_size));
        ((CUdeviceptr*)(batchesToAddPointsPointers))[i] = new_ptr;
        pointers.push_back(new_ptr);
    }
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(
        (CUdeviceptr)hostStaging.batchesToAddPoints,
        batchesToAddPointsPointers,
        hostStaging.maxNbBatches * sizeof(CUdeviceptr)
    ));


    // Lru caches
    hostStaging.updatesCache = alloc<CIdAABB>(hostStaging.updatesCacheSize * 2); // Times 2 for prefix scan
    hostStaging.exchangedAABBIndicesVisPoints = alloc<CIdAABB>(hostStaging.maxNbConcurrentNodes);
    hostStaging.exchangedAABBIndicesVisPointsTmp = alloc<CIdAABB>(hostStaging.maxNbConcurrentNodes);
    hostStaging.exchangedAABBIndicesVisPointsScreenSpaceSize = alloc<float>(hostStaging.maxNbConcurrentNodes);
    hostStaging.exchangedAABBIndicesVisVoxels = alloc<CIdAABB>(hostStaging.maxNbConcurrentNodes);
    hostStaging.exchangedAABBIndicesVisVoxelsTmp = alloc<CIdAABB>(hostStaging.maxNbConcurrentNodes);
    hostStaging.exchangedAABBIndicesVisVoxelsScreenSpaceSize = alloc<float>(hostStaging.maxNbConcurrentNodes);

    hostStaging.renderedPoints = alloc<CPoint>(hostStaging.maxNbRenderedPoints);
    hostStaging.renderedVoxels = alloc<CPoint>(hostStaging.maxNbRenderedVoxels);
    // hostStaging.renderedVoxelsNodes = alloc<CIdAABB>(hostStaging.maxNbRenderedVoxels);
    hostStaging.renderedPoints2 = alloc<CPoint>(hostStaging.maxNbRenderedPoints);
    hostStaging.renderedVoxels2 = alloc<CPoint>(hostStaging.maxNbRenderedVoxels);
    // hostStaging.renderedVoxelsNodes2 = alloc<CIdAABB>(hostStaging.maxNbRenderedVoxels);


    // Temporary buffers
    hostStaging.spilledPoints = alloc<CPoint>(hostStaging.maxNbSpilledPoints);
    hostStaging.spillingNodes = alloc<COctreeNode*>(hostStaging.maxNbSpilledPoints);
    hostStaging.spilledChunksCounter = alloc<uint32_t>(hostStaging.maxNbSpilledPoints);
    hostStaging.spillingChunks = alloc<CChunk*>(hostStaging.maxNbSpilledPoints);
    hostStaging.allocatedChunks = alloc<CChunk*>(hostStaging.maxAllocatedChunks);

    hostStaging.allocationPlans = alloc<CAllocationPlan>(hostStaging.maxNbConcurrentNodes);

    hostStaging.backlogVoxels = alloc<CPoint>(hostStaging.maxNbBacklogVoxels);
    hostStaging.backlogVoxelsNodes = alloc<COctreeNode*>(hostStaging.maxNbBacklogVoxels);

    hostStaging.memoizedBatchPointsNodes = alloc<COctreeNode*>(hostStaging.maxNbBatches * hostStaging.maxBatchSize);
    hostStaging.memoizedSpilledPointsNodes = alloc<COctreeNode*>(hostStaging.maxNbSpilledPoints);

    hostStaging.temporaryBufferSize = hostStaging.maxNbConcurrentNodes;
    hostStaging.temporaryIdBuffer = alloc<CIdAABB>(hostStaging.temporaryBufferSize);
    hostStaging.temporaryIdBuffer2 = alloc<CIdAABB>(hostStaging.temporaryBufferSize);
    hostStaging.temporaryNodeBuffer = alloc<COctreeNode*>(hostStaging.temporaryBufferSize);



    // Final allocation
    deviceStaging = prog->getGlobalsPointer("globalVariables");
    if (deviceStaging == 0) {
        throw std::runtime_error("globalVariables symbol not found");
    }
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(deviceStaging, &hostStaging, sizeof(CGlobalVariables)));
}


void GpuVersion::initAllocators(CuRast* editor, CUcontext* context, CUstream* stream) {
    CMemoryAllocator hostStaging = {};

    hostStaging.chunksAllocator = allocAllocator<CChunk>(
        OocSimLodSettings::NB_ALLOCABLE_CHUNKS, AllocatorId::ChunkAllocator
    );
    hostStaging.gridsAllocator = allocAllocator<COccupancyGrid>(
        OocSimLodSettings::NB_ALLOCABLE_GRIDS, AllocatorId::OccupancyGridAllocator
    );
    hostStaging.nodesAllocator = allocAllocator<COctreeNode>(
        OocSimLodSettings::NB_ALLOCABLE_NODES, AllocatorId::OctreeNodeAllocator
    );

    CUdeviceptr global_allocator_ptr = prog->getGlobalsPointer("globalAllocator");
    if (global_allocator_ptr == 0) {
        throw std::runtime_error("globalAllocator symbol not found");
    }
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(global_allocator_ptr, &hostStaging, sizeof(CMemoryAllocator)));
}






void GpuVersion::init(CuRast* editor, CUcontext* context) {
    prog = new CudaModularProgram({
        "./src/kernels/ooc/init.cu",
        "./src/kernels/ooc/bottomUp.cu",
        "./src/kernels/ooc/simlod.cu",
        "./src/kernels/ooc/caches.cu",
        "./src/kernels/ooc/render.cu"
    });

    CURuntime::assertCudaSuccess(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CURuntime::assertCudaSuccess(cuStreamCreate(&visStream, CU_STREAM_NON_BLOCKING));
    initConstraints(editor, context);
    initHostSide(editor, context);
    initBuffers(editor, context);
    initAllocators(editor, context, &stream);
    cudaDeviceSynchronize(); // Needed because of the batch copies in the init functions

    // size_t heap_size = 1024 * 1024 * 1024; // 1Gb for now
    // CURuntime::assertCudaSuccess(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, heap_size));

    // OptionalLaunchSettings launch_settings = {
    //     .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
    //     .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    // };

    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1)
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };

    // prog->launch("kernel_init_global_allocators", {}, launch_settings);
    prog->launch("kernel_init_global_buffers", {}, launch_settings);
    LoaderGpuVersion::init();
}


void GpuVersion::destroy(CuRast *editor, CUcontext *context){
    // The detached visibility workers write into the staging buffers freed below
    while(!isDoneDeserializingForVisibility){
        std::this_thread::yield();
    }

    if(hostCache){delete(hostCache);}

    cudaDeviceSynchronize();
    for(CUdeviceptr& ptr : pointers){
        if(ptr){
            CURuntime::assertCudaSuccess(cuMemFree(ptr));
        }
    }
    free(batchesToAddPointsPointers);
    CURuntime::assertCudaSuccess(cuMemFreeHost(nbExchangedNodes));
    CURuntime::assertCudaSuccess(cuMemFreeHost(isDoneLoading));
    CURuntime::assertCudaSuccess(cuMemFreeHost(isDoneStoring));
    CURuntime::assertCudaSuccess(cuMemFreeHost(isDoneIterating));

    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedIds));
    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedParentsIds));
    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedAABBs));
    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedChildrenIds));
    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedPointsCounters));
    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedVoxelsCounters));

    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedAABBIndicesVisPoints));
    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedAABBIndicesVisVoxels));
    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedPrioritiesVisPoints));
    CURuntime::assertCudaSuccess(cuMemFreeHost(exchangedPrioritiesVisVoxels));
    CURuntime::assertCudaSuccess(cuMemFreeHost(visStagingPoints));
    CURuntime::assertCudaSuccess(cuMemFreeHost(visStagingVoxels));

    CURuntime::assertCudaSuccess(cuMemFreeHost(voxelsNodesToSend));
    CURuntime::assertCudaSuccess(cuMemFreeHost(nbRenderedPoints));
    CURuntime::assertCudaSuccess(cuMemFreeHost(nbRenderedVoxels));
    CURuntime::assertCudaSuccess(cuMemFreeHost(nbRenderedNodes));

    CURuntime::assertCudaSuccess(cuMemFreeHost(voxelsNodesToSend2));
    CURuntime::assertCudaSuccess(cuMemFreeHost(nbRenderedPoints2));
    CURuntime::assertCudaSuccess(cuMemFreeHost(nbRenderedVoxels2));
    CURuntime::assertCudaSuccess(cuMemFreeHost(nbRenderedNodes2));


    CURuntime::assertCudaSuccess(cuEventDestroy(eventVisPreprocess));
    CURuntime::assertCudaSuccess(cuEventDestroy(eventLoadingComplete));
    CURuntime::assertCudaSuccess(cuEventDestroy(eventStoringComplete));
    CURuntime::assertCudaSuccess(cuEventDestroy(eventVisibilityUpdateComplete));

    LoaderGpuVersion::destroy();
    HostStorageNode::destroy();
    GpuVersionUI::destroy();

    cudaDeviceSynchronize();
    CURuntime::assertCudaSuccess(cuStreamDestroy(visStream));
    CURuntime::assertCudaSuccess(cuStreamDestroy(stream));
}













void GpuVersion::octreeUpdateInit(CuRast* editor, CUcontext* context){
    // OptionalLaunchSettings launch_settings = {
    //     .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
    //     .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    // };

    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1)
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };

    prog->launch("kernel_init_octree_part_1_aabb_measuring", {}, launch_settings);
    COPY_FROM_GPU(isInitialised, isInitialised, bool);
    if(!*(bool*)isInitialised){return;}

    launch_settings = {
        .gridsize = 1,
        .blocksize = 1
    };
    prog->launch("kernel_init_octree_part_2_refining", {}, launch_settings);
}














void GpuVersion::octreeUpdateBottomUp(CuRast* editor, CUcontext* context){
    // OptionalLaunchSettings launch_settings = {
    //     .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
    //     .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    // };

    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1)
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };
    OptionalLaunchSettings single_launch = {
        .gridsize  = 1,
        .blocksize = 1
    };

    prog->launch("kernel_bottom_up_update_part_0_flagging", {}, single_launch);
    COPY_FROM_GPU(isUpdating, isUpdating, bool);
    if(!*(bool*)isUpdating){return;}
    prog->launch("kernel_bottom_up_update_part_1_counting", {}, launch_settings);
    prog->launch("kernel_bottom_up_update_part_2_instancing", {}, single_launch);
}













void GpuVersion::octreeUpdateFillNewGrids(CuRast* editor, CUcontext* context){
    uint32_t block_size = min(
        OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM,
        OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
    );
    OptionalLaunchSettings launch_settings = {
        .gridsize = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM,
        .blocksize = block_size
    };
    prog->launch("kernel_fill_new_grids", {}, launch_settings);
}







void GpuVersion::octreeUpdateSimLODLoad(CuRast* editor, CUcontext* context){
    if(!hasStartedLoading){
        if(*(bool*)isDoneLoading){
            COPY_TO_GPU_ASYNC(nbNodesExchangedBeforeLoadComplete, &RESET, uint32_t);
        }
        *(bool*)isDoneLoading = true;
        COPY_TO_GPU_ASYNC(isDoneLoading, isDoneLoading, bool);
        COPY_TO_GPU_ASYNC(nbNodesExchanged, &RESET, uint32_t);

        // launch_settings = {
        //     .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
        //     .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        // };
        uint32_t block_size = 256;
        uint32_t grid_size =
            (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1)
            / block_size
        ;
        OptionalLaunchSettings launch_settings = {
            .gridsize  = grid_size,
            .blocksize = block_size
        };
        prog->launch("kernel_simlod_load_part_1_flagging", {}, launch_settings);
        COPY_FROM_GPU(isDoneLoading, isDoneLoading, bool);

        // Get the number of nodes to load
        COPY_FROM_GPU(nbNodesExchanged, nbExchangedNodes, uint32_t);
        nbNodesToLoad = min(*(uint32_t*)(nbExchangedNodes), hostStaging.maxNbNodesExchanged);
        if(nbNodesToLoad == 0){return;}
        // println("\n\nNb nodes to load: {}\n\n\n", nbNodesToLoad);

        // Get the ids of the nodes to load
        CURuntime::assertCudaSuccess(cuMemcpyDtoH(
            exchangedIds,
            (CUdeviceptr)hostStaging.exchangedAABBIndices,
            nbNodesToLoad * sizeof(CIdAABB)
        ));

        // Load from CPU cache
        CIdAABB* ids = static_cast<CIdAABB*>(exchangedIds);
        // Deserialise in parallel
        loadingToDeserialise = {};
        for(uint32_t i=0; i < nbNodesToLoad; i++){
            const CIdAABB& id = ids[i];
            hostCache->add(id);
            currentlyInUpdatesCache.insert(id);

            // TODO: to remove
            if(!storedNodes.contains(id)){println("node {} is not yet stored; why is it here ??", id);}

            if(!updateNodes.contains(id)){
                updateNodes[id] = std::make_shared<HostStorageNode>(HostStorageNodeOwner::Update);
                loadingToDeserialise.push_back(id);
            }
        }
        hasStartedLoading = true;

        if(!loadingToDeserialise.empty()){
            isDoneDeserializingForLoading = false;

            if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
                std::thread deserialize_thread([&]() {
                    std::for_each(loadingToDeserialise.begin(), loadingToDeserialise.end(),
                        [](const CIdAABB& id){
                            OctreeNodeSerializable::deserializeV2(updateNodes[id].get(), id, "From simlod load");
                        }
                    );
                    isDoneDeserializingForLoading = true;
                });
                deserialize_thread.detach();
                // Skip sending to device this frame; the next frame will pick it up
                return;
            } else {
                std::for_each(loadingToDeserialise.begin(), loadingToDeserialise.end(), [](const CIdAABB& id){
                    OctreeNodeSerializable::deserializeV2(updateNodes[id].get(), id, "From simlod load");
                });
                isDoneDeserializingForLoading = true;
            }
        }
    }
    hasStartedLoading = false;

    // Send the nodes back to the device
    std::vector<CUdeviceptr> srcs_host = {
        (CUdeviceptr)exchangedChildrenIds,
        (CUdeviceptr)exchangedPointsCounters,
        (CUdeviceptr)exchangedVoxelsCounters
    };
    std::vector<CUdeviceptr> dsts_device = {
        (CUdeviceptr)hostStaging.exchangedChildrenIds,
        (CUdeviceptr)hostStaging.exchangedPointsCounters,
        (CUdeviceptr)hostStaging.exchangedVoxelsCounters
    };
    std::vector<uint64_t> sizes = {
        nbNodesToLoad * sizeof(uint32_t),
        nbNodesToLoad * sizeof(uint32_t),
        nbNodesToLoad * sizeof(uint32_t)
    };
    CIdAABB* ids = static_cast<CIdAABB*>(exchangedIds);

    for(uint32_t i = 0; i<nbNodesToLoad; i++){
        HostStorageNode* cur_node = updateNodes[ids[i]].get();
        static_cast<uint32_t*>(exchangedChildrenIds)[i] = cur_node->node.children_ids;
        static_cast<uint32_t*>(exchangedPointsCounters)[i] = cur_node->node.points_counter;
        static_cast<uint32_t*>(exchangedVoxelsCounters)[i] = cur_node->node.voxels_counter;

        // Send grids info
        if(cur_node->node.voxels_counter > 0){
            srcs_host.push_back((CUdeviceptr)cur_node->occupancy_indices);
            dsts_device.push_back(exchangedGridsPointers[i]);
            uint32_t nb_voxels = min(cur_node->node.voxels_counter, MAX_NB_VOXELS);
            if(nb_voxels != cur_node->node.voxels_counter){
                println("WARN: Host side, simlodLoad too many voxels to send; some will be skipped\n");
            }
            sizes.push_back(min(cur_node->node.voxels_counter, MAX_NB_VOXELS) * sizeof(uint64_t));
        }
    }

    uint64_t nb_copies = sizes.size();
    CURuntime::assertCudaSuccess(cuMemcpyBatchAsync(
        dsts_device.data(), srcs_host.data(), sizes.data(), nb_copies,
        batchLoadingAttributes.data(),
        batchLoadingAttributesIndices.data(),
        batchLoadingAttributes.size(),
        stream
    ));
    CURuntime::assertCudaSuccess(cuEventRecord(eventLoadingComplete, stream));
    cudaStreamWaitEvent(0, eventLoadingComplete);

    OptionalLaunchSettings launch_settings = {
        .gridsize = OocSimLodSettings::MAX_NB_NODES_TO_EXCHANGE,
        // .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        .blocksize = 256
    };
    prog->launch("kernel_simlod_load_part_2_rebuilding_nodes", {}, launch_settings);
    octreeUpdateFillNewGrids(editor, context);
    prog->launch("kernel_simlod_load_part_3_rebuilding_children", {}, launch_settings);
}














void GpuVersion::octreeUpdateSimLODCountSplit(CuRast* editor, CUcontext* context){
    bool is_first_iteration = (*(bool*)isDoneIterating == true);
    COPY_TO_GPU(isFirstCountSplitIteration, isDoneIterating, bool);
    *(bool*)isDoneIterating = false;
    COPY_TO_GPU(isDoneIterating, isDoneIterating, bool);

    // OptionalLaunchSettings launch_settings = {
    //     .gridsize = 0, // Not used with launchCoopertative
    //     // .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    //     .blocksize = 256
    // };
    // prog->launchCooperative("kernel_simlod_count_split", {}, launch_settings);

    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1)
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };
    OptionalLaunchSettings single_launch = {
        .gridsize  = 1,
        .blocksize = 1
    };

    for(uint32_t i=0; i<OocSimLodSettings::MAX_NB_COUNT_SPLIT_ITERATION; i++){
        prog->launch("kernel_simlod_count_split_part_1_count", {&i, &is_first_iteration}, launch_settings);
        prog->launch("kernel_simlod_count_split_part_1_5_prefix_sum_1", {}, launch_settings);
        prog->launch("kernel_simlod_count_split_part_1_5_prefix_sum_2", {}, single_launch);
        prog->launch("kernel_simlod_count_split_part_2_split", {}, launch_settings);
        prog->launch("kernel_simlod_count_split_part_3_chunks_delete", {}, launch_settings);
        prog->launch("kernel_simlod_count_split_part_4_reset", {}, single_launch);
        COPY_FROM_GPU(isDoneIterating, isDoneIterating, bool);
        if(*(bool*)isDoneIterating){break;}
    }

}














void GpuVersion::octreeUpdateSimLODVoxelSampling(CuRast* editor, CUcontext* context){
    octreeUpdateFillNewGrids(editor, context);
    COPY_TO_GPU(nbGridsToInit, &RESET, uint32_t);

    if(!*(bool*)isDoneIterating){return;}

    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1)
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };
    prog->launch("kernel_simlod_voxel_sampling", {}, launch_settings);
}












void GpuVersion::octreeUpdateSimLODInsertion(CuRast* editor, CUcontext* context){
    // OptionalLaunchSettings launch_settings = {
    //     .gridsize  = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_GRID_SIZE_FOR_MAX_BLOCK_SIZE,
    //     .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    // };

    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1)
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };

    prog->launch("kernel_simlod_insertion_part_1_1_chunks_allocations_plan", {}, launch_settings);
    prog->launch("kernel_simlod_insertion_part_1_2_chunks_allocations_alloc", {}, launch_settings);
    prog->launch("kernel_simlod_insertion_part_1_3_chunks_allocations_link", {}, launch_settings);
    prog->launch("kernel_simlod_insertion_part_2_filling", {}, launch_settings);
}















void GpuVersion::octreeUpdateSimLOD(CuRast* editor, CUcontext* context){
    if(*(bool*)isDoneIterating){
        octreeUpdateSimLODLoad(editor, context);
        if(!isDoneDeserializingForLoading){return;}
    }

    if(*(bool*)isDoneLoading){
        octreeUpdateSimLODCountSplit(editor, context);
    }

    if(*(bool*)isDoneLoading){
        octreeUpdateSimLODVoxelSampling(editor, context);
    }

    if(*(bool*)isDoneLoading && *(bool*)isDoneIterating){
        octreeUpdateSimLODInsertion(editor, context);
    }
}















void GpuVersion::octreeUpdateCacheUpdate(CuRast* editor, CUcontext* context){
    if(!hasStartedStoring && !hasStartedSynchronisingEventInStoreNodes){
        COPY_TO_GPU(nbNodesExchanged, &RESET, uint32_t);
        uint32_t block_size = 256;
        uint32_t grid_size =
            (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1)
            / block_size
        ;
        OptionalLaunchSettings launch_settings = {
            .gridsize  = grid_size,
            .blocksize = block_size
        };
        OptionalLaunchSettings single_launch = {
            .gridsize  = 1,
            .blocksize = 1
        };
        if(*(bool*)isDoneStoring){
            prog->launch("kernel_update_updates_cache_part_1_counting", {}, launch_settings);
            prog->launch("kernel_update_updates_cache_part_2_sorting", {}, launch_settings);
            prog->launch("kernel_update_updates_cache_part_3_prefix_sum", {}, single_launch);
        }
        *(bool*)isDoneStoring = true;
        COPY_TO_GPU(isDoneStoring, isDoneStoring, bool);

        prog->launch("kernel_prepare_store_part_0_check_stored", {}, launch_settings);
        prog->launch("kernel_prepare_store_part_1_filling_buffers", {}, launch_settings);
        prog->launch("kernel_prepare_store_part_2_resetting_children", {}, launch_settings);

        prog->launch("kernel_prepare_store_part_3_pack_nodes", {}, single_launch);
        launch_settings = {
            .gridsize = 0,
            // .blocksize = OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            .blocksize = 256,
        };
        prog->launchCooperative("kernel_prepare_store_part_4_updating_levels", {}, launch_settings);
        COPY_FROM_GPU(isDoneStoring, isDoneStoring, bool);

        COPY_FROM_GPU(nbNodesExchanged, nbExchangedNodes, uint32_t);
        nbNodesToStore = min(*(uint32_t*)(nbExchangedNodes), hostStaging.maxNbNodesExchanged);
        if(nbNodesToStore == 0){return;}

        // New eviction generation: `kernel_prepare_store_part_3_pack_nodes' has just recorded its
        // slots (same condition: at least one node evicted); the device draws them until the
        // published host rendering buffers show them
        evictionGenerationHost++;
        COPY_TO_GPU(evictionGeneration, &evictionGenerationHost, uint32_t);
        // println("\n\nNb nodes to store: {}\n\n\n", nbNodesToStore);
    }

    storeNodes(nbNodesToStore);
}


void GpuVersion::storeNodes(uint32_t nb_nodes_to_store){
    if(!hasStartedStoring){
        if(!hasStartedSynchronisingEventInStoreNodes){
            std::vector<CUdeviceptr> srcs_device = {
                (CUdeviceptr)hostStaging.exchangedAABBIndices,
                (CUdeviceptr)hostStaging.exchangedPointsCounters,
                (CUdeviceptr)hostStaging.exchangedVoxelsCounters,
                (CUdeviceptr)hostStaging.exchangedAABBParentsIndices,
                (CUdeviceptr)hostStaging.exchangedChildrenIds,
                (CUdeviceptr)hostStaging.exchangedAABBs
            };
            std::vector<CUdeviceptr> dsts_host = {
                (CUdeviceptr)exchangedIds,
                (CUdeviceptr)exchangedPointsCounters,
                (CUdeviceptr)exchangedVoxelsCounters,
                (CUdeviceptr)exchangedParentsIds,
                (CUdeviceptr)exchangedChildrenIds,
                (CUdeviceptr)exchangedAABBs
            };
            std::vector<uint64_t> sizes = {
                nb_nodes_to_store * sizeof(CIdAABB),
                nb_nodes_to_store * sizeof(uint32_t),
                nb_nodes_to_store * sizeof(uint32_t),
                nb_nodes_to_store * sizeof(CIdAABB),
                nb_nodes_to_store * sizeof(uint32_t),
                nb_nodes_to_store * sizeof(CAABB)
            };

            // Wait for the node properties
            uint64_t nb_copies = sizes.size();
            CURuntime::assertCudaSuccess(cuMemcpyBatchAsync(
                dsts_host.data(), srcs_device.data(), sizes.data(), nb_copies,
                batchStoringAttributes.data(),
                batchStoringAttributesIndices.data(),
                batchStoringAttributes.size(),
                stream
            ));
            CURuntime::assertCudaSuccess(cuEventRecord(eventStoringComplete, stream));

            hasStartedSynchronisingEventInStoreNodes = true;
        }
        switch(cuEventQuery(eventStoringComplete)){
            case CUDA_SUCCESS:
                isDoneSynchronisingEventInStoreNodes = true;
                break;
            case CUDA_ERROR_NOT_READY:
                isDoneSynchronisingEventInStoreNodes = false;
                return;
            default:
                println("ERROR: store nodes failed to query event");
                throw(EXIT_FAILURE);
        }
        hasStartedSynchronisingEventInStoreNodes = false;

        CIdAABB* ids = static_cast<CIdAABB*>(exchangedIds);

        // Deserialise in parallel
        storingToDeserialise = {};
        for(uint32_t i=0; i < nb_nodes_to_store; i++){
            const CIdAABB& id = ids[i];
            hostCache->add(id);
            currentlyInUpdatesCache.erase(id);

            if(!updateNodes.contains(id)){
                updateNodes[id] = std::make_shared<HostStorageNode>(HostStorageNodeOwner::Update);
                updateNodes[id]->node.aabb_index = id;
                if(storedNodes.contains(id)){
                    storingToDeserialise.push_back(id);
                }
            }
        }

        hasStartedStoring = true;

        if(!storingToDeserialise.empty()){
            isDoneDeserializingForStoring = false;

            if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
                std::thread deserialize_thread([&]() {
                    std::for_each(storingToDeserialise.begin(), storingToDeserialise.end(),
                        [](const CIdAABB& id){
                            OctreeNodeSerializable::deserializeV2(updateNodes[id].get(), id, "From update cache");
                        }
                    );
                    isDoneDeserializingForStoring = true;
                });
                deserialize_thread.detach();
                // Skip sending to device this frame; the next frame will pick it up
                return;
            } else {
                std::for_each(storingToDeserialise.begin(), storingToDeserialise.end(), [](const CIdAABB& id){
                    OctreeNodeSerializable::deserializeV2(updateNodes[id].get(), id, "From update cache");
                });
                isDoneDeserializingForStoring = true;
            }
        }
    }
    hasStartedStoring = false;

    std::vector<CUdeviceptr> srcs_device = {};
    std::vector<CUdeviceptr> dsts_host = {};
    std::vector<uint64_t> sizes = {};
    CIdAABB* ids = static_cast<CIdAABB*>(exchangedIds);
    uint32_t* nbs_points = static_cast<uint32_t*>(exchangedPointsCounters);
    uint32_t* nbs_voxels = static_cast<uint32_t*>(exchangedVoxelsCounters);
    uint32_t* children_ids = static_cast<uint32_t*>(exchangedChildrenIds);
    CIdAABB* parents_ids = static_cast<CIdAABB*>(exchangedParentsIds);
    CAABB* aabbs = static_cast<CAABB*>(exchangedAABBs);

    for(uint32_t i=0; i < nb_nodes_to_store; i++){
        const CIdAABB& id = ids[i];

        // Update node properties
        HostStorageNode* cur_node = updateNodes[id].get();
        cur_node->node.children_ids = children_ids[i];

        // Update the CPU version of the relationship map
        parentsMap[id] = parents_ids[i];
        aabbsMap[id] = aabbs[i];
        currentlyInUpdatesCache.erase(id);
        hostCache->add(id);

        // Prepare to load points
        if(nbs_points[i] > 0){
            uint32_t old_counter = cur_node->node.points_counter;
            cur_node->node.points_counter += nbs_points[i];
            srcs_device.push_back(exchangedPointsPointers[i]);
            dsts_host.push_back((CUdeviceptr)(cur_node->points + old_counter));
            sizes.push_back(nbs_points[i] * sizeof(CPoint));
        }

        // Prepare to load voxels and grids
        if(nbs_voxels[i] > 0){
            uint32_t old_counter = cur_node->node.voxels_counter;
            uint32_t new_counter = old_counter + nbs_voxels[i];
            cur_node->node.voxels_counter = min(OocSimLodSettings::MAX_POINTS_PER_LEAF, new_counter);
            if(new_counter > OocSimLodSettings::MAX_POINTS_PER_LEAF){
                // These voxels (and their occupancy cells) are dropped: the device has flagged the
                // node (`CFlagHasIncompleteVoxels') and draws its children instead. Printed once.
                static bool has_warned = false;
                if(!has_warned){
                    println("INFO: node {} has {} voxels but at most {} are stored per node; its children will be drawn instead of its voxels (further cases not reported)",
                        id, new_counter, OocSimLodSettings::MAX_POINTS_PER_LEAF
                    );
                    has_warned = true;
                }
            }
            uint32_t nb_voxels_to_load = cur_node->node.voxels_counter - old_counter;

            if(nb_voxels_to_load > 0){
                srcs_device.push_back(exchangedVoxelsPointers[i]);
                dsts_host.push_back((CUdeviceptr)(cur_node->voxels + old_counter));
                sizes.push_back(nb_voxels_to_load * sizeof(CPoint));

                srcs_device.push_back(exchangedGridsPointers[i]);
                dsts_host.push_back((CUdeviceptr)(cur_node->occupancy_indices + old_counter));
                sizes.push_back(nb_voxels_to_load * sizeof(uint64_t));
            }
        }
    }

    // Load the remaining data
    uint64_t nb_copies = sizes.size();
    if(nb_copies > 0){
        CURuntime::assertCudaSuccess(cuMemcpyBatchAsync(
            dsts_host.data(), srcs_device.data(), sizes.data(), nb_copies,
            batchStoringAttributes.data(),
            batchStoringAttributesIndices.data(),
            batchStoringAttributes.size(),
            stream
        ));
        CURuntime::assertCudaSuccess(cuEventRecord(eventStoringComplete, stream));
    }

    // The data of the last eviction generation is on its way to host RAM (complete once
    // `eventStoringComplete' is reached): record what each of its nodes must show in the host
    // rendering buffers before the device can stop drawing it (counts capped like the host reads)
    lastEvictedNodes.clear();
    lastEvictedNodes.reserve(nb_nodes_to_store);
    for(uint32_t i = 0; i < nb_nodes_to_store; i++){
        const HostStorageNode* node = updateNodes[ids[i]].get();
        EvictedNodeRecord record = {};
        record.id = ids[i];
        record.nb_points = std::min(node->node.points_counter, OocSimLodSettings::MAX_POINTS_PER_LEAF);
        record.nb_voxels = std::min(node->node.voxels_counter, OocSimLodSettings::MAX_POINTS_PER_LEAF);
        lastEvictedNodes.push_back(record);
    }
    evictionGenerationStored = evictionGenerationHost;
}









float getScreenSpaceSize(const CAABB& aabb){
    const mat4 view = VKRenderer::view.view;
    const mat4 proj = VKRenderer::view.proj;
    const float fwidth = VKRenderer::width;
    const float fheight = VKRenderer::height;

    // compute node boundaries in screen space
    vec4 p000 = {aabb.mins.x, aabb.mins.y, aabb.mins.z, 1.0f};
    vec4 p001 = {aabb.mins.x, aabb.mins.y, aabb.maxs.z, 1.0f};
    vec4 p010 = {aabb.mins.x, aabb.maxs.y, aabb.mins.z, 1.0f};
    vec4 p011 = {aabb.mins.x, aabb.maxs.y, aabb.maxs.z, 1.0f};
    vec4 p100 = {aabb.maxs.x, aabb.mins.y, aabb.mins.z, 1.0f};
    vec4 p101 = {aabb.maxs.x, aabb.mins.y, aabb.maxs.z, 1.0f};
    vec4 p110 = {aabb.maxs.x, aabb.maxs.y, aabb.mins.z, 1.0f};
    vec4 p111 = {aabb.maxs.x, aabb.maxs.y, aabb.maxs.z, 1.0f};

    mat4 transform = proj * view;
    vec4 ndc000 = transform * p000;
    vec4 ndc001 = transform * p001;
    vec4 ndc010 = transform * p010;
    vec4 ndc011 = transform * p011;
    vec4 ndc100 = transform * p100;
    vec4 ndc101 = transform * p101;
    vec4 ndc110 = transform * p110;
    vec4 ndc111 = transform * p111;

    vec4 s000 = ((ndc000 / ndc000.w) * 0.5f + 0.5f) * vec4{fwidth, fheight, 1.0f, 1.0f};
    vec4 s001 = ((ndc001 / ndc001.w) * 0.5f + 0.5f) * vec4{fwidth, fheight, 1.0f, 1.0f};
    vec4 s010 = ((ndc010 / ndc010.w) * 0.5f + 0.5f) * vec4{fwidth, fheight, 1.0f, 1.0f};
    vec4 s011 = ((ndc011 / ndc011.w) * 0.5f + 0.5f) * vec4{fwidth, fheight, 1.0f, 1.0f};
    vec4 s100 = ((ndc100 / ndc100.w) * 0.5f + 0.5f) * vec4{fwidth, fheight, 1.0f, 1.0f};
    vec4 s101 = ((ndc101 / ndc101.w) * 0.5f + 0.5f) * vec4{fwidth, fheight, 1.0f, 1.0f};
    vec4 s110 = ((ndc110 / ndc110.w) * 0.5f + 0.5f) * vec4{fwidth, fheight, 1.0f, 1.0f};
    vec4 s111 = ((ndc111 / ndc111.w) * 0.5f + 0.5f) * vec4{fwidth, fheight, 1.0f, 1.0f};

    auto min8 = [](float f0, float f1, float f2, float f3, 
        float f4, float f5, float f6, float f7
    ){
		float m0 = min(f0, f1);
		float m1 = min(f2, f3);
		float m2 = min(f4, f5);
		float m3 = min(f6, f7);
		float n0 = min(m0, m1);
		float n1 = min(m2, m3);
		return min(n0, n1);
	};

	auto max8 = [](float f0, float f1, float f2, float f3, 
        float f4, float f5, float f6, float f7
    ){
		float m0 = max(f0, f1);
		float m1 = max(f2, f3);
		float m2 = max(f4, f5);
		float m3 = max(f6, f7);
		float n0 = max(m0, m1);
		float n1 = max(m2, m3);
		return max(n0, n1);
	};

    float smin_x = min8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
    float smin_y = min8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);
    float smax_x = max8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
    float smax_y = max8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);

    float dx = smax_x - smin_x;
    float dy = smax_y - smin_y;

    return dx*dy;
}



namespace {

/// Max-min fair-share quotas (water filling)
/// Every node gets min(count, fair share), where the fair share is at least floor(budget / n):
/// a minimum of elements of every node is kept even when everything does not fit.
/// The rounding leftover is handed out to the truncated nodes; the sum never exceeds the budget.
std::vector<uint32_t> visWaterFillQuotas(const std::vector<uint32_t>& counts, uint32_t budget){
    const uint32_t n = static_cast<uint32_t>(counts.size());
    std::vector<uint32_t> quota(n, 0);
    if(n == 0 || budget == 0){ return quota; }

    std::vector<uint32_t> order(n);
    for(uint32_t i = 0; i < n; i++){ order[i] = i; }
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b){
        return counts[a] < counts[b];
    });

    uint32_t remaining = budget;
    uint32_t remaining_nodes = n;
    for(uint32_t idx : order){
        const uint32_t fair_share = remaining / remaining_nodes;
        quota[idx] = std::min(counts[idx], fair_share);
        remaining -= quota[idx];
        remaining_nodes--;
    }

    // Hand out the rounding leftover (at most one extra element per truncated node)
    for(uint32_t i = n; i > 0 && remaining > 0; i--){
        const uint32_t idx = order[i - 1];
        if(quota[idx] < counts[idx]){
            quota[idx]++;
            remaining--;
        }
    }
    return quota;
}

/// Fraction of the budget shared equally between every visible node (guaranteed minimum per node)
/// The rest goes to the nodes according to their distance to the camera (see `VIS_DISTANCE_EXPONENT')
// constexpr double VIS_MIN_SHARE_FRACTION = 0.1;
constexpr double VIS_MIN_SHARE_FRACTION = 0.0;

/// Distance weighting of the water filling: weight = (d_closest / d_node)^VIS_DISTANCE_EXPONENT
/// d_node is the distance from the camera to the CENTER of the node, d_closest the smallest one
/// among the selected nodes (so the weights are in (0, 1] whatever the scene units)
/// The weight strictly decreases with the distance: among the nodes that do not get all their
/// points, a closer node always gets more points than a farther one
/// 1 = a node twice as far gets half the points, 2 = a quarter, ...
// constexpr double VIS_DISTANCE_EXPONENT = 2.0;
constexpr double VIS_DISTANCE_EXPONENT = 3.0;

/// The distance is floored to this fraction of the node diagonal (the camera can be at the center)
constexpr double VIS_DISTANCE_FLOOR_FRACTION = 0.01;

/// Floored distance from `position' to the center of `aabb'; negative if it can't be computed
double visNodeCenterDistance(const glm::vec3& position, const CAABB& aabb){
    if(!(aabb.mins.x <= aabb.maxs.x && aabb.mins.y <= aabb.maxs.y && aabb.mins.z <= aabb.maxs.z)){
        return -1.0;   // never filled
    }
    const double cx = 0.5 * (double(aabb.mins.x) + double(aabb.maxs.x));
    const double cy = 0.5 * (double(aabb.mins.y) + double(aabb.maxs.y));
    const double cz = 0.5 * (double(aabb.mins.z) + double(aabb.maxs.z));
    const double dx = double(position.x) - cx;
    const double dy = double(position.y) - cy;
    const double dz = double(position.z) - cz;
    const double sx = double(aabb.maxs.x) - double(aabb.mins.x);
    const double sy = double(aabb.maxs.y) - double(aabb.mins.y);
    const double sz = double(aabb.maxs.z) - double(aabb.mins.z);
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double diagonal = std::sqrt(sx * sx + sy * sy + sz * sz);
    const double floored = std::max(distance, VIS_DISTANCE_FLOOR_FRACTION * diagonal);
    return (std::isfinite(floored) && floored > 0.0) ? floored : -1.0;
}

/// Water-filling weights of the given nodes from their center distance to the camera
/// A node whose distance can't be computed (should not happen) is treated as the farthest one
std::vector<float> visDistanceWeights(const std::vector<CAABB>& aabbs, const glm::vec3& camera_pos){
    const uint32_t n = static_cast<uint32_t>(aabbs.size());
    std::vector<double> distances(n, -1.0);
    double closest = std::numeric_limits<double>::infinity();
    double farthest = 0.0;
    for(uint32_t i = 0; i < n; i++){
        distances[i] = visNodeCenterDistance(camera_pos, aabbs[i]);
        if(distances[i] > 0.0){
            closest = std::min(closest, distances[i]);
            farthest = std::max(farthest, distances[i]);
        }
    }

    std::vector<float> weights(n, 1.0f);
    if(!std::isfinite(closest)){ return weights; }   // no distance known: equal weights
    for(uint32_t i = 0; i < n; i++){
        const double d = (distances[i] > 0.0) ? distances[i] : farthest;
        weights[i] = static_cast<float>(std::pow(closest / d, VIS_DISTANCE_EXPONENT));
    }
    return weights;
}

/// Weighted quotas (the weights are the distance weights, see `visDistanceWeights')
///  1. `VIS_MIN_SHARE_FRACTION' of the budget is water-filled equally: every node keeps at least
///     min(count, floor(VIS_MIN_SHARE_FRACTION * budget / n)) elements
///  2. the rest is water-filled with the weights: every node that is not saturated gets
///     lambda * weight elements (the same lambda for all), and the nodes that need fewer than
///     that get all their elements
///  3. what is left (rounding, nodes with a zero weight) is water-filled equally
/// The quotas never exceed the counts and their sum never exceeds the budget; if everything fits,
/// everything is kept
std::vector<uint32_t> visPriorityQuotas(const std::vector<uint32_t>& counts,
    const std::vector<float>& priorities, uint32_t budget
){
    const uint32_t n = static_cast<uint32_t>(counts.size());
    if(n == 0 || budget == 0){ return std::vector<uint32_t>(n, 0); }

    // 1. Guaranteed minimum
    const uint32_t min_budget = static_cast<uint32_t>(double(budget) * VIS_MIN_SHARE_FRACTION);
    std::vector<uint32_t> quota = visWaterFillQuotas(counts, min_budget);
    uint64_t used = 0;
    for(uint32_t q : quota){ used += q; }
    uint64_t remaining = uint64_t(budget) - used;   // used <= min_budget <= budget

    // 2. Weighted water filling on what is not taken yet
    std::vector<uint32_t> order = {};
    std::vector<double> weights(n, 0.0);
    order.reserve(n);
    for(uint32_t i = 0; i < n; i++){
        const float p = (i < priorities.size()) ? priorities[i] : 0.0f;
        weights[i] = (std::isfinite(p) && p > 0.0f) ? double(p) : 0.0;
        if(weights[i] > 0.0 && quota[i] < counts[i]){ order.push_back(i); }
    }
    // Ascending residual / weight: the nodes saturating first are served first
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b){
        const double ra = double(counts[a] - quota[a]);
        const double rb = double(counts[b] - quota[b]);
        return ra * weights[b] < rb * weights[a];
    });
    // Suffix sums of the weights (avoids accumulating subtraction errors)
    std::vector<double> weights_left(order.size() + 1, 0.0);
    for(uint32_t k = static_cast<uint32_t>(order.size()); k > 0; k--){
        weights_left[k - 1] = weights_left[k] + weights[order[k - 1]];
    }
    for(uint32_t k = 0; k < order.size() && remaining > 0; k++){
        const uint32_t idx = order[k];
        const uint64_t residual = counts[idx] - quota[idx];
        const double share = double(remaining) * weights[idx] / weights_left[k];
        uint64_t q = (share >= double(residual)) ? residual : static_cast<uint64_t>(std::floor(share));
        q = std::min(q, remaining);
        quota[idx] += static_cast<uint32_t>(q);
        remaining -= q;
    }

    // 3. Leftover shared equally
    if(remaining > 0){
        std::vector<uint32_t> residuals(n);
        for(uint32_t i = 0; i < n; i++){ residuals[i] = counts[i] - quota[i]; }
        std::vector<uint32_t> extra = visWaterFillQuotas(residuals, static_cast<uint32_t>(remaining));
        for(uint32_t i = 0; i < n; i++){ quota[i] += extra[i]; }
    }
    return quota;
}

/// Read the first `nb_elements' points of an append-only points / voxels file
/// Returns the number of elements really read
uint32_t visReadPointsFile(const std::string& filepath, uint32_t nb_elements, std::vector<CPoint>& out){
    if(nb_elements == 0){ return 0; }
    if(out.size() < nb_elements){ out.resize(nb_elements); }

    std::ifstream file(filepath, std::ios::binary);
    if(!file.is_open()){ return 0; }
    file.read(
        reinterpret_cast<char*>(out.data()),
        static_cast<std::streamsize>(uint64_t(nb_elements) * sizeof(CPoint))
    );
    return static_cast<uint32_t>(uint64_t(file.gcount()) / sizeof(CPoint));
}

/// Copy `quota' uniformly chosen elements among the first `nb_read' ones into `dst'
/// Deterministic for a given seed: a smaller quota selects a prefix of a larger one's selection,
/// so the rendered subset of a node stays stable from one update to the next
uint32_t visCopyRandomSubset(std::vector<CPoint>& data, uint32_t nb_read, uint32_t quota, uint32_t seed, CPoint* dst){
    const uint32_t nb_to_copy = std::min(quota, nb_read);
    if(nb_to_copy == 0){ return 0; }

    if(nb_to_copy < nb_read){
        // Partial Fisher-Yates: the first `nb_to_copy' elements become a uniform random subset
        std::minstd_rand rng(seed);
        for(uint32_t i = 0; i < nb_to_copy; i++){
            std::uniform_int_distribution<uint32_t> dist(i, nb_read - 1);
            std::swap(data[i], data[dist(rng)]);
        }
    }

    std::copy_n(data.data(), nb_to_copy, dst);
    return nb_to_copy;
}

std::atomic<bool> visHasWarnedShortRead{false};

/// True if the host RAM nodes (`updateNodes') can be read on the main thread right now
/// Their buffers are written by:
///  - the load / store deserialization threads (they only fill freshly created nodes),
///  - the device->host copies of `storeNodes' (tracked by `eventStoringComplete'); their counters
///    are increased on the main thread BEFORE the copies complete.
/// The serialization thread of `updateHostCache' only reads them (and writes the `*_last_stored'
/// fields, not read here); it deallocates nodes, but only nodes already erased from `updateNodes'.
bool visCanReadUpdateNodes(){
    if(!GpuVersion::isDoneDeserializingForLoading || !GpuVersion::isDoneDeserializingForStoring){ return false; }
    if(GpuVersion::hasStartedSynchronisingEventInStoreNodes){ return false; }
    const CUresult res = cuEventQuery(GpuVersion::eventStoringComplete);
    if(res == CUDA_ERROR_NOT_READY){ return false; }
    CURuntime::assertCudaSuccess(res);
    return true;
}

/// Copy `quota' elements out of the first `nb_elements' of `src' into `dst' WITHOUT modifying `src'
/// (a host RAM node is append-only and may be read by the serialization thread): evenly strided
/// over the insertion order, deterministic, a smaller quota is close to a subset of a larger one
uint32_t visCopyStridedSubset(const CPoint* src, uint32_t nb_elements, uint32_t quota, CPoint* dst){
    const uint32_t nb_to_copy = std::min(quota, nb_elements);
    if(nb_to_copy == 0){ return 0; }
    if(nb_to_copy == nb_elements){
        std::copy_n(src, nb_elements, dst);
        return nb_elements;
    }
    for(uint32_t j = 0; j < nb_to_copy; j++){
        dst[j] = src[uint32_t((uint64_t(j) * nb_elements) / nb_to_copy)];
    }
    return nb_to_copy;
}

/// Stream one node: read it from disk into regular RAM, then copy its quota into the staging buffers
/// `points_buffer' / `voxels_buffer' are per-worker scratch buffers, reused between nodes
void visLoadTask(GpuVersion::VisibilityTask& task,
    std::vector<CPoint>& points_buffer, std::vector<CPoint>& voxels_buffer
){
    // Already copied from the host RAM on the main thread (`visibilityUpdateSort')
    if(task.is_from_update_node){ return; }

    task.points_written = 0;
    task.voxels_written = 0;

    const bool wants_points = task.points_quota > 0;
    const bool wants_voxels = task.voxels_quota > 0;
    if(!wants_points && !wants_voxels){ return; }

    uint32_t nb_points_read = 0;
    uint32_t nb_voxels_read = 0;
    {
        // The serialization thread writes these files under the same lock
        std::lock_guard<std::mutex> lock(GpuVersion::storedNodesMtx[task.id]);
        if(wants_points){
            nb_points_read = visReadPointsFile(getChunkFilePathV2(task.id, false), task.nb_points_on_disk, points_buffer);
        }
        if(wants_voxels){
            nb_voxels_read = visReadPointsFile(getChunkFilePathV2(task.id, true), task.nb_voxels_on_disk, voxels_buffer);
        }
    }

    if((wants_points && nb_points_read < task.nb_points_on_disk)
        || (wants_voxels && nb_voxels_read < task.nb_voxels_on_disk)
    ){
        if(!visHasWarnedShortRead.exchange(true)){
            println("WARN: visibility update, node {} has less data on disk than expected (points {}/{}, voxels {}/{})",
                task.id, nb_points_read, task.nb_points_on_disk, nb_voxels_read, task.nb_voxels_on_disk
            );
        }
    }

    const uint32_t seed = static_cast<uint32_t>(task.id) * 2654435761u;
    if(wants_points){
        task.points_written = visCopyRandomSubset(points_buffer, nb_points_read, task.points_quota, seed + 1u,
            GpuVersion::visStagingPoints + task.points_offset
        );
    }
    if(wants_voxels){
        task.voxels_written = visCopyRandomSubset(voxels_buffer, nb_voxels_read, task.voxels_quota, seed + 2u,
            GpuVersion::visStagingVoxels + task.voxels_offset
        );
    }
}

/// Process every task of `GpuVersion::visibilityTasks' with a pool of workers
void visRunTasks(uint32_t max_nb_workers){
    std::vector<GpuVersion::VisibilityTask>& tasks = GpuVersion::visibilityTasks;
    const uint32_t nb_tasks = static_cast<uint32_t>(tasks.size());
    const uint32_t nb_workers = std::min(nb_tasks, std::max(1u, max_nb_workers));
    if(nb_tasks == 0){ return; }

    std::atomic<uint32_t> next = 0;
    auto work = [&](){
        std::vector<CPoint> points_buffer;
        std::vector<CPoint> voxels_buffer;
        for(uint32_t i = next.fetch_add(1); i < nb_tasks; i = next.fetch_add(1)){
            visLoadTask(tasks[i], points_buffer, voxels_buffer);
        }
    };

    if(nb_workers == 1){
        work();
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(nb_workers);
    for(uint32_t w = 0; w < nb_workers; w++){
        workers.emplace_back(work);
    }
    for(std::thread& t : workers){ t.join(); }
}

} // namespace



/// Find the visible nodes (single pass, no node limit) and build the streaming tasks:
/// which nodes, how many points / voxels each one may write, and where in the staging buffers
void GpuVersion::visibilityUpdateSort(CuRast* editor, CUcontext* context){
    visibilityTasks.clear();

    uint32_t block_size = 256;
    uint32_t grid_size =
        (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1)
        / block_size
    ;
    OptionalLaunchSettings launch_settings = {
        .gridsize  = grid_size,
        .blocksize = block_size
    };
    OptionalLaunchSettings launch_settings_coop = {
        .gridsize  = 0,
        .blocksize = 256
    };

    // Single pass: every node of the cut is kept, the budget is enforced on points / voxels instead
    renderingSettings.min_pixel_span = CuRastSettings::minPixelSpan;
    prog->launch("kernel_get_renderable_nodes_part_1_visibility", {&renderingTarget, &renderingSettings}, launch_settings);
    prog->launch("kernel_get_renderable_nodes_part_2_flagging_large", {&renderingTarget, &renderingSettings}, launch_settings);
    prog->launch("kernel_get_renderable_nodes_part_3_large_nodes", {&renderingTarget, &renderingSettings}, launch_settings);
    prog->launch("kernel_get_renderable_nodes_part_4_small_nodes", {&renderingTarget, &renderingSettings}, launch_settings);
    // Sort by screen space size on device side (also outputs the complete lists)
    prog->launchCooperative("kernel_get_renderable_nodes_part_5_reorder", {&renderingTarget, &renderingSettings}, launch_settings_coop);

    COPY_FROM_GPU(nbNodesExchangedVisPoints, nbNodesExchangedVisPoints, uint32_t);
    COPY_FROM_GPU(nbNodesExchangedVisVoxels, nbNodesExchangedVisVoxels, uint32_t);

    // The device lists (and the host copies) hold at most `maxNbConcurrentNodes' entries
    const uint32_t nb_vis_points = min(*(uint32_t*)nbNodesExchangedVisPoints, hostStaging.maxNbConcurrentNodes);
    const uint32_t nb_vis_voxels = min(*(uint32_t*)nbNodesExchangedVisVoxels, hostStaging.maxNbConcurrentNodes);
    *(uint32_t*)nbNodesExchangedVisPoints = nb_vis_points;
    *(uint32_t*)nbNodesExchangedVisVoxels = nb_vis_voxels;

    std::vector<CUdeviceptr> dsts_host = {};
    std::vector<CUdeviceptr> srcs_device = {};
    std::vector<uint64_t> sizes = {};

    // Ids and their priorities: `kernel_get_renderable_nodes_part_5_reorder' sorts both arrays
    // together, so entry i of the priorities belongs to entry i of the ids
    if(nb_vis_points > 0){
        dsts_host.push_back((CUdeviceptr)exchangedAABBIndicesVisPoints);
        srcs_device.push_back((CUdeviceptr)hostStaging.exchangedAABBIndicesVisPoints);
        sizes.push_back(sizeof(CIdAABB) * nb_vis_points);

        dsts_host.push_back((CUdeviceptr)exchangedPrioritiesVisPoints);
        srcs_device.push_back((CUdeviceptr)hostStaging.exchangedAABBIndicesVisPointsScreenSpaceSize);
        sizes.push_back(sizeof(float) * nb_vis_points);
    }
    if(nb_vis_voxels > 0){
        dsts_host.push_back((CUdeviceptr)exchangedAABBIndicesVisVoxels);
        srcs_device.push_back((CUdeviceptr)hostStaging.exchangedAABBIndicesVisVoxels);
        sizes.push_back(sizeof(CIdAABB) * nb_vis_voxels);

        dsts_host.push_back((CUdeviceptr)exchangedPrioritiesVisVoxels);
        srcs_device.push_back((CUdeviceptr)hostStaging.exchangedAABBIndicesVisVoxelsScreenSpaceSize);
        sizes.push_back(sizeof(float) * nb_vis_voxels);
    }

    // Nothing visible: no copy, but the rest still runs (empty tasks, eviction hand-over below)
    if(!sizes.empty()){
        CURuntime::assertCudaSuccess(cuMemcpyBatchAsync(
            dsts_host.data(), srcs_device.data(), sizes.data(), sizes.size(),
            batchStoringAttributes.data(),
            batchStoringAttributesIndices.data(),
            batchStoringAttributes.size(),
            visStream
        ));
        CURuntime::assertCudaSuccess(cuEventRecord(eventVisPreprocess, visStream));
        CURuntime::assertCudaSuccess(cuEventSynchronize(eventVisPreprocess));
    }

    // One task per distinct node; a node can be in both lists (points and voxels)
    // Each node is read either from its files on disk, or from its host RAM copy (`updateNodes')
    // when it is not written to disk yet or when the host RAM copy is newer (node just evicted from
    // the device): this closes the gap between the eviction (the device deletes the node) and the
    // moment its files are written, during which the node was drawn by nobody.
    // No memory is needed for this: the host RAM copies already exist, they are copied straight
    // into the staging buffers below.
    struct VisSource {
        bool valid = false;                         // some data is available
        bool from_update_node = false;
        const HostStorageNode* update_node = nullptr;
        uint32_t nb_points = 0;
        uint32_t nb_voxels = 0;
    };
    const bool can_read_update_nodes = visCanReadUpdateNodes();
    // Deterministic within this call (nothing modifies `updateNodes' or the counters meanwhile: the
    // counters are only written under `storedNodesVisibilityMtx', held below), so a node in both
    // lists gets the same source for its points and its voxels
    auto resolve_source = [&](const CIdAABB& id) -> VisSource {
        VisSource source = {};
        uint32_t disk_points = 0;
        uint32_t disk_voxels = 0;
        bool is_on_disk = false;
        auto disk_it = storedNodesVisibilityCounters.find(id);
        if(disk_it != storedNodesVisibilityCounters.end()){
            is_on_disk = true;
            disk_points = disk_it->second.nb_points;
            disk_voxels = disk_it->second.nb_voxels;
        }

        if(can_read_update_nodes){
            auto ram_it = updateNodes.find(id);
            if(ram_it != updateNodes.end() && ram_it->second
                && ram_it->second->points && ram_it->second->voxels
            ){
                const HostStorageNode* node = ram_it->second.get();
                const uint32_t ram_points = std::min(node->node.points_counter, OocSimLodSettings::MAX_POINTS_PER_LEAF);
                const uint32_t ram_voxels = std::min(node->node.voxels_counter, OocSimLodSettings::MAX_POINTS_PER_LEAF);
                // A host RAM copy holds the on-disk data plus the newer one (append-only)
                const bool is_complete = ram_points >= disk_points && ram_voxels >= disk_voxels;
                const bool is_newer = !is_on_disk || ram_points > disk_points || ram_voxels > disk_voxels;
                if(is_complete && is_newer && (ram_points > 0 || ram_voxels > 0)){
                    source.valid = true;
                    source.from_update_node = true;
                    source.update_node = node;
                    source.nb_points = ram_points;
                    source.nb_voxels = ram_voxels;
                    return source;
                }
            }
        }

        if(is_on_disk){
            source.valid = true;
            source.nb_points = disk_points;
            source.nb_voxels = disk_voxels;
        }
        return source;
    };

    std::unordered_map<CIdAABB, uint32_t> task_index = {};
    task_index.reserve(nb_vis_points + nb_vis_voxels);
    auto get_task = [&](const CIdAABB& id, const VisSource& source) -> VisibilityTask& {
        auto [it, inserted] = task_index.try_emplace(id, static_cast<uint32_t>(visibilityTasks.size()));
        if(inserted){
            VisibilityTask task = {};
            task.id = id;
            task.is_from_update_node = source.from_update_node;
            task.update_node = source.update_node;
            visibilityTasks.push_back(task);
        }
        return visibilityTasks[it->second];
    };

    {
        std::lock_guard<std::mutex> lock(storedNodesVisibilityMtx);
        for(uint32_t i = 0; i < nb_vis_voxels; i++){
            const CIdAABB id = ((CIdAABB*)(exchangedAABBIndicesVisVoxels))[i];
            const VisSource source = resolve_source(id);
            if(!source.valid || source.nb_voxels == 0){continue;}
            VisibilityTask& task = get_task(id, source);
            task.nb_voxels_on_disk = source.nb_voxels;
            task.voxels_priority = ((float*)(exchangedPrioritiesVisVoxels))[i];
        }
        for(uint32_t i = 0; i < nb_vis_points; i++){
            const CIdAABB id = ((CIdAABB*)(exchangedAABBIndicesVisPoints))[i];
            const VisSource source = resolve_source(id);
            if(!source.valid || source.nb_points == 0){continue;}
            VisibilityTask& task = get_task(id, source);
            task.nb_points_on_disk = source.nb_points;
            task.points_priority = ((float*)(exchangedPrioritiesVisPoints))[i];
        }

        // Eviction hand-over: this update shows the last evicted batch if every node of it can be
        // read (from host RAM or from disk) with at least the data it had when it reached host RAM.
        // Once these buffers are published (`updateVisCache'), the device stops drawing the batch
        // from the exchange buffers. Otherwise the device keeps drawing it (if a node is also
        // drawn by these buffers meanwhile, it is drawn twice with the same points: no visual change)
        uint32_t covered_generation = (evictionGenerationHost > 0) ? evictionGenerationHost - 1 : 0;
        if(evictionGenerationHost > 0 && evictionGenerationStored == evictionGenerationHost){
            bool is_covered = true;
            for(const EvictedNodeRecord& record : lastEvictedNodes){
                if(record.nb_points == 0 && record.nb_voxels == 0){continue;}
                const VisSource source = resolve_source(record.id);
                if(!source.valid || source.nb_points < record.nb_points || source.nb_voxels < record.nb_voxels){
                    is_covered = false;
                    break;
                }
            }
            if(is_covered){ covered_generation = evictionGenerationHost; }
        }
        visCoveredEvictionGeneration = covered_generation;
    }

    // Quotas and reserved slots in the staging buffers (the sum of the quotas fits the budget)
    // The device sort (screen space size) selected the nodes; the budget is shared between them
    // according to their distance to the camera: the closer, the more points / voxels
    // (`aabbsMap' is filled by `storeNodes' for every node sent to the host, so for every node on disk;
    // `renderingTarget' is the camera the kernels above used)
    const glm::vec3 camera_pos = renderingTarget.camera_pos;
    {
        std::vector<uint32_t> task_ids = {};
        std::vector<uint32_t> counts = {};
        std::vector<CAABB> aabbs = {};
        for(uint32_t i = 0; i < visibilityTasks.size(); i++){
            if(visibilityTasks[i].nb_points_on_disk == 0){continue;}
            task_ids.push_back(i);
            counts.push_back(visibilityTasks[i].nb_points_on_disk);
            aabbs.push_back(aabbsMap[visibilityTasks[i].id]);
        }
        const std::vector<float> weights = visDistanceWeights(aabbs, camera_pos);
        std::vector<uint32_t> quotas = visPriorityQuotas(counts, weights, hostStaging.maxNbRenderedPoints);
        uint64_t offset = 0;
        for(uint32_t j = 0; j < task_ids.size(); j++){
            VisibilityTask& task = visibilityTasks[task_ids[j]];
            task.points_quota = quotas[j];
            task.points_offset = offset;
            offset += quotas[j];
        }
    }
    {
        std::vector<uint32_t> task_ids = {};
        std::vector<uint32_t> counts = {};
        std::vector<CAABB> aabbs = {};
        for(uint32_t i = 0; i < visibilityTasks.size(); i++){
            if(visibilityTasks[i].nb_voxels_on_disk == 0){continue;}
            task_ids.push_back(i);
            counts.push_back(visibilityTasks[i].nb_voxels_on_disk);
            aabbs.push_back(aabbsMap[visibilityTasks[i].id]);
        }
        const std::vector<float> weights = visDistanceWeights(aabbs, camera_pos);
        std::vector<uint32_t> quotas = visPriorityQuotas(counts, weights, hostStaging.maxNbRenderedVoxels);
        uint64_t offset = 0;
        for(uint32_t j = 0; j < task_ids.size(); j++){
            VisibilityTask& task = visibilityTasks[task_ids[j]];
            task.voxels_quota = quotas[j];
            task.voxels_offset = offset;
            offset += quotas[j];
        }
    }

    // Host RAM nodes: copied now, on the main thread which owns `updateNodes' (nothing modified them
    // since `resolve_source'). The staging buffers are free: `visibilityUpdate' only starts a new
    // update once the previous copy is complete. The workers skip these tasks.
    for(VisibilityTask& task : visibilityTasks){
        if(!task.is_from_update_node){continue;}
        const HostStorageNode* src = task.update_node;
        task.points_written = visCopyStridedSubset(src->points, task.nb_points_on_disk, task.points_quota,
            visStagingPoints + task.points_offset
        );
        task.voxels_written = visCopyStridedSubset(src->voxels, task.nb_voxels_on_disk, task.voxels_quota,
            visStagingVoxels + task.voxels_offset
        );
        task.update_node = nullptr;   // not valid after this function
    }
}

void GpuVersion::shuffleForRendering(HostStorageNode* node, CIdAABB id){
    std::minstd_rand rng(static_cast<uint32_t>(id) * 2654435761u + 1u);
    std::shuffle(node->points, node->points + node->node.points_counter, rng);
    std::shuffle(node->voxels, node->voxels + node->node.voxels_counter, rng);
}


/// Stream the visible nodes from disk into the rendering buffers
/// 1. (main thread) find the visible nodes and their quotas, see `visibilityUpdateSort'
/// 2. (workers) read each node into regular RAM, copy a random subset of its quota into the
///    pinned staging buffers, free it
/// 3. (main thread) one batched copy from the staging buffers to the write-side device buffers;
///    the device switches to them in `updateVisCache' once the copy is complete
void GpuVersion::visibilityUpdate(CuRast* editor, CUcontext* context){
    if(!hasStartedVisibilityUpdate){
        // The previous content must be published (the device reads it) and its copy complete
        // (the staging buffers are about to be overwritten) before starting a new one
        if(isVisibilityPublishPending){return;}
        CUresult res = cuEventQuery(eventVisibilityUpdateComplete);
        if(res == CUDA_ERROR_NOT_READY){return;}
        CURuntime::assertCudaSuccess(res);

        // Write into the buffer pair the device is NOT reading
        isUsingSecondRenderingBuffer = !isUsingSecondRenderingBuffer;

        visibilityUpdateSort(editor, context);
        hasStartedVisibilityUpdate = true;

        if(!visibilityTasks.empty()){
            const uint32_t max_nb_workers = std::max(1u, std::thread::hardware_concurrency() / 4);
            if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
                isDoneDeserializingForVisibility = false;
                // `visibilityTasks' and the staging buffers belong to the workers until
                // `isDoneDeserializingForVisibility' is true again
                std::thread coordinator([max_nb_workers](){
                    visRunTasks(max_nb_workers);
                    isDoneDeserializingForVisibility = true;
                });
                coordinator.detach();
                // The next frames will pick it up
                return;
            } else {
                visRunTasks(1);
            }
        }
    }

    if(!isDoneDeserializingForVisibility){return;}
    hasStartedVisibilityUpdate = false;

    // Select the write-side buffers
    void* nb_rendered_points_host = isUsingSecondRenderingBuffer ? nbRenderedPoints2 : nbRenderedPoints;
    void* nb_rendered_voxels_host = isUsingSecondRenderingBuffer ? nbRenderedVoxels2 : nbRenderedVoxels;

    CUdeviceptr dst_rendered_points = isUsingSecondRenderingBuffer
        ? (CUdeviceptr)hostStaging.renderedPoints2
        : (CUdeviceptr)hostStaging.renderedPoints;
    CUdeviceptr dst_rendered_voxels = isUsingSecondRenderingBuffer
        ? (CUdeviceptr)hostStaging.renderedVoxels2
        : (CUdeviceptr)hostStaging.renderedVoxels;

    CUdeviceptr dst_nb_points = deviceStaging + (
        reinterpret_cast<uintptr_t>(isUsingSecondRenderingBuffer
            ? &hostStaging.nbRenderedPoints2
            : &hostStaging.nbRenderedPoints)
        - reinterpret_cast<uintptr_t>(&hostStaging)
    );
    CUdeviceptr dst_nb_voxels = deviceStaging + (
        reinterpret_cast<uintptr_t>(isUsingSecondRenderingBuffer
            ? &hostStaging.nbRenderedVoxels2
            : &hostStaging.nbRenderedVoxels)
        - reinterpret_cast<uintptr_t>(&hostStaging)
    );

    // Totals of what the workers really wrote (<= quotas <= budgets)
    uint64_t total_points = 0;
    uint64_t total_voxels = 0;
    uint64_t total_points_on_disk = 0;
    uint64_t total_voxels_on_disk = 0;
    for(const VisibilityTask& task : visibilityTasks){
        total_points += task.points_written;
        total_voxels += task.voxels_written;
        total_points_on_disk += task.nb_points_on_disk;
        total_voxels_on_disk += task.nb_voxels_on_disk;
    }
    *(uint32_t*)nb_rendered_points_host = static_cast<uint32_t>(total_points);
    *(uint32_t*)nb_rendered_voxels_host = static_cast<uint32_t>(total_voxels);

    GpuVersionUI::visNbNodes  = static_cast<uint32_t>(visibilityTasks.size());
    GpuVersionUI::visNbPoints = static_cast<uint32_t>(total_points);
    GpuVersionUI::visNbVoxels = static_cast<uint32_t>(total_voxels);

    std::vector<CUdeviceptr> srcs_host = {
        (CUdeviceptr)nb_rendered_points_host,
        (CUdeviceptr)nb_rendered_voxels_host
    };
    std::vector<CUdeviceptr> dsts_device = {
        dst_nb_points,
        dst_nb_voxels
    };
    std::vector<uint64_t> sizes = {
        sizeof(uint32_t),
        sizeof(uint32_t)
    };

    // The written elements are packed contiguously on the device. A node that wrote less than its
    // quota leaves a gap in the staging buffer: consecutive full slots are merged into one copy
    auto append_segments = [&](const CPoint* staging, CUdeviceptr dst_base,
        uint64_t VisibilityTask::* offset_member, uint32_t VisibilityTask::* written_member
    ){
        uint64_t run_src = 0;
        uint64_t run_dst = 0;
        uint64_t run_len = 0;
        uint64_t dst_cursor = 0;
        auto flush = [&](){
            if(run_len == 0){return;}
            srcs_host.push_back((CUdeviceptr)(staging + run_src));
            dsts_device.push_back(dst_base + (CUdeviceptr)(run_dst * sizeof(CPoint)));
            sizes.push_back(run_len * sizeof(CPoint));
            run_len = 0;
        };
        for(const VisibilityTask& task : visibilityTasks){
            const uint64_t written = task.*written_member;
            if(written == 0){continue;}
            const uint64_t src = task.*offset_member;
            if(run_len > 0 && src == run_src + run_len){
                run_len += written;
            } else {
                flush();
                run_src = src;
                run_dst = dst_cursor;
                run_len = written;
            }
            dst_cursor += written;
        }
        flush();
    };
    append_segments(visStagingVoxels, dst_rendered_voxels, &VisibilityTask::voxels_offset, &VisibilityTask::voxels_written);
    append_segments(visStagingPoints, dst_rendered_points, &VisibilityTask::points_offset, &VisibilityTask::points_written);

    // // TODO: to remove
    // if(total_voxels < total_voxels_on_disk || total_points < total_points_on_disk){
    //     println("{} nodes, voxels: {} / {}, points: {} / {}",
    //         visibilityTasks.size(),
    //         total_voxels, total_voxels_on_disk,
    //         total_points, total_points_on_disk
    //     );
    // }

    // Send the data to the device
    CURuntime::assertCudaSuccess(cuMemcpyBatchAsync(
        dsts_device.data(), srcs_host.data(), sizes.data(), sizes.size(),
        batchLoadingAttributes.data(),
        batchLoadingAttributesIndices.data(),
        batchLoadingAttributes.size(),
        visStream
    ));
    CURuntime::assertCudaSuccess(cuEventRecord(eventVisibilityUpdateComplete, visStream));
    isVisibilityPublishPending = true;
}


void GpuVersion::updateHostCache(){
    if(!isDoneUpdatingHostCache){return;}
    if(!isDoneDeserializingForLoading 
        || !isDoneDeserializingForStoring
        || !isDoneSynchronisingEventInStoreNodes
    ){return;}

    if(!hasStartedSynchronisingStoringCompleteEvent){
        hostCacheToSerialise = {};
        hostCacheToDelete = {};
        for(auto it = updateNodes.begin(); it != updateNodes.end();){
            const CIdAABB& id = it->first;
            std::shared_ptr<HostStorageNode>& node = it->second;
            if(!hostCache->contains(id)){
                hostCacheToSerialise.push_back(node);
                hostCacheToDelete.push_back(node);
                it = updateNodes.erase(it);
            } else {
                // if(!storedNodes.contains(id)){
                //     hostCacheToSerialise.push_back(node);
                // }
                // it++;

                const COctreeNode& n = node->node;
                bool is_dirty = n.points_counter != n.points_last_stored
                    || n.voxels_counter != n.voxels_last_stored;
                if(!storedNodes.contains(id) || is_dirty){
                    hostCacheToSerialise.push_back(node);
                }
                it++;
            }
        }

        hasStartedSynchronisingStoringCompleteEvent = true;
    }

    switch(cuEventQuery(eventStoringComplete)){
        case CUDA_SUCCESS:
            isDoneSynchronisingStoringCompleteEvent = true;
            break;
        case CUDA_ERROR_NOT_READY:
            isDoneSynchronisingStoringCompleteEvent = false;
            return;
        default:
            println("ERROR: update host cache failed to query event");
            throw(EXIT_FAILURE);
    }
    hasStartedSynchronisingStoringCompleteEvent = false;

    if(hostCacheToSerialise.empty()){return;}
    isDoneUpdatingHostCache = false;

    if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
        std::thread update_host_cache_thread([]()  {
                serialiseHostCacheNodes();
                isDoneUpdatingHostCache = true;
            }
        );
        update_host_cache_thread.detach();
    } else {
        serialiseHostCacheNodes();
        isDoneUpdatingHostCache = true;
    }

}


void GpuVersion::serialiseHostCacheNodes(){
    // Write the files, keeping the counters each node was written with
    std::vector<std::pair<CIdAABB, StoredNodeCounters>> written_counters = {};
    written_counters.reserve(hostCacheToSerialise.size());
    for(std::shared_ptr<HostStorageNode>& node : hostCacheToSerialise){
        const CIdAABB id = node->node.aabb_index;
        std::lock_guard<std::mutex> lock(storedNodesMtx[id]);
        OctreeNodeSerializable::serializeV2(node);
        written_counters.emplace_back(id, StoredNodeCounters{node->node.points_counter, node->node.voxels_counter});
    }

    // Publish them for the visibility update only now that the data is on disk
    for(const auto& [id, counters] : written_counters){
        if(!storedNodes.contains(id)){
            storedNodes.insert(id);
            nbStoredNodesInVisibility.fetch_add(1);
        }
        std::lock_guard<std::mutex> lock(storedNodesVisibilityMtx);
        storedNodesVisibilityCopy.insert(id);
        storedNodesVisibilityCounters[id] = counters;
    }

    // Deallocate the nodes that left the host cache
    for(std::shared_ptr<HostStorageNode>& node : hostCacheToDelete){
        node->deallocate();
    }
}


void GpuVersion::updateVisCache(){
    // Nothing to publish while the workers are filling the staging buffers
    if(!isDoneDeserializingForVisibility || hasStartedVisibilityUpdate){return;}
    if(!isDoneSortingVisibleNodes || hasStartedSortingVisibleNodes){return;}

    // Only switch the device to the new buffers once their copy is complete
    switch(cuEventQuery(eventVisibilityUpdateComplete)){
        case CUDA_SUCCESS:
            isDoneUpdatingVisCache = true;
            if(isVisibilityPublishPending){
                COPY_TO_GPU(isUsingSecondRenderingBuffer, &isUsingSecondRenderingBuffer, bool);
                // From now on the device draws these buffers: it can stop drawing the evicted batch
                // they cover (see `kernel_draw_evicted_nodes')
                COPY_TO_GPU(coveredEvictionGeneration, &visCoveredEvictionGeneration, uint32_t);
                isVisibilityPublishPending = false;
            }
            break;
        case CUDA_ERROR_NOT_READY:
            isDoneUpdatingVisCache = false;
            return;
        default:
            println("ERROR: update visibility cache failed to query event");
            throw(EXIT_FAILURE);
    }
}





void GpuVersion::updateOctree(CuRast* editor, CUcontext* context){
    if(isTakingScreenshots){return;}
	cuCtxSetCurrent(*context);

    GpuVersionUI::lastUpdateStart = high_resolution_clock::now();
    if(GpuVersionUI::nbTotalUpdates == 0){
        GpuVersionUI::firstUpdateStart = GpuVersionUI::lastUpdateStart;
    }
    LoaderGpuVersion::run(editor, context);

    if(isDoneUpdatingVisCache 
        && isDoneDeserializingForVisibility 
        && isDoneSortingVisibleNodes
        && *(bool*)isInitialised
    ){
        GpuVersion::visibilityUpdate(editor, context);
    }

    // if(!isDoneUpdatingHostCache){println("stall updating host cache");}
    // if(!isDoneSynchronisingStoringCompleteEvent){println("stall receiving nodes");}
    // if(!isDoneDeserializingForLoading){println("stall deserializing for load");}
    // if(!isDoneDeserializingForStoring){println("stall deserializing for store");}

    if(!isDoneUpdatingHostCache
        || !isDoneSynchronisingStoringCompleteEvent
        || !isDoneDeserializingForLoading
        || !isDoneDeserializingForStoring
    ){return;}

    if(!hasStartedStoring && isDoneSynchronisingEventInStoreNodes && !hasStartedLoading){
        // Only run the initialisation kernel once
        if(!*(bool*)isInitialised){
            octreeUpdateInit(editor, context);
            if(!*(bool*)isInitialised){return;}
        }

        // Only run the bottom up kernels if nothing else is stalling
        if(*(bool*)isDoneLoading && *(bool*)isDoneStoring && *(bool*)isDoneIterating){
            octreeUpdateBottomUp(editor, context);
        }
    }

    if(*(bool*)isUpdating){
        if(!hasStartedStoring && isDoneSynchronisingEventInStoreNodes && *(bool*)isDoneStoring){
            octreeUpdateSimLOD(editor, context);
            if(!isDoneDeserializingForLoading){return;}
        }

        if(!hasStartedLoading && *(bool*)isDoneLoading && *(bool*)isDoneIterating){
            octreeUpdateCacheUpdate(editor, context);
            if(!isDoneDeserializingForStoring){return;}
        }

        if(!hasStartedStoring && isDoneSynchronisingEventInStoreNodes && !hasStartedLoading){
            if(*(bool*)isDoneLoading && *(bool*)isDoneStoring && *(bool*)isDoneIterating){
                OptionalLaunchSettings launch_settings = {
                    .gridsize = 1,
                    .blocksize = 1
                };
                prog->launch("kernel_reset_batches", {}, launch_settings);
            }

            GpuVersionUI::update();
        }
    }
}


void GpuVersion::renderOctree(RenderTarget& target){
    // renderingSettings.debug_lod_to_render = CuRastSettings::debugLodToRender;
    renderingSettings.debug_lod_to_render = -1;
    renderingSettings.draw_every_x_points = (CuRastSettings::debugLodToRender <= 0) ? 1 : uint32_t(CuRastSettings::debugLodToRender);
    renderingSettings.use_voxels_debug_color = CuRastSettings::voxelsDebugColor;
    renderingSettings.min_pixel_span = CuRastSettings::minPixelSpan;
    renderingSettings.voxels_nb_points_per_axis = uint32_t(CuRastSettings::voxelsPointsPerAxis);
    renderingSettings.use_multiscale = CuRastSettings::useMultiScale;

    renderingTarget.view = target.view;
    renderingTarget.proj = target.proj;
    renderingTarget.camera_pos = target.cameraPos;
    renderingTarget.width = target.width;
    renderingTarget.height = target.height;

    uint64_t offset = 0; // in units of uint64_t elements (8 bytes each)
    for(uint32_t level = 0; level < CRenderingSettings::NB_PYRAMID_LEVELS; level++){
        renderingTarget.framebuffers[level] = target.framebuffer + offset;
        renderingTarget.colorbuffers[level] = target.colorbuffer + offset;
        uint32_t w = target.width >> level;
        uint32_t h = target.height >> level;
        offset += w * h;
    }

    // Render nodes
    {
        // uint32_t block_size = min(
        //     OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM,
        //     OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
        // );
        // OptionalLaunchSettings launch_settings = {
        //     .gridsize = OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM,
        //     .blocksize = block_size
        // };
        uint32_t block_size = 256;
        uint32_t grid_size =
            (OocSimLodSettings::DEVICE_ATTRIBUTE_NB_SM * OocSimLodSettings::DEVICE_ATTRIBUTE_MAX_THREADS_PER_SM + block_size - 1)
            / block_size
        ;
        OptionalLaunchSettings launch_settings = {
            .gridsize  = grid_size,
            .blocksize = block_size
        };

        // Render bounding boxes
        if(CuRastSettings::showBoundingBoxes){
            prog->launch("kernel_render_bounding_boxes", {&renderingTarget, &renderingSettings}, launch_settings);
        }

        prog->launch("kernel_visibility_pass", {&renderingTarget, &renderingSettings}, launch_settings);
        // prog->launch("kernel_replace_unloaded_nodes", {&renderingTarget, &renderingSettings}, launch_settings);
        prog->launch("kernel_draw_visibility_cache", {&renderingTarget, &renderingSettings}, launch_settings);
        prog->launch("kernel_draw_evicted_nodes", {&renderingTarget, &renderingSettings}, launch_settings);
        prog->launch("kernel_draw_octree_large", {&renderingTarget, &renderingSettings}, launch_settings);
        prog->launch("kernel_draw_octree_small", {&renderingTarget, &renderingSettings}, launch_settings);
    }

    updateHostCache();
    updateVisCache();
}




void GpuVersionUI::init() {
    CURuntime::assertCudaSuccess(cuMemAllocHost((void**)&pinnedStats, sizeof(UIStats)));
    CURuntime::assertCudaSuccess(cuEventCreate(&eventStatsReady, CU_EVENT_DISABLE_TIMING));
    CURuntime::assertCudaSuccess(cuStreamCreate(&uiStream, CU_STREAM_NON_BLOCKING));
}

void GpuVersionUI::destroy() {
    CURuntime::assertCudaSuccess(cuMemFreeHost(pinnedStats));
    CURuntime::assertCudaSuccess(cuEventDestroy(eventStatsReady));
    CURuntime::assertCudaSuccess(cuStreamDestroy(uiStream));
}

void GpuVersionUI::kickoffStatsCopy(){
    // Single batched async copy of all UI fields into pinned memory
    // The fields must be contiguous in CGlobalVariables for this to work
    const uint64_t pad = reinterpret_cast<uintptr_t>(&GpuVersion::hostStaging.nbTotalUpdates)
                       - reinterpret_cast<uintptr_t>(&GpuVersion::hostStaging);
    CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
        pinnedStats,
        GpuVersion::deviceStaging + pad,
        sizeof(UIStats),
        uiStream
    ));

    CURuntime::assertCudaSuccess(cuEventRecord(eventStatsReady, uiStream));
    statsCopyInFlight = true;
}

void GpuVersionUI::update() {
    if(statsCopyInFlight){
        CUresult res = cuEventQuery(eventStatsReady);
        if(res == CUDA_ERROR_NOT_READY){ 
            return;
        }
        CURuntime::assertCudaSuccess(res);
        statsCopyInFlight = false;

        if(lastNbTotalUpdates == pinnedStats->nbTotalUpdates){ 
            kickoffStatsCopy();
            return;
        }

        std::chrono::time_point<std::chrono::high_resolution_clock> now = high_resolution_clock::now();

        lastNbTotalUpdates       = pinnedStats->nbTotalUpdates;
        nbTotalUpdates           = pinnedStats->nbTotalUpdates;
        currentNbNodes           = pinnedStats->curNbNodes;
        currentNbChunks          = pinnedStats->currentNbChunks;
        currentNbGrids           = pinnedStats->currentNbGrids;
        currentNbPoints          = pinnedStats->currentNbPoints;
        currentNbVoxels          = pinnedStats->currentNbVoxels;
        nbTotalPoints            = pinnedStats->nbTotalPoints;
        nbTotalVoxels            = pinnedStats->nbTotalVoxels;
        nbTotalNewNodes          = pinnedStats->nbTotalNewNodes;
        nbTotalNewGrids          = pinnedStats->nbTotalNewGrids;
        nbTotalNewChunks         = pinnedStats->nbTotalNewChunks;
        nbTotalDeletedNodes      = pinnedStats->nbTotalDeletedNodes;
        nbTotalDeletedGrids      = pinnedStats->nbTotalDeletedGrids;
        nbTotalDeletedChunks     = pinnedStats->nbTotalDeletedChunks;
        nbTotalLoadedNodes       = pinnedStats->nbTotalLoadedNodes;
        nbTotalSplitNodes        = pinnedStats->nbTotalSplitNodes;
        nbTotalStoredNodes       = pinnedStats->nbTotalStoredNodes;
        nbNewPointsThisUpdate    = pinnedStats->nbNewPointsThisUpdate;
        nbNewVoxelsThisUpdate    = pinnedStats->nbNewVoxelsThisUpdate;
        nbNewNodesThisUpdate     = pinnedStats->nbNewNodesThisUpdate;
        nbLoadedNodesThisUpdate  = pinnedStats->nbLoadedNodesThisUpdate;
        nbStoredNodesThisUpdate  = pinnedStats->nbStoredNodesThisUpdate;
        nbSplitNodesThisUpdate   = pinnedStats->nbSplitNodesThisUpdate;
        nbDeletedNodesThisUpdate = pinnedStats->nbDeletedNodesThisUpdate;
        nbDeletedChunksThisUpdate= pinnedStats->nbDeletedChunksThisUpdate;
        nbDeletedGridsThisUpdate = pinnedStats->nbDeletedGridsThisUpdate;
        nbNewChunksThisUpdate    = pinnedStats->nbNewChunksThisUpdate;
        nbNewGridsThisUpdate     = pinnedStats->nbNewGridsThisUpdate;

        auto updateStats = [](uint32_t value, uint32_t& minValue, uint32_t& maxValue, uint32_t& avgValue) {
            if(nbTotalUpdates == 1){
                minValue = value; maxValue = value; avgValue = value;
            } else {
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
                avgValue = static_cast<uint32_t>(
                    (static_cast<uint64_t>(avgValue) * (nbTotalUpdates - 1) + value) / nbTotalUpdates
                );
            }
        };

        updateStats(nbNewPointsThisUpdate,    minNbNewPointsPerUpdate,    maxNbNewPointsPerUpdate,    avgNbNewPointsPerUpdate);
        updateStats(nbNewVoxelsThisUpdate,    minNbNewVoxelsPerUpdate,    maxNbNewVoxelsPerUpdate,    avgNbNewVoxelsPerUpdate);
        updateStats(nbNewNodesThisUpdate,     minNbNewNodesPerUpdate,     maxNbNewNodesPerUpdate,     avgNbNewNodesPerUpdate);
        updateStats(nbLoadedNodesThisUpdate,  minNbLoadedNodesPerUpdate,  maxNbLoadedNodesPerUpdate,  avgNbLoadedNodesPerUpdate);
        updateStats(nbStoredNodesThisUpdate,  minNbStoredNodesPerUpdate,  maxNbStoredNodesPerUpdate,  avgNbStoredNodesPerUpdate);
        updateStats(nbSplitNodesThisUpdate,   minNbSplitNodesPerUpdate,   maxNbSplitNodesPerUpdate,   avgNbSplitNodesPerUpdate);
        updateStats(nbDeletedNodesThisUpdate, minNbDeletedNodesPerUpdate, maxNbDeletedNodesPerUpdate, avgNbDeletedNodesPerUpdate);
        updateStats(nbDeletedChunksThisUpdate,minNbDeletedChunksPerUpdate,maxNbDeletedChunksPerUpdate,avgNbDeletedChunksPerUpdate);
        updateStats(nbDeletedGridsThisUpdate, minNbDeletedGridsPerUpdate, maxNbDeletedGridsPerUpdate, avgNbDeletedGridsPerUpdate);
        updateStats(nbNewChunksThisUpdate,    minNbNewChunksPerUpdate,    maxNbNewChunksPerUpdate,    avgNbNewChunksPerUpdate);
        updateStats(nbNewGridsThisUpdate,     minNbNewGridsPerUpdate,     maxNbNewGridsPerUpdate,     avgNbNewGridsPerUpdate);

        uint64_t total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - firstUpdateStart
        ).count();
        const uint64_t cur_updates_per_seconds = static_cast<uint64_t>(
            1000.0 * static_cast<uint64_t>(nbTotalUpdates) / total_duration
        );
        updateStats(cur_updates_per_seconds, minNbUpdatesPerSecond, maxNbUpdatesPerSecond, avgNbUpdatesPerSecond);
    }

    if(!statsCopyInFlight){
        kickoffStatsCopy();
    }
}



















void GpuVersion::tryToCenterTarget(){
    if(!CuRastSettings::tryToCenterTarget){return;}

    // Get the bounding box of the scene
    CUdeviceptr main_octree = 0;
    CIdAABB main_id = CINVALID_ID;
    CGlobalVariables::Relationship main_relationship = {};
    COPY_FROM_GPU(mainOctree, &main_octree, CUdeviceptr);
    COctreeNode tmp = {};
    const uint64_t pad =
        reinterpret_cast<uintptr_t>(&(tmp.aabb_index)) -
        reinterpret_cast<uintptr_t>(&tmp);
    CUdeviceptr src_device = main_octree + pad;
    CURuntime::assertCudaSuccess(
        cuMemcpyDtoH(&main_id, src_device, sizeof(CIdAABB))
    );
    src_device = reinterpret_cast<CUdeviceptr>(hostStaging.relationshipMap)
        + static_cast<CUdeviceptr>(main_id * sizeof(CGlobalVariables::Relationship))
    ;
    CURuntime::assertCudaSuccess(
        cuMemcpyDtoH(&main_relationship, src_device, sizeof(CGlobalVariables::Relationship))
    );
    CAABB main_aabb = main_relationship.aabb;

    // for(auto& [id, node] : GpuVersion::persistentStoredNodes){
    //     for(uint32_t i = 0; i < node->node.points_counter; i++){
    //         const CPoint& point = node->points[i];
    //         mainAABB.mins.x = min(mainAABB.mins.x, point.position.x);
    //         mainAABB.mins.y = min(mainAABB.mins.y, point.position.y);
    //         mainAABB.mins.z = min(mainAABB.mins.z, point.position.z);
    //         mainAABB.maxs.x = max(mainAABB.maxs.x, point.position.x);
    //         mainAABB.maxs.y = max(mainAABB.maxs.y, point.position.y);
    //         mainAABB.maxs.z = max(mainAABB.maxs.z, point.position.z);
    //     }
    // }

    glm::dvec3 center = glm::dvec3((main_aabb.mins + main_aabb.maxs) * 0.5f);
    double diagonal = double(glm::length(main_aabb.maxs - main_aabb.mins));

    Runtime::controls->target = center;
    Runtime::controls->radius = CuRastSettings::tryToCenterTargetDiagonal * diagonal;
    Runtime::controls->update();

    CuRastSettings::tryToCenterTarget = false;
}




/// For another project
void GpuVersion::takeRandomScreenShots(){
    static const uint32_t NB_RESOLUTIONS = 4;
    static const uint32_t NB_SCREENSHOTS = 256;
    static const uint32_t NB_TOTAL_SCREENSHOTS = NB_SCREENSHOTS * NB_RESOLUTIONS * 2;

    // How far the camera sphere's radius can range, relative to the AABB diagonal.
    const double MIN_RADIUS_FACTOR = 0.;
    const double MAX_RADIUS_FACTOR = 1.;
    const double TWO_PI = 6.283185307179586;

    static bool oldSettingsIsLeftDown = false;
    static double oldSettingsScroll = 0;
    static double oldSettingsPosX = 0;
    static double oldSettingsPosY = 0;
    static uint32_t oldSettingsNbPointsPerAxis = 0;
    static vec4 oldSettingsBackgroundColor = {};
    static double oldSettingsYaw = 0;
    static double oldSettingsPitch = 0;
    static double oldSettingsRadius = 0;
    static glm::dvec3 oldSettingsTarget = {};

    static uint32_t screenshotCounter = NB_TOTAL_SCREENSHOTS;
    static bool buttonWasPressed = false;

    // Camera + background shared within one (perturbed, target) pair.
    static double pairYaw = 0;
    static double pairPitch = 0;
    static double pairRadius = 0;
    static vec4 pairBg = {};

    static CAABB mainAABB = {};

    static uint32_t old_width = VKRenderer::width;
    static uint32_t old_height = VKRenderer::height;

    static uint32_t INITIAL_ID = 1024;

    // Begin screenshots
    if(CuRastSettings::bruteForceRendering && !buttonWasPressed){
        buttonWasPressed = true;
        screenshotCounter = 0;

        // Save old settings
        oldSettingsBackgroundColor = CuRastSettings::background;
        oldSettingsIsLeftDown      = Runtime::controls->isLeftDown;
        oldSettingsScroll          = Runtime::mouseEvents.wheel_y;
        oldSettingsPosX            = Runtime::mouseEvents.pos_x;
        oldSettingsPosY            = Runtime::mouseEvents.pos_y;
        oldSettingsYaw             = Runtime::controls->yaw;
        oldSettingsPitch           = Runtime::controls->pitch;
        oldSettingsRadius          = Runtime::controls->radius;
        oldSettingsTarget          = Runtime::controls->target;

        // // Get the bounding box of the scene
        // CUdeviceptr main_octree = 0;
        // CIdAABB main_id = CINVALID_ID;
        // CGlobalVariables::Relationship main_relationship = {};
        // COPY_FROM_GPU(mainOctree, &main_octree, CUdeviceptr);
        // COctreeNode tmp = {};
        // const uint64_t pad =
        //     reinterpret_cast<uintptr_t>(&(tmp.aabb_index)) -
        //     reinterpret_cast<uintptr_t>(&tmp);
        // CUdeviceptr src_device = main_octree + pad;
        // CURuntime::assertCudaSuccess(
        //     cuMemcpyDtoH(&main_id, src_device, sizeof(CIdAABB))
        // );
        // src_device = reinterpret_cast<CUdeviceptr>(hostStaging.relationshipMap)
        //     + static_cast<CUdeviceptr>(main_id * sizeof(CGlobalVariables::Relationship))
        // ;
        // CURuntime::assertCudaSuccess(
        //     cuMemcpyDtoH(&main_relationship, src_device, sizeof(CGlobalVariables::Relationship))
        // );
        // mainAABB = main_relationship.aabb;

        for(auto& [id, node] : GpuVersion::updateNodes){
            for(uint32_t i = 0; i < node->node.points_counter; i++){
                const CPoint& point = node->points[i];
                mainAABB.mins.x = min(mainAABB.mins.x, point.position.x);
                mainAABB.mins.y = min(mainAABB.mins.y, point.position.y);
                mainAABB.mins.z = min(mainAABB.mins.z, point.position.z);
                mainAABB.maxs.x = max(mainAABB.maxs.x, point.position.x);
                mainAABB.maxs.y = max(mainAABB.maxs.y, point.position.y);
                mainAABB.maxs.z = max(mainAABB.maxs.z, point.position.z);
            }
        }

        // println("HOST: .mins = ({}, {}, {}), .maxs = ({}, {}, {})",
        //     mainAABB.mins.x, mainAABB.mins.y, mainAABB.mins.z,
        //     mainAABB.maxs.x, mainAABB.maxs.y, mainAABB.maxs.z
        // );
        // throw(EXIT_FAILURE);

        for(int i = INITIAL_ID; i <= 10'000'000; i++){
			fs::create_directories("./screenshots");
			std::string path = format("./screenshots/id_{}_res_{}x{}_perturbed.png",
                i, VKRenderer::width, VKRenderer::height
            );
            INITIAL_ID = i;
			if(!fs::exists(path)) break;
		}


        GpuVersion::isTakingScreenshots = true;
    }

    // Reset after screenshots are done
    if(buttonWasPressed && screenshotCounter == NB_TOTAL_SCREENSHOTS){
        Runtime::controls->isLeftDown = oldSettingsIsLeftDown;
        Runtime::mouseEvents.wheel_y  = oldSettingsScroll;
        Runtime::mouseEvents.pos_x    = oldSettingsPosX;
        Runtime::mouseEvents.pos_y    = oldSettingsPosY;
        CuRastSettings::background    = oldSettingsBackgroundColor;

        Runtime::controls->yaw    = oldSettingsYaw;
        Runtime::controls->pitch  = oldSettingsPitch;
        Runtime::controls->radius = oldSettingsRadius;
        Runtime::controls->target = oldSettingsTarget;
        Runtime::controls->update();

        buttonWasPressed = false;
        CuRastSettings::bruteForceRendering = false;

        VKRenderer::width = old_width;
        VKRenderer::height = old_height;

        GpuVersion::isTakingScreenshots = false;
    }

    // If screenshots
    if(screenshotCounter < NB_TOTAL_SCREENSHOTS){
        fs::create_directories("./screenshots");

        // We drive yaw/pitch/radius directly now, so no fake dragging needed,
        // and zero out any leftover real scroll so it can't sneak a zoom in
        // on a frame while we wait for the screenshot request to be handled.
        Runtime::controls->isLeftDown = false;
        Runtime::mouseEvents.wheel_x = 0;
        Runtime::mouseEvents.wheel_y = 0;

        vec3 aabbMin = mainAABB.mins;
        vec3 aabbMax = mainAABB.maxs;
        glm::dvec3 center = glm::dvec3((aabbMin + aabbMax) * 0.5f);
        double diagonal = double(glm::length(aabbMax - aabbMin));

        uint32_t pairId = screenshotCounter / (2*NB_RESOLUTIONS) + INITIAL_ID;

        if(screenshotCounter % (2*NB_RESOLUTIONS) == 0){
            // First half of the pair: roll a fresh random sphere radius and a
            // random position on that sphere, plus a random background. This
            // camera gets reused by the matching "target" shot right after it.
            double u1 = double(rand()) / double(RAND_MAX);
            double u2 = double(rand()) / double(RAND_MAX);
            double u3 = double(rand()) / double(RAND_MAX);

            double radiusFactor = MIN_RADIUS_FACTOR + u1 * (MAX_RADIUS_FACTOR - MIN_RADIUS_FACTOR);
            pairRadius = radiusFactor * diagonal;

            pairYaw = u2 * TWO_PI;
            // uniform sampling on the sphere surface (avoids bunching near the poles)
            pairPitch = std::asin(2.0 * u3 - 1.0);

            // pairBg.r = float(double(rand()) / double(RAND_MAX));
            // pairBg.g = float(double(rand()) / double(RAND_MAX));
            // pairBg.b = float(double(rand()) / double(RAND_MAX));
            pairBg.r = 1.0f;
            pairBg.g = 1.0f;
            pairBg.b = 1.0f;
            pairBg.a = 1.0f;

            Runtime::controls->target = center;
            Runtime::controls->radius = pairRadius;
            Runtime::controls->yaw    = pairYaw;
            Runtime::controls->pitch  = pairPitch;
            Runtime::controls->update();

            randomOffset = 1 + rand();

            VKRenderer::width = old_width;
            VKRenderer::height = old_height;
            CuRastSettings::background = pairBg;

            CuRastSettings::requestScreenshot = std::make_shared<string>(
                format("./screenshots/id_{}_res_{}x{}_perturbed.png",
                    pairId,
                    VKRenderer::width, VKRenderer::height
                )
            );

        } else if(screenshotCounter % (2*NB_RESOLUTIONS) == NB_RESOLUTIONS) {
            // Second half of the pair: ground-truth shot, same camera and
            // background as the perturbed shot above — only voxelsPointsPerAxis differs.
            Runtime::controls->target = center;
            Runtime::controls->radius = pairRadius;
            Runtime::controls->yaw    = pairYaw;
            Runtime::controls->pitch  = pairPitch;
            Runtime::controls->update();

            randomOffset = 0;

            VKRenderer::width = old_width;
            VKRenderer::height = old_height;
            CuRastSettings::background = pairBg;

            CuRastSettings::requestScreenshot = std::make_shared<string>(
                format("./screenshots/id_{}_res_{}x{}_target.png",
                    pairId,
                    VKRenderer::width, VKRenderer::height
                )
            );
        } else if(screenshotCounter % (2*NB_RESOLUTIONS) < NB_RESOLUTIONS) {
            VKRenderer::width = (old_width >> (screenshotCounter % (2*NB_RESOLUTIONS)));
            VKRenderer::height = (old_height >> (screenshotCounter % (2*NB_RESOLUTIONS)));
            CuRastSettings::background = pairBg;

            CuRastSettings::requestScreenshot = std::make_shared<string>(
                format("./screenshots/id_{}_res_{}x{}_perturbed.png",
                    pairId,
                    VKRenderer::width, VKRenderer::height
                )
            );
        } else if(screenshotCounter % (2*NB_RESOLUTIONS) > NB_RESOLUTIONS) {
            VKRenderer::width = (old_width >> (screenshotCounter % NB_RESOLUTIONS));
            VKRenderer::height = (old_height >> (screenshotCounter % NB_RESOLUTIONS));
            CuRastSettings::background = pairBg;

            CuRastSettings::requestScreenshot = std::make_shared<string>(
                format("./screenshots/id_{}_res_{}x{}_target.png",
                    pairId,
                    VKRenderer::width, VKRenderer::height
                )
            );
        }

        screenshotCounter++;
    }
}


















/// For another project
void GpuVersion::takeSingleScreenShot(){
    static uint32_t NB_SCREENSHOTS = 4;
    static uint32_t NB_TOTAL_SCREENSHOTS = NB_SCREENSHOTS * 2;

    static bool oldSettingsIsLeftDown = false;
    static double oldSettingsScroll = 0;
    static double oldSettingsPosX = 0;
    static double oldSettingsPosY = 0;
    static double oldSettingsYaw = 0;
    static double oldSettingsPitch = 0;
    static double oldSettingsRadius = 0;
    static glm::dvec3 oldSettingsTarget = {};

    static uint32_t screenshotCounter = NB_TOTAL_SCREENSHOTS;
    static bool buttonWasPressed = false;

    static uint32_t old_width = VKRenderer::width;
    static uint32_t old_height = VKRenderer::height;

    static uint32_t cpt = 0;
    static std::vector<int32_t> density = {};

    static uint32_t INITIAL_ID = 0;
    static int32_t old_lod = -1;

    // Begin screenshots
    if(CuRastSettings::bruteForceRendering && !buttonWasPressed){
        buttonWasPressed = true;
        screenshotCounter = 0;

        oldSettingsIsLeftDown      = Runtime::controls->isLeftDown;
        oldSettingsScroll          = Runtime::mouseEvents.wheel_y;
        oldSettingsPosX            = Runtime::mouseEvents.pos_x;
        oldSettingsPosY            = Runtime::mouseEvents.pos_y;
        oldSettingsYaw             = Runtime::controls->yaw;
        oldSettingsPitch           = Runtime::controls->pitch;
        oldSettingsRadius          = Runtime::controls->radius;
        oldSettingsTarget          = Runtime::controls->target;
        old_lod                    = CuRastSettings::debugLodToRender;

        for(int i = INITIAL_ID; i <= 10'000'000; i++){
            fs::create_directories("./screenshots");
            std::string path = format("./screenshots/id_{}_res_{}x{}_target.png",
                i, VKRenderer::width, VKRenderer::height
            );
            INITIAL_ID = i;
            if(!fs::exists(path)) break;
        }

        NB_SCREENSHOTS = CuRastSettings::lastScreenshotValue - CuRastSettings::firstScreenshotValue + 1;
        NB_TOTAL_SCREENSHOTS = NB_SCREENSHOTS * 2; // just perturbed + target per id
        density.resize(NB_SCREENSHOTS);
        std::iota(density.begin(), density.end(), CuRastSettings::firstScreenshotValue);

        CuRastSettings::useMultiScale = true;
        GpuVersion::isTakingScreenshots = true;
        cpt = 0;
    }

    // Reset after screenshots are done
    if(buttonWasPressed && screenshotCounter == NB_TOTAL_SCREENSHOTS){
        Runtime::controls->isLeftDown = oldSettingsIsLeftDown;
        Runtime::mouseEvents.wheel_y  = oldSettingsScroll;
        Runtime::mouseEvents.pos_x    = oldSettingsPosX;
        Runtime::mouseEvents.pos_y    = oldSettingsPosY;
        CuRastSettings::debugLodToRender = old_lod;

        Runtime::controls->yaw    = oldSettingsYaw;
        Runtime::controls->pitch  = oldSettingsPitch;
        Runtime::controls->radius = oldSettingsRadius;
        Runtime::controls->target = oldSettingsTarget;
        Runtime::controls->update();

        VKRenderer::width  = old_width;
        VKRenderer::height = old_height;

        CuRastSettings::useMultiScale       = false;
        CuRastSettings::bruteForceRendering = false;
        GpuVersion::isTakingScreenshots     = false;
        buttonWasPressed = false;
    }

    if(screenshotCounter < NB_TOTAL_SCREENSHOTS){
        fs::create_directories("./screenshots");

        Runtime::controls->isLeftDown = false;
        Runtime::mouseEvents.wheel_x  = 0;
        Runtime::mouseEvents.wheel_y  = 0;

        Runtime::controls->yaw    = oldSettingsYaw;
        Runtime::controls->pitch  = oldSettingsPitch;
        Runtime::controls->radius = oldSettingsRadius;
        Runtime::controls->target = oldSettingsTarget;
        Runtime::controls->update();

        VKRenderer::width  = old_width;
        VKRenderer::height = old_height;

        uint32_t pairId   = screenshotCounter / 2 + INITIAL_ID;
        bool isPerturbed  = (screenshotCounter % 2 == 0);

        if(isPerturbed){
            CuRastSettings::debugLodToRender = int32_t(1 << density[cpt]);
            cpt++;
            CuRastSettings::requestScreenshot = std::make_shared<string>(
                format("./screenshots/id_{}_res_{}x{}_perturbed.png",
                    pairId, old_width, old_height)
            );
        } else {
            CuRastSettings::debugLodToRender = -1;
            CuRastSettings::requestScreenshot = std::make_shared<string>(
                format("./screenshots/id_{}_res_{}x{}_target.png",
                    pairId, old_width, old_height)
            );
        }

        screenshotCounter++;
    }
}