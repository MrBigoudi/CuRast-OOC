#include "gpuVersion.h"

void GpuVersion::initBuffers(CuRast* editor, CUcontext* context) {
    CGlobalVariables host_staging = {};
    // Unbounded data
    host_staging.maxNbAABBs = OocSimLodSettings::INITIAL_MAX_NB_NODES;
    host_staging.relationshipMap = alloc<CGlobalVariables::Relationship>(host_staging.maxNbAABBs);
    host_staging.allAABBs = alloc<CAABB>(host_staging.maxNbAABBs);
    
    // Exchangeable data
    host_staging.maxNbNodesToLoad = OocSimLodSettings::MAX_NB_NODES_TO_LOAD;
    host_staging.nodesToLoadBuffer = alloc<CIdAABB>(host_staging.maxNbNodesToLoad);

    host_staging.maxNbNodesToStore = OocSimLodSettings::MAX_NB_NODES_TO_STORE;
    host_staging.nodesToStoreBuffer = alloc<COctreeNode*>(host_staging.maxNbNodesToStore);

    host_staging.maxNbBatches = OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE;
    host_staging.batchesAddedMask = alloc<bool>(host_staging.maxNbBatches);
    
    // Lru caches
    // TODO:
    
    // Temporary buffers
    host_staging.maxNbSpilledPoints = OocSimLodSettings::MAX_NB_SPILLING_POINTS;
    host_staging.spilledPoints = alloc<CPoint>(host_staging.maxNbSpilledPoints);
    host_staging.spillingNodes = alloc<COctreeNode*>(host_staging.maxNbSpilledPoints);

    host_staging.maxNbBacklogVoxels = OocSimLodSettings::MAX_NB_BACKLOG_VOXELS;
    host_staging.backlogVoxels = alloc<CPoint>(host_staging.maxNbBacklogVoxels);
    host_staging.backlogVoxelsNodes = alloc<COctreeNode*>(host_staging.maxNbBacklogVoxels);
    
    // Final allocation
    CUdeviceptr global_variables_ptr = prog->getGlobalsPointer("globalVariables");
    if (global_variables_ptr == 0) {
        throw std::runtime_error("globalVariables symbol not found");
    }
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(global_variables_ptr, &host_staging, sizeof(CGlobalVariables)));
}


void GpuVersion::initAllocators(CuRast* editor, CUcontext* context) {
    CMemoryAllocator host_staging = {};

    host_staging.chunksAllocator = allocAllocator<CChunk>(
        OocSimLodSettings::NB_ALLOCATED_CHUNKS, AllocatorId::ChunkAllocator
    );
    host_staging.gridsAllocator = allocAllocator<COccupancyGrid>(
        OocSimLodSettings::NB_ALLOCATED_GRIDS, AllocatorId::OccupancyGridAllocator
    );
    host_staging.nodesAllocator = allocAllocator<COctreeNode>(
        OocSimLodSettings::NB_ALLOCATED_NODES, AllocatorId::OctreeNodeAllocator
    );

    CUdeviceptr global_allocator_ptr = prog->getGlobalsPointer("globalAllocator");
    if (global_allocator_ptr == 0) {
        throw std::runtime_error("globalAllocator symbol not found");
    }
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(global_allocator_ptr, &host_staging, sizeof(CMemoryAllocator)));
}






void GpuVersion::init(CuRast* editor, CUcontext* context) {
    prog = new CudaModularProgram({
        "./src/kernels/ooc/init.cu",
        "./src/kernels/ooc/test.cu",
    });

    initBuffers(editor, context);
    initAllocators(editor, context);

    OptionalLaunchSettings launch_settings = {
        .gridsize = 1,
        .blocksize = 1
    };
    prog->launch("kernel_init", {}, launch_settings);
    // prog->launch("kernel_test", {}, launch_settings);
}

void GpuVersion::destroy(CuRast *editor, CUcontext *context){
    for(CUdeviceptr& ptr : pointers){
        if(ptr){
            CURuntime::assertCudaSuccess(cuMemFree(ptr));
        }
    }
}