#include "gpuVersion.h"

template<typename T>
T* alloc(uint32_t size){
    CUdeviceptr ptr = 0;
    CURuntime::assertCudaSuccess(cuMemAlloc(&ptr, size * sizeof(T)));
    return reinterpret_cast<T*>(ptr);
}

void GpuVersion::init(CuRast* editor, CUcontext* context) {
    prog = new CudaModularProgram({
        "./src/kernels/ooc/init.cu",
        "./src/kernels/ooc/test.cu",
    });

    CGlobalVariables host_staging = {};

    ///////////////////////////////////////////////////////////////////////
    /////////////////////////// UNBOUNDED DATA ////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    host_staging.maxNbAABBs = OocSimLodSettings::INITIAL_MAX_NB_NODES;
    host_staging.relationshipMap = alloc<CGlobalVariables::Relationship>(host_staging.maxNbAABBs);
    host_staging.allAABBs = alloc<CAABB>(host_staging.maxNbAABBs);


    ///////////////////////////////////////////////////////////////////////
    ////////////////////////// EXCHANGEABLE DATA //////////////////////////
    ///////////////////////////////////////////////////////////////////////
    host_staging.maxNbNodesToLoad = OocSimLodSettings::MAX_NB_NODES_TO_LOAD;
    host_staging.nodesToLoadBuffer = alloc<CIdAABB>(host_staging.maxNbNodesToLoad);

    host_staging.maxNbNodesToStore = OocSimLodSettings::MAX_NB_NODES_TO_STORE;
    host_staging.nodesToStoreBuffer = alloc<COctreeNode*>(host_staging.maxNbNodesToStore);

    host_staging.maxNbBatches = OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE;
    host_staging.batchesAddedMask = alloc<bool>(host_staging.maxNbBatches);



    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////// LRU CACHES //////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    // TODO:



    ///////////////////////////////////////////////////////////////////////
    ////////////////////////// TEMPORARY BUFFERS //////////////////////////
    ///////////////////////////////////////////////////////////////////////
    host_staging.maxNbSpilledPoints = OocSimLodSettings::MAX_NB_SPILLING_POINTS;
    host_staging.spilledPoints = alloc<CPoint>(host_staging.maxNbSpilledPoints);
    host_staging.spillingNodes = alloc<COctreeNode*>(host_staging.maxNbSpilledPoints);

    host_staging.maxNbBacklogVoxels = OocSimLodSettings::MAX_NB_BACKLOG_VOXELS;
    host_staging.backlogVoxels = alloc<CPoint>(host_staging.maxNbBacklogVoxels);
    host_staging.backlogVoxelsNodes = alloc<COctreeNode*>(host_staging.maxNbBacklogVoxels);


    

    ///////////////////////////////////////////////////////////////////////
    ////////////////////////// FINAL ALLOCATION ///////////////////////////
    ///////////////////////////////////////////////////////////////////////
    CUdeviceptr global_variables_ptr = prog->getGlobalsPointer("globalVariables");
    if (global_variables_ptr == 0) {
        throw std::runtime_error("globalVariables symbol not found");
    }
    CURuntime::assertCudaSuccess(cuMemcpyHtoD(global_variables_ptr, &host_staging, sizeof(CGlobalVariables)));


    
    OptionalLaunchSettings launch_settings = {
        .gridsize = 1,
        .blocksize = 1
    };
    prog->launch("kernel_init", {}, launch_settings);
    prog->launch("kernel_test", {}, launch_settings);

}