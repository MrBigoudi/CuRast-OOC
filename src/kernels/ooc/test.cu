#include "utils.cuh"

extern "C" __global__
void kernel_test(){
    if(!globalVariables.isDoneLoading || !globalVariables.isDoneLoading){
        return;
    }
    for(uint32_t i=0; i<globalVariables.maxNbBatches; i++){
        globalVariables.batchesAddedMask[i] = true;
    }
}

extern "C" __global__
void kernel_test_display(PipelineLevel level, bool display_octree){
    if(!globalVariables.isInitialised){return;}
    printf("\n\n\n\n\n");
    switch (level) {
        case LevelInit:
            printf("Init\n\n");
            break;
        case LevelBottomUp:
            printf("Bottom up\n\n");
            break;
        case LevelSimlodLoad:
            printf("Simlod load\n\n");
            break;
        case LevelSimlodSplitCount:
            printf("Simlod split count\n\n");
            break;
        case LevelSimlodVoxelSampling:
            printf("Simlod voxel sampling\n\n");
            break;
        case LevelSimlodInsertion:
            printf("Simlod insertion\n\n");
            break;
        case LevelSimlod:
            printf("Simlod\n\n");
            break;
        case LevelCacheUpdate:
            printf("Cache update\n\n");
            break;
    }
    
    if(display_octree){
        displayOctreeIt(globalVariables.mainOctree);
        printf("\n\n");
    }
}