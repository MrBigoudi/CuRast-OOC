#include "utils.cuh"

extern "C" __global__
void kernel_test(){
    for(uint32_t i=0; i<globalVariables.maxNbBatches; i++){
        if(globalVariables.batchesAddedMask[i]){continue;}
        printf("DEVICE: batchesAddedMask[%d]: count = %d, first point = (%f, %f, %f)\n", 
            i, 
            globalVariables.batchesToAddCounts[i],
            globalVariables.batchesToAddPoints[i][0].position.x, 
            globalVariables.batchesToAddPoints[i][0].position.y, 
            globalVariables.batchesToAddPoints[i][0].position.z 
        );
        globalVariables.batchesAddedMask[i] = true;
    }
}