#define CUB_DISABLE_BF16_SUPPORT

// === required by GLM ===
#define GLM_FORCE_CUDA
#define GLM_FORCE_NO_CTOR_INIT
#define CUDA_VERSION 12000
namespace std {
	using size_t = ::size_t;
};
// =======================

// #include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "./glm/glm/glm.hpp"
#include "./glm/glm/gtc/matrix_transform.hpp"
#include "./glm/glm/gtc/matrix_access.hpp"
#include "./glm/glm/gtx/transform.hpp"
#include "./glm/glm/gtc/quaternion.hpp"

#include "./GpuVersionInterface.h"

using glm::ivec2;
using glm::i8vec4;
using glm::vec4;

namespace cg = cooperative_groups;



extern "C" __global__
void kernel_test(){
    printf("FROM TEST\n");
    printf("    - Nb AABBs: %d / %d\n", globalVariables.nbAABBs, globalVariables.maxNbAABBs);
    printf("    - Nb nodes to load: %d / %d\n", globalVariables.nbNodesToLoad, globalVariables.maxNbNodesToLoad);
    printf("    - Nb nodes to store: %d / %d\n", globalVariables.nbNodesToStore, globalVariables.maxNbNodesToStore);
    printf("    - Nb spilled points: %d / %d\n", globalVariables.nbSpilledPoints, globalVariables.maxNbSpilledPoints);
    printf("    - Nb backlog voxels: %d / %d\n", globalVariables.nbBacklogVoxels, globalVariables.maxNbBacklogVoxels);
}