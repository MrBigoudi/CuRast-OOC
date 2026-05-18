#define CUB_DISABLE_BF16_SUPPORT

// === required by GLM ===
#define GLM_FORCE_CUDA
#define GLM_FORCE_NO_CTOR_INIT
#define CUDA_VERSION 12000
namespace std {
	using size_t = ::size_t;
};
// =======================

#include <cooperative_groups.h>

#include "./libs/glm/glm/glm.hpp"
#include "./libs/glm/glm/gtc/matrix_transform.hpp"
#include "./libs/glm/glm/gtc/matrix_access.hpp"
#include "./libs/glm/glm/gtx/transform.hpp"
#include "./libs/glm/glm/gtc/quaternion.hpp"

#include "./HostDeviceInterface.h"

using glm::ivec2;
using glm::i8vec4;
using glm::vec4;
namespace cg = cooperative_groups;

extern "C" __global__
void kernel_drawPointcloud(
	CPointcloud pointcloud,
	RenderTarget target
){
	auto grid = cg::this_grid();

	uint32_t index = grid.thread_rank();

	if(index >= pointcloud.numPoints) return;

	vec3 position = pointcloud.positions[index];
	uint32_t color = pointcloud.colors[index];

	vec4 projected = target.proj * target.view * pointcloud.world * vec4(position, 1.0f);
	float depth = projected.w;

	int px = ((projected.x / depth) * 0.5f + 0.5f) * target.width;
	int py = ((projected.y / depth) * 0.5f + 0.5f) * target.height;
	int pixelID = px + py * target.width;

	if(px < 0 || px >= target.width) return;
	if(py < 0 || py >= target.height) return;
	if(pixelID < 0 || pixelID >= target.width * target.height) return;

	uint64_t udepth = __float_as_uint(depth);
	uint64_t fragment = (udepth << 32) | color;

	if(fragment < target.colorbuffer[pixelID]){
		atomicMin(&target.colorbuffer[pixelID], fragment);
	}

}
