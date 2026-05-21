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

#include "./utils.cuh"
#include "./HostDeviceInterface.h"

using glm::ivec2;
using glm::i8vec4;
using glm::vec4;

namespace cg = cooperative_groups;


__device__
vec3 worldToNDC(vec3 v, mat4 worldView, float f, float aspect){
	vec4 viewSpace = worldView * vec4(v.x, v.y, v.z, 1.0f);
	float depth = -viewSpace.z;
	float x_ndc = (f / aspect) * viewSpace.x / depth;
	float y_ndc = f * viewSpace.y / depth;

	return vec3{x_ndc, y_ndc, depth};
}


__device__
vec2 ndcToScreen(vec3 ndc, float width, float height){
	return vec2{
		(ndc.x * 0.5f + 0.5f) * width,
		(ndc.y * 0.5f + 0.5f) * height,
	};
}

__device__
void drawLine(RenderTarget target, vec3 start, vec3 end, uint32_t color = 0xff0000ff){

	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int i = block.thread_rank();
	int iterations = 50;
	int max_samples = iterations * block.size();
	for(int j = 0; j < iterations; j++)
	{
		float w = float(i + j * block.size()) / float(max_samples);

		vec3 worldPos = (1.0f - w) * start + w * end;
		
		float f = target.proj[1][1];
		float aspect = float(target.width) / float(target.height);

		vec3 pos_ndc = worldToNDC(worldPos, target.view, f, aspect);
		vec2 pos_screen = ndcToScreen(pos_ndc, target.width, target.height);

		if(pos_ndc.x < -1.0f) continue;
		if(pos_ndc.x >  1.0f) continue;
		if(pos_ndc.y < -1.0f) continue;
		if(pos_ndc.y >  1.0f) continue;
		if(pos_ndc.z <  0.0f) continue;

		int2 pixelCoords = make_int2(pos_screen.x, pos_screen.y);
		int pixelID = pixelCoords.x + pixelCoords.y * target.width;
		pixelID = clamp(pixelID, 0, int(target.width * target.height) - 1);

		float depth = pos_ndc.z;

		if(depth > 0.0f){
			uint64_t udepth = __float_as_uint(depth);
			uint64_t pixel = (udepth << 32) | color;
			atomicMin(&target.colorbuffer[pixelID], pixel);
		}

	}
}

__device__
void drawBoundingBox(RenderTarget target, mat4 world, CAABB aabb, uint32_t color = 0xff0000ff){
    vec3 worldMin = {Infinity, Infinity, Infinity};
    vec3 worldMax = {-Infinity, -Infinity, -Infinity};

    auto sample = [&](vec3 pos){
        vec3 worldPos = world * vec4(pos, 1.0f);
        worldMin.x = min(worldMin.x, worldPos.x);
        worldMin.y = min(worldMin.y, worldPos.y);
        worldMin.z = min(worldMin.z, worldPos.z);
        worldMax.x = max(worldMax.x, worldPos.x);
        worldMax.y = max(worldMax.y, worldPos.y);
        worldMax.z = max(worldMax.z, worldPos.z);
    };

    sample({aabb.mins.x, aabb.mins.y, aabb.mins.z});
    sample({aabb.mins.x, aabb.mins.y, aabb.maxs.z});
    sample({aabb.mins.x, aabb.maxs.y, aabb.mins.z});
    sample({aabb.mins.x, aabb.maxs.y, aabb.maxs.z});
    sample({aabb.maxs.x, aabb.mins.y, aabb.mins.z});
    sample({aabb.maxs.x, aabb.mins.y, aabb.maxs.z});
    sample({aabb.maxs.x, aabb.maxs.y, aabb.mins.z});
    sample({aabb.maxs.x, aabb.maxs.y, aabb.maxs.z});

    // BOTTOM
    drawLine(target, {worldMin.x, worldMin.y, worldMin.z}, {worldMax.x, worldMin.y, worldMin.z}, color);
    drawLine(target, {worldMin.x, worldMax.y, worldMin.z}, {worldMax.x, worldMax.y, worldMin.z}, color);
    drawLine(target, {worldMin.x, worldMin.y, worldMin.z}, {worldMin.x, worldMax.y, worldMin.z}, color);
    drawLine(target, {worldMax.x, worldMin.y, worldMin.z}, {worldMax.x, worldMax.y, worldMin.z}, color);
    // BOTTOM to TOP
    drawLine(target, {worldMin.x, worldMin.y, worldMin.z}, {worldMin.x, worldMin.y, worldMax.z}, color);
    drawLine(target, {worldMin.x, worldMax.y, worldMin.z}, {worldMin.x, worldMax.y, worldMax.z}, color);
    drawLine(target, {worldMax.x, worldMin.y, worldMin.z}, {worldMax.x, worldMin.y, worldMax.z}, color);
    drawLine(target, {worldMax.x, worldMax.y, worldMin.z}, {worldMax.x, worldMax.y, worldMax.z}, color);
    // TOP
    drawLine(target, {worldMin.x, worldMin.y, worldMax.z}, {worldMax.x, worldMin.y, worldMax.z}, color);
    drawLine(target, {worldMin.x, worldMax.y, worldMax.z}, {worldMax.x, worldMax.y, worldMax.z}, color);
    drawLine(target, {worldMin.x, worldMin.y, worldMax.z}, {worldMin.x, worldMax.y, worldMax.z}, color);
    drawLine(target, {worldMax.x, worldMin.y, worldMax.z}, {worldMax.x, worldMax.y, worldMax.z}, color);
}

uint32_t linearGradient(float factor, uint32_t left_color = 0x00ff00ff, uint32_t right_color = 0xff0000ff){
    // Extract channels
    uint8_t r1 = (left_color >> 24) & 0xFF;
    uint8_t g1 = (left_color >> 16) & 0xFF;
    uint8_t b1 = (left_color >> 8)  & 0xFF;
    uint8_t a1 = left_color  & 0xFF;

    uint8_t r2 = (right_color >> 24) & 0xFF;
    uint8_t g2 = (right_color >> 16) & 0xFF;
    uint8_t b2 = (right_color >> 8)  & 0xFF;
    uint8_t a2 = right_color  & 0xFF;

    // Linear interpolation
    uint8_t r = uint8_t(factor * r2 + (1.f - factor) * r1);
    uint8_t g = uint8_t(factor * g2 + (1.f - factor) * g1);
    uint8_t b = uint8_t(factor * b2 + (1.f - factor) * b1);
    uint8_t a = uint8_t(factor * a2 + (1.f - factor) * a1);

    // Repack
    uint32_t color =
        (uint32_t(r) << 24) |
        (uint32_t(g) << 16) |
        (uint32_t(b) << 8)  |
        uint32_t(a);
    
    return color;
}

extern "C" __global__
void kernel_drawOctree(
	CFullOctree octree,
	RenderTarget target
){
	auto grid = cg::this_grid();

	uint32_t index = grid.thread_rank();

	if(index >= octree.num_nodes) return;

    COctreeNode* node = octree.nodes[index];
    CAABB* aabb = octree.aabbs[index];

    // if(index == 0){
    //     printf("\n\n\n\n//////////////////////////////////////////////////\n");
    //     printf("//////////////////////////////////////////////////\n");
    //     printf("//////////////////////////////////////////////////\n");
    //     printf("aabb: mins = (%f, %f, %f), maxs = (%f, %f, %f)\n",
    //         aabb->mins.x, aabb->mins.y, aabb->mins.z,
    //         aabb->maxs.x, aabb->maxs.y, aabb->maxs.z
    //     );
    //     printf("//////////////////////////////////////////////////\n");
    //     printf("//////////////////////////////////////////////////\n");
    //     printf("//////////////////////////////////////////////////\n");
    // }

    float factor = float(node->level) / float(octree.max_lod_level);
    factor = clamp(factor, 0.0f, 1.0f);
    uint32_t min_level_color = 0xff00ff00; // green
    uint32_t max_level_color = 0xff0000ff; // red
    uint32_t color = linearGradient(factor, min_level_color, max_level_color);
    
    drawBoundingBox(target, octree.world, *aabb, color);

	// vec3 position = 0.5f * (aabbs[index].mins + aabbs[index].maxs);
    // uint32_t color = 0xFF0000FF; // RED

	// vec4 projected = target.proj * target.view * octree.world * vec4(position, 1.0f);
	// float depth = projected.w;

	// int px = ((projected.x / depth) * 0.5f + 0.5f) * target.width;
	// int py = ((projected.y / depth) * 0.5f + 0.5f) * target.height;
	// int pixelID = px + py * target.width;

	// if(px < 0 || px >= target.width) return;
	// if(py < 0 || py >= target.height) return;
	// if(pixelID < 0 || pixelID >= target.width * target.height) return;

	// uint64_t udepth = __float_as_uint(depth);
	// uint64_t fragment = (udepth << 32) | color;

	// if(fragment < target.colorbuffer[pixelID]){
	// 	atomicMin(&target.colorbuffer[pixelID], fragment);
	// }

}