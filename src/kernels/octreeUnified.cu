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
void drawBoundingBox(RenderTarget target, mat4 world, CAABBUnified* aabb, uint32_t color = 0xff0000ff){
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

    sample({aabb->mins.x, aabb->mins.y, aabb->mins.z});
    sample({aabb->mins.x, aabb->mins.y, aabb->maxs.z});
    sample({aabb->mins.x, aabb->maxs.y, aabb->mins.z});
    sample({aabb->mins.x, aabb->maxs.y, aabb->maxs.z});
    sample({aabb->maxs.x, aabb->mins.y, aabb->mins.z});
    sample({aabb->maxs.x, aabb->mins.y, aabb->maxs.z});
    sample({aabb->maxs.x, aabb->maxs.y, aabb->mins.z});
    sample({aabb->maxs.x, aabb->maxs.y, aabb->maxs.z});

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

__device__
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

__device__
uint32_t getColor(CPointUnified& point){
    return (uint32_t)point.color[0]
        | ((uint32_t)point.color[1] << 8)
        | ((uint32_t)point.color[2] << 16)
        | (0xFFu << 24);
}

__device__
void drawPoint(
	vec3 position,
    uint32_t color,
	RenderTarget target,
    mat4 world
){
	vec4 projected = target.proj * target.view * world * vec4(position, 1.0f);
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

__device__
void drawAllPoints(
	COctreeNodeUnified* node,
	RenderTarget target,
    mat4 world
){
    auto block = cg::this_thread_block();
    CChunkUnified* cur_points = node->points;

    while(cur_points){
        // Assign each thread in block a separate starting point.
        // Advance each thread by block size, e.g. 256, with each iteration.
        // Might make sense to have a multiple of 256 points per CChunkUnified, e.g. 1024
        for(
            uint32_t i = block.thread_rank(); 
            i < cur_points->size; 
            i += block.num_threads()
        ){
            CPointUnified point = cur_points->points[i];
            drawPoint(point.position, getColor(point), target, world);
        }
        
        cur_points = cur_points->next;
    }
}

__device__
void getScreenSpaceSquare(
    vec3 mins, vec3 maxs,
    RenderTarget target, mat4 world,
    float* smin_x, float* smax_x, float* smin_y, float* smax_y,
    float* depth
){
    // compute node boundaries in screen space
    vec4 p000 = {mins.x, mins.y, mins.z, 1.0f};
    vec4 p001 = {mins.x, mins.y, maxs.z, 1.0f};
    vec4 p010 = {mins.x, maxs.y, mins.z, 1.0f};
    vec4 p011 = {mins.x, maxs.y, maxs.z, 1.0f};
    vec4 p100 = {maxs.x, mins.y, mins.z, 1.0f};
    vec4 p101 = {maxs.x, mins.y, maxs.z, 1.0f};
    vec4 p110 = {maxs.x, maxs.y, mins.z, 1.0f};
    vec4 p111 = {maxs.x, maxs.y, maxs.z, 1.0f};

    mat4 transform = target.proj * target.view * world;
    vec4 ndc000 = transform * p000;
    vec4 ndc001 = transform * p001;
    vec4 ndc010 = transform * p010;
    vec4 ndc011 = transform * p011;
    vec4 ndc100 = transform * p100;
    vec4 ndc101 = transform * p101;
    vec4 ndc110 = transform * p110;
    vec4 ndc111 = transform * p111;

    float fwidth = target.width;
    float fheight = target.height;
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

    *smin_x = min8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
    *smin_y = min8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);
    *smax_x = max8(s000.x, s001.x, s010.x, s011.x, s100.x, s101.x, s110.x, s111.x);
    *smax_y = max8(s000.y, s001.y, s010.y, s011.y, s100.y, s101.y, s110.y, s111.y);
    *depth = min8(ndc000.w, ndc001.w, ndc010.w, ndc011.w, ndc100.w, ndc101.w, ndc110.w, ndc111.w);
}

__device__
bool isLargerThanMinSpanning(
    float min_pixel_span,
    CAABBUnified* aabb,
	RenderTarget target,
    mat4 world
){
    float smin_x = 0.;
    float smax_x = 0.;
    float smin_y = 0.;
    float smax_y = 0.;
    float depth = 0.;
    getScreenSpaceSquare(aabb->mins, aabb->maxs, 
        target, world, 
        &smin_x, &smax_x, &smin_y, &smax_y, &depth
    );

    // screen-space size
    float dx = smax_x - smin_x;
    float dy = smax_y - smin_y;

    float threshold = 2. * min_pixel_span;
    return dx > threshold || dy > threshold;
}

__device__
void cubeRasterizer(
    vec3 mins, vec3 maxs,
    uint32_t cube_color,
    RenderTarget target,
    mat4 world
){
	float smin_x = 0.;
    float smax_x = 0.;
    float smin_y = 0.;
    float smax_y = 0.;
    float depth = 0.;
    getScreenSpaceSquare(mins, maxs, 
        target, world, 
        &smin_x, &smax_x, &smin_y, &smax_y, &depth
    );

    for(int px = int(smin_x); px < int(smax_x); px++)
    for(int py = int(smin_y); py < int(smax_y); py++)
    {
        int pixelID = px + py * target.width;
        if(px < 0 || px >= target.width) continue;
        if(py < 0 || py >= target.height) continue;
        if(pixelID < 0 || pixelID >= target.width * target.height) continue;

        uint64_t udepth = __float_as_uint(depth);
        uint64_t fragment = (udepth << 32) | cube_color;

        if(fragment < target.colorbuffer[pixelID]){
            atomicMin(&target.colorbuffer[pixelID], fragment);
        }
    }
}

__device__
void drawVoxel(
	vec3 voxel_position,
    uint32_t voxel_color,
    RenderTarget target,
    mat4 world,
    vec3 voxel_size,
    uint32_t nb_points
){    
    // vec3 mins = voxel_position - 0.5f*voxel_size;
    // vec3 maxs = voxel_position + 0.5f*voxel_size;
    // cubeRasterizer(
    //     mins, maxs, voxel_color,
    //     target, world
    // );

    // Draw the middle point
    // Usually 1 point is enough to represent a voxel from far away
    if(nb_points % 2 == 1){
        drawPoint(voxel_position, voxel_color, target, world);
    }
    if(nb_points <= 1){
        return;
    }

    float step = 1. / float(nb_points);
    for(float cx = -0.5; cx <= 0.5; cx+=step)
    for(float cy = -0.5; cy <= 0.5; cy+=step)
    for(float cz = -0.5; cz <= 0.5; cz+=step){
        vec3 position = voxel_position + vec3(cx, cy, cz)*voxel_size;
        drawPoint(position, voxel_color, target, world);
    }
}

__device__
void drawAllVoxels(
    COctreeNodeUnified* node,
	RenderTarget target,
    mat4 world,
    uint32_t nb_points,
    uint32_t max_lod_level,
    bool use_debug_color
){
    auto block = cg::this_thread_block();
    CChunkUnified* cur_voxels = node->voxels;
    CAABBUnified* aabb = node->aabb;
    vec3 voxel_size = (aabb->maxs - aabb->mins) / float(C_GRID_SIZE);

    float color_factor = float(node->level) / float(max(max_lod_level, 1));
    color_factor = clamp(color_factor, 0.0f, 1.0f);
    uint32_t min_level_color = 0xffffff00; // cyan
    uint32_t max_level_color = 0xff00ffff; // yellow
    uint32_t color = linearGradient(color_factor, min_level_color, max_level_color);

    while(cur_voxels){
        for(
            uint32_t i = block.thread_rank(); 
            i < cur_voxels->size; 
            i += block.num_threads()
        ){
            CPointUnified voxel = cur_voxels->points[i];
            uint32_t voxel_color = use_debug_color ? color : getColor(voxel);
            drawVoxel(voxel.position, voxel_color, target, world,
                voxel_size, nb_points
            );
        }
        
        cur_voxels = cur_voxels->next;
    }
}


extern "C" __global__
void kernel_drawOctreeAABB(
	CFullOctreeUnified octree,
	RenderTarget target
){
	auto grid = cg::this_grid();

	uint32_t index = grid.thread_rank();

	if(index >= octree.num_nodes) return;

    COctreeNodeUnified* node = octree.nodes[index];

    float factor = float(node->level) / float(max(octree.max_lod_level, 1));
    factor = clamp(factor, 0.0f, 1.0f);
    uint32_t min_level_color = 0xff00ff00; // green
    uint32_t max_level_color = 0xff0000ff; // red
    uint32_t color = linearGradient(factor, min_level_color, max_level_color);
    
    drawBoundingBox(target, octree.world, node->aabb, color);

}


extern "C" __global__
void kernel_visibilityPass(
	CFullOctreeUnified octree,
	RenderTarget target
){
	auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    
    // Assign each node to one thread block
    uint32_t node_index = grid.block_rank();
    if(node_index >= octree.num_nodes) return;
    
    COctreeNodeUnified* node = octree.nodes[node_index];

    bool intersects_frustum = true; // TODO: add frustum culling
    node->is_visible = intersects_frustum;
    
    if(octree.debug_lod_to_render == -1){
        node->is_large = isLargerThanMinSpanning(octree.min_pixel_span, node->aabb, target, octree.world);
        node->is_cut = false;
    }
}


extern "C" __global__
void kernel_drawOctreeLarge(
	CFullOctreeUnified octree,
	RenderTarget target
){
	auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    
    // Assign each node to one thread block
    uint32_t node_index = grid.block_rank();
    if(node_index >= octree.num_nodes) return;
    
    COctreeNodeUnified* node = octree.nodes[node_index];

    if(octree.debug_lod_to_render == -1){
        if(!node->is_large){return;}

        bool has_points = node->points && node->points->size > 0;
        if(has_points && node->is_visible){
            drawAllPoints(node, target, octree.world);
        }
        node->is_visible = false;

        for(uint32_t i=0; i<8; i++){
            COctreeNodeUnified* child = node->children[i];
            if(!child){continue;}
            if(child->is_large){continue;}
            if(!child->is_visible){continue;}
            child->is_cut = true;
        }
    }
}

extern "C" __global__
void kernel_drawOctreeSmall(
	CFullOctreeUnified octree,
	RenderTarget target
){
	auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    
    // Assign each node to one thread block
    uint32_t node_index = grid.block_rank();
    if(node_index >= octree.num_nodes) return;
    
    COctreeNodeUnified* node = octree.nodes[node_index];

    if(octree.debug_lod_to_render != -1){
        if(node->level == octree.debug_lod_to_render){
            drawAllVoxels(
                node, target, 
                octree.world, octree.voxels_nb_points_per_axis,
                octree.max_lod_level, octree.use_voxels_debug_color
            );
            drawAllPoints(node, target, octree.world);
        }
    } else {
        bool should_draw = !node->is_large && node->is_visible && node->is_cut;
        bool is_minimal_draw = (node->level == 0) && !node->is_large;
        if(should_draw || is_minimal_draw){
            drawAllPoints(node, target, octree.world);
            drawAllVoxels(
                node, target, 
                octree.world, octree.voxels_nb_points_per_axis,
                octree.max_lod_level, octree.use_voxels_debug_color
            );
        }
    }

    node->is_visible = false;
    node->is_large = false;
}