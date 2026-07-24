#include "utils.cuh"


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
void drawLine(CRenderTarget target, vec3 start, vec3 end, uint32_t color = 0xff0000ff){

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
void drawBoundingBox(CRenderTarget target, const CAABB& aabb, uint32_t color = 0xff0000ff){
    vec3 worldMin = {INFINITY, INFINITY, INFINITY};
    vec3 worldMax = {-INFINITY, -INFINITY, -INFINITY};

    auto sample = [&](vec3 pos){
        worldMin.x = min(worldMin.x, pos.x);
        worldMin.y = min(worldMin.y, pos.y);
        worldMin.z = min(worldMin.z, pos.z);
        worldMax.x = max(worldMax.x, pos.x);
        worldMax.y = max(worldMax.y, pos.y);
        worldMax.z = max(worldMax.z, pos.z);
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

__device__
uint32_t linearGradient(float factor, uint32_t left_color, uint32_t right_color){
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
void kernel_renderBoundingBoxes(
	CRenderTarget target
){
	uint32_t index = cg::this_grid().thread_rank();
	if(index >= globalVariables.nbAABBs) return;

    COctreeNode* node = globalVariables.nodes[index];
    const CAABB& aabb = getAABB(node->aabb_index);

    float factor = float(node->level) / float(max(globalVariables.mainOctreeMaxLevel, 1));
    factor = clamp(factor, 0.0f, 1.0f);
    uint32_t min_level_color = 0xff00ff00; // green
    uint32_t max_level_color = 0xff0000ff; // red

    uint32_t color = linearGradient(factor, min_level_color, max_level_color);
    
    drawBoundingBox(target, aabb, color);
}