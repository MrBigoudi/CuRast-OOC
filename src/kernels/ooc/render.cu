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
void drawLine(const CRenderTarget& target, vec3 start, vec3 end, uint32_t color = 0xff0000ff){

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
void drawBoundingBox(const CRenderTarget& target, const CAABB& aabb, uint32_t color = 0xff0000ff){
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

__device__
void drawPoint(
	const CRenderTarget& target,
	vec3 position,
    uint32_t color,
    uint8_t lod = 0
){
	vec4 projected = target.proj * target.view * vec4(position, 1.0f);
	float depth = projected.w;

	int px = ((projected.x / depth) * 0.5f + 0.5f) * target.width;
	int py = ((projected.y / depth) * 0.5f + 0.5f) * target.height;
	int pixelID = px + py * target.width;

	if(px < 0 || px >= target.width) return;
	if(py < 0 || py >= target.height) return;
	if(pixelID < 0 || pixelID >= target.width * target.height) return;

	uint64_t udepth = __float_as_uint(depth);
	uint64_t fragment = (udepth << 32) | color;

	uint64_t lod_fragment = lod;

	if(fragment < target.colorbuffer[pixelID]){
		atomicMin(&target.colorbuffer[pixelID], fragment);
		atomicMin(&target.framebuffer[pixelID], lod_fragment);
	}
}

__device__
void drawVoxel(
    const CRenderTarget& target,
	vec3 voxel_position,
    uint32_t voxel_color,
    vec3 voxel_size,
    uint32_t nb_points_per_axis,
    uint8_t node_level = 0
){
    // Draw the middle point
    // Usually 1 point is enough to represent a voxel from far away
    if(nb_points_per_axis % 2 == 1){
        drawPoint(target, voxel_position, voxel_color);
    }
    if(nb_points_per_axis <= 1){
        return;
    }

    float step = 1. / float(nb_points_per_axis);

    // Left-Right
    for(float cy = -0.5; cy <= 0.5; cy+=step)
    for(float cz = -0.5; cz <= 0.5; cz+=step){
        vec3 position = voxel_position + vec3(-0.5, cy, cz)*voxel_size;
        drawPoint(target, position, voxel_color, node_level);
        position = voxel_position + vec3(0.5, cy, cz)*voxel_size;
        drawPoint(target, position, voxel_color, node_level);
    }
    // Top-Down
    for(float cx = -0.5+step; cx <= 0.5-step; cx+=step)
    for(float cz = -0.5; cz <= 0.5; cz+=step){
        vec3 position = voxel_position + vec3(cx, -0.5, cz)*voxel_size;
        drawPoint(target, position, voxel_color, node_level);
        position = voxel_position + vec3(cx, 0.5, cz)*voxel_size;
        drawPoint(target, position, voxel_color, node_level);
    }
    // Front-Back
    for(float cx = -0.5+step; cx <= 0.5-step; cx+=step)
    for(float cy = -0.5+step; cy <= 0.5-step; cy+=step){
        vec3 position = voxel_position + vec3(cx, cy, -0.5)*voxel_size;
        drawPoint(target, position, voxel_color, node_level);
        position = voxel_position + vec3(cx, cy, 0.5)*voxel_size;
        drawPoint(target, position, voxel_color, node_level);
    }
}

__device__
void drawAllVoxels(
	const CRenderTarget& target,
	CRenderingSettings settings,
    COctreeNode* node,
    uint32_t nb_points_per_axis,
    uint8_t subtrees,
    bool from_missing_nodes
){
    auto block = cg::this_thread_block();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    CChunk* cur_voxels = node->voxels;
    const CAABB& aabb = globalVariables.relationshipMap[node->aabb_index].aabb;
    vec3 voxel_size = (aabb.maxs - aabb.mins) / float(OocSimLodSettings::GRID_SIZE_PER_DIMENSION);

    // uint32_t depth = globalVariables.isTemporarySwitching ? globalVariables.octreeDepth : globalVariables.renderingOctreeDepth;
    uint32_t depth = globalVariables.octreeDepth;

    float color_factor = float(node->level) / float(max(depth, 1));
    color_factor = clamp(color_factor, 0.0f, 1.0f);
    uint32_t min_level_color = 0xffffff00; // cyan
    uint32_t max_level_color = 0xff00ffff; // yellow
    uint32_t color = linearGradient(color_factor, min_level_color, max_level_color);   

    while(cur_voxels){
        for(uint32_t i = thread_id; i < cur_voxels->size; i += nb_threads_per_block){
            const CPoint& voxel = cur_voxels->points[i];
            
            uint32_t index = aabb.getNextChildIndex(voxel.position);
            if(subtrees & (0x01 << index)){
                uint32_t voxel_color = settings.use_voxels_debug_color ? color : voxel.color;
                voxel_color = from_missing_nodes ? 0xff0000ff : voxel_color;
                drawVoxel(target, voxel.position, voxel_color,
                    voxel_size, nb_points_per_axis, node->level + 1
                );
            }
        }
        
        cur_voxels = cur_voxels->next;
    }
}

__device__
void drawAllPoints(
	const CRenderTarget& target,
	COctreeNode* node
){
    auto block = cg::this_thread_block();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    CChunk* cur_points = node->points;

    while(cur_points){
        for(uint32_t i = thread_id; i < cur_points->size; i += nb_threads_per_block){
            const CPoint& point = cur_points->points[i];
            drawPoint(target, point.position, point.color);
        }
        cur_points = cur_points->next;
    }
}


__device__
void getScreenSpaceSquare(
    const CRenderTarget& target, 
    vec3 mins, vec3 maxs,
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

    mat4 transform = target.proj * target.view;
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
void getScreenSpaceSize(const CRenderTarget& target, const CAABB& aabb, float& dx, float& dy){
    float smin_x = 0.;
    float smax_x = 0.;
    float smin_y = 0.;
    float smax_y = 0.;
    float depth = 0.;
    getScreenSpaceSquare(target, aabb.mins, aabb.maxs, 
        &smin_x, &smax_x, &smin_y, &smax_y, &depth
    );

    // screen-space size
    dx = smax_x - smin_x;
    dy = smax_y - smin_y;
}


__device__
bool isLargerThanMinSpanning(
    const CRenderTarget& target,
	CRenderingSettings settings,
    COctreeNode* node
){

    const CAABB& aabb = globalVariables.relationshipMap[node->aabb_index].aabb;

    // Check if Camera is inside the node
    vec3 cam = target.camera_pos;
    bool cam_inside = cam.x > aabb.mins.x && cam.x < aabb.maxs.x
        && cam.y > aabb.mins.y && cam.y < aabb.maxs.y
        && cam.z > aabb.mins.z && cam.z < aabb.maxs.z
    ;
    if(cam_inside){return true;}

    float dx = 0.;
    float dy = 0.;
    getScreenSpaceSize(target, aabb, dx, dy);

    float threshold = 2. * settings.min_pixel_span;
    return dx > threshold || dy > threshold;
}




















/// Run on "NB SMs" blocks of size min("Max threads per SM", "Max block dim")
extern "C" __global__
void kernel_render_bounding_boxes(
	CRenderTarget target,
    CRenderingSettings settings
){
	auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    uint32_t nb_nodes = globalVariables.curNbNodes;
    uint32_t depth = globalVariables.octreeDepth;

    for(uint32_t node_index = thread_id; node_index < nb_nodes; node_index += nb_threads){
        COctreeNode* node = globalVariables.packedNodes[node_index];

        const CAABB& aabb = globalVariables.relationshipMap[node->aabb_index].aabb;
        if(settings.debug_lod_to_render != -1){
            if(settings.debug_lod_to_render != node->level
                || !globalVariables.isInUpdatesCache(node->aabb_index)
            ){return;}
        }

        float factor = float(node->level) / float(max(depth, 1));
        factor = clamp(factor, 0.0f, 1.0f);
        uint32_t min_level_color = 0xff00ff00; // green
        uint32_t max_level_color = 0xff0000ff; // red

        uint32_t color = linearGradient(factor, min_level_color, max_level_color);
        
        drawBoundingBox(target, aabb, color);
    }
}





/// Run on "NB SMs" blocks of size min("Max threads per SM", "Max block dim")
extern "C" __global__
void kernel_visibilityPass(
	CRenderTarget target,
    CRenderingSettings settings
){
	auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){
        COctreeNode* node = globalVariables.packedNodes[node_index];
        globalVariables.setFlagSync(node->aabb_index, CFlagIsVisible);

        if(settings.debug_lod_to_render != -1){
            continue;
        }

        if(isLargerThanMinSpanning(target, settings, node)){
            globalVariables.setFlagSync(node->aabb_index, CFlagIsLarge);
        }
    }

    // Also flag the nodes from the visibility cache
    for(uint32_t node_index = thread_id; node_index < globalVariables.visibilityCacheCurrentSize; node_index += nb_threads){
        const CIdAABB& id = globalVariables.visibilityCache[node_index];
        globalVariables.setFlagSync(id, CFlagIsInVisibilityCache);
    }
    // for(uint32_t voxel_id = thread_id; voxel_id < globalVariables.nbRenderedVoxels; voxel_id += nb_threads){
    //     const CPoint& voxel = globalVariables.renderedVoxels[voxel_id];
    //     const CIdAABB& node_id = globalVariables.renderedVoxelsNodes[voxel_id];
    //     globalVariables.setFlag(node_id, CFlagIsFromVoxelsInVisibilityCache);
    // }
}



/// Run on "NB SMs" blocks of size min("Max threads per SM", "Max block dim")
extern "C" __global__
void kernel_drawVisibilityCache(
	CRenderTarget target,
    CRenderingSettings settings
){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    // Render points
    for(uint32_t point_id = thread_id; point_id < globalVariables.nbRenderedPoints; point_id += nb_threads){
        const CPoint& point = globalVariables.renderedPoints[point_id];
        drawPoint(
            target, point.position, 
            settings.use_voxels_debug_color ? 0xff00ffff : point.color
        );
    }

    // Render voxels
    for(uint32_t voxel_id = thread_id; voxel_id < globalVariables.nbRenderedVoxels; voxel_id += nb_threads){
        const CPoint& voxel = globalVariables.renderedVoxels[voxel_id];
        const CIdAABB& node_id = globalVariables.renderedVoxelsNodes[voxel_id];

        const CAABB& node_aabb = globalVariables.relationshipMap[node_id].aabb;
        const CNodePosition next_child_pos = node_aabb.getNextChildIndex(voxel.position);
        const CIdAABB& child_index = globalVariables.relationshipMap[node_id].children[next_child_pos];
        const vec3& voxel_size = globalVariables.relationshipMap[node_id].aabb.getSize() / float(OocSimLodSettings::GRID_SIZE_PER_DIMENSION);
        if(child_index == CINVALID_ID){continue;}

        // Only render the voxel if the corresponding child is not present
        if(globalVariables.isInVisibilityCache(child_index)){continue;}
        bool child_is_visible = globalVariables.isVisible(child_index);
        bool child_has_enough_points = globalVariables.hasEnoughPoints(child_index);
        // bool child_has_enough_voxels = globalVariables.hasEnoughVoxels(child_index);
        
        if(!child_is_visible || !child_has_enough_points){
        // if(!child_is_visible){
            const vec3 root_aabb_size = globalVariables.relationshipMap[globalVariables.mainOctree->aabb_index].aabb.getSize();
            float root_size = max(root_aabb_size.x, max(root_aabb_size.y, root_aabb_size.z));
            float cur_size = max(voxel_size.x, max(voxel_size.y, voxel_size.z));
            uint32_t nb_points_per_axis = clamp(uint32_t(root_size / cur_size), 1u, 8u);

            drawVoxel(
                target, 
                voxel.position, 
                settings.use_voxels_debug_color ? 0xffff00ff : voxel.color, 
                voxel_size, 
                nb_points_per_axis
            );

        }
    }
}






/// Run on "NB SMs" blocks of size min("Max threads per SM", "Max block dim")
extern "C" __global__
void kernel_drawOctreeLarge(
	CRenderTarget target,
    CRenderingSettings settings
){
    if(settings.debug_lod_to_render != -1){return;}
	auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    uint32_t nb_nodes = globalVariables.curNbNodes;
    uint32_t depth = globalVariables.octreeDepth;

    // Assign each node to one thread block
    for(uint32_t node_index = block_id; node_index < nb_nodes; node_index += nb_blocks){
        COctreeNode* node = globalVariables.packedNodes[node_index];

        // Render unloaded nodes
        {
            uint32_t flags = 0b00000000;
            for(uint32_t i=0; i<8; i++){
                CIdAABB child_index = globalVariables.relationshipMap[node->aabb_index].children[i];
                if(child_index == CINVALID_ID){continue;}

                bool child_is_in_vis_cache = globalVariables.isInVisibilityCache(child_index);
                if(child_is_in_vis_cache){
                    flags |= ((0x01) << i);
                    continue;
                }
                if(!node->children[i]){continue;}
                
                bool child_has_enough_points = globalVariables.hasEnoughPoints(child_index);
                // bool child_has_enough_points = true;
                // bool child_has_enough_voxels = globalVariables.hasEnoughVoxels(child_index);
                bool child_has_enough_voxels = true;
                if(child_has_enough_points && child_has_enough_voxels){
                    flags |= ((0x01) << i);
                    continue;
                }
            }

            uint32_t nb_points_per_axis = min(8, depth + 1 - node->level);
            drawAllVoxels(
                target, settings, node, nb_points_per_axis,
                flags ^ 0b11111111, 
                // true
                false
            );
        }

        if(!globalVariables.isLarge(node->aabb_index)){continue;}

        drawAllPoints(target, node);

        // Update flags
        if(thread_id == 0){
            for(uint32_t i=0; i<8; i++){
                CIdAABB child_index = globalVariables.relationshipMap[node->aabb_index].children[i];
                if(child_index == CINVALID_ID){continue;}
                if(globalVariables.isLarge(child_index)){continue;}
                if(!globalVariables.isVisible(child_index)){continue;}
                globalVariables.setFlag(child_index, CFlagIsCut);
            }
        }
    }
}










/// Run on "NB SMs" blocks of size min("Max threads per SM", "Max block dim")
extern "C" __global__
void kernel_drawOctreeSmall(
	CRenderTarget target,
    CRenderingSettings settings
){
	auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    uint32_t nb_nodes = globalVariables.curNbNodes;

    // Assign each node to one thread block
    for(uint32_t node_index = block_id; node_index < nb_nodes; node_index += nb_blocks){
        COctreeNode* node = globalVariables.packedNodes[node_index];

        if(!globalVariables.isVisible(node->aabb_index)){
            continue;
        }

        if(settings.debug_lod_to_render != -1){
            if(node->level == settings.debug_lod_to_render){
                drawAllVoxels(
                    target, settings, node,
                    settings.voxels_nb_points_per_axis,
                    0b11111111, false
                );
                drawAllPoints(target, node);
            }
        } else {
            bool is_minimal_draw = (node->level == 0) && !globalVariables.isLarge(node->aabb_index);
            if(globalVariables.isCut(node->aabb_index) || is_minimal_draw){
                drawAllPoints(target, node);
                drawAllVoxels(
                    target, settings, node,
                    settings.voxels_nb_points_per_axis,
                    0b11111111, 
                    false
                    // true
                );
            }
        }

        __syncthreads();
        if(thread_id == 0){
            globalVariables.unsetFlagSync(node->aabb_index, CFlagIsVisible);
            globalVariables.unsetFlagSync(node->aabb_index, CFlagIsLarge);
            globalVariables.unsetFlagSync(node->aabb_index, CFlagIsCut);
        }
    }

    // Also unflag the nodes from the cache
    uint32_t first_point = block_id * nb_threads_per_block + thread_id;
    uint32_t step = nb_blocks * nb_threads_per_block;
    for(uint32_t node_index = first_point; node_index < globalVariables.visibilityCacheCurrentSize; node_index += step){
        const CIdAABB& id = globalVariables.visibilityCache[node_index];
        globalVariables.unsetFlagSync(id, CFlagIsInVisibilityCache);
        globalVariables.unsetFlagSync(id, CFlagIsFromVoxelsInVisibilityCache);
    }
}










/// Run on "NB SMs" blocks of size min("Max threads per SM", "Max block dim")
extern "C" __global__
void kernel_test_multi_resolution(
	CRenderTarget target,
    CRenderingSettings settings, 
    uint32_t random_offset
){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    uint32_t nb_blocks = grid.num_blocks();

    uint32_t block_id = grid.block_rank();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    uint32_t nb_nodes = globalVariables.curNbNodes;

    for(uint32_t node_index = 0; node_index < nb_nodes; node_index++){
        COctreeNode* node = globalVariables.packedNodes[node_index];

        CChunk* cur_points = node->points;

        uint32_t offset = (random_offset == 0) ? 1 : (1 + random_offset + node_index) % 128;

        while(cur_points){
            for(uint32_t i = thread_id; i < cur_points->size; i += nb_threads_per_block){
                // if(i % settings.voxels_nb_points_per_axis != 0){continue;}
                if(i % offset != 0){continue;}

                const CPoint& point = cur_points->points[i];
                drawPoint(target, point.position, point.color, uint8_t(offset));
            }
            cur_points = cur_points->next;
        }

        if(thread_id == 0){
            globalVariables.unsetFlag(node->aabb_index, CFlagIsVisible);
        }
    }
}