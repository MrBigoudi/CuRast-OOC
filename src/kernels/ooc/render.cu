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
			atomicMin(&target.colorbuffers[0][pixelID], pixel);
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
    const CRenderingSettings settings,
	vec3 position,
    uint32_t color,
    uint8_t lod = 0
){
	vec4 projected = target.proj * target.view * vec4(position, 1.0f);
    float depth = projected.w;
    if(depth <= 0.0f) return;

    uint64_t udepth   = __float_as_uint(depth);
    uint64_t fragment  = (udepth << 32) | color;
    uint64_t lod_frag  = lod;

    // Project at finest resolution, then scale down per level
    float ndc_x = projected.x / depth;  // [-1, 1]
    float ndc_y = projected.y / depth;

    uint32_t loop_end = settings.use_multiscale ? CRenderingSettings::NB_PYRAMID_LEVELS : 1;
    #pragma unroll
    for(uint32_t level = 0; level < loop_end; level++){
        uint64_t* colorbuffer  = target.colorbuffers[level];
        uint64_t* framebuffer  = target.framebuffers[level];

        uint32_t w = target.width >> level;
        uint32_t h = target.height >> level;

        int px = int((ndc_x * 0.5f + 0.5f) * float(w));
        int py = int((ndc_y * 0.5f + 0.5f) * float(h));

        if(px < 0 || px >= int(w)) continue;
        if(py < 0 || py >= int(h)) continue;

        int pixelID = px + py * w;

        if(fragment < colorbuffer[pixelID]){
            atomicMin(&colorbuffer[pixelID], fragment);
            atomicMin(&framebuffer[pixelID], lod_frag);
        }
    }
}

__device__
void drawVoxel(
    const CRenderTarget& target,
    const CRenderingSettings settings,
	vec3 voxel_position,
    uint32_t voxel_color,
    vec3 voxel_size,
    uint32_t nb_points_per_axis,
    uint8_t node_level = 0
){
    // Draw the middle point
    // Usually 1 point is enough to represent a voxel from far away
    if(nb_points_per_axis % 2 == 1){
        drawPoint(target, settings, voxel_position, voxel_color);
    }
    if(nb_points_per_axis <= 1){
        return;
    }

    float step = 1. / float(nb_points_per_axis);

    // Left-Right
    for(float cy = -0.5; cy <= 0.5; cy+=step)
    for(float cz = -0.5; cz <= 0.5; cz+=step){
        vec3 position = voxel_position + vec3(-0.5, cy, cz)*voxel_size;
        drawPoint(target, settings, position, voxel_color, node_level);
        position = voxel_position + vec3(0.5, cy, cz)*voxel_size;
        drawPoint(target, settings, position, voxel_color, node_level);
    }
    // Top-Down
    for(float cx = -0.5+step; cx <= 0.5-step; cx+=step)
    for(float cz = -0.5; cz <= 0.5; cz+=step){
        vec3 position = voxel_position + vec3(cx, -0.5, cz)*voxel_size;
        drawPoint(target, settings, position, voxel_color, node_level);
        position = voxel_position + vec3(cx, 0.5, cz)*voxel_size;
        drawPoint(target, settings, position, voxel_color, node_level);
    }
    // Front-Back
    for(float cx = -0.5+step; cx <= 0.5-step; cx+=step)
    for(float cy = -0.5+step; cy <= 0.5-step; cy+=step){
        vec3 position = voxel_position + vec3(cx, cy, -0.5)*voxel_size;
        drawPoint(target, settings, position, voxel_color, node_level);
        position = voxel_position + vec3(cx, cy, 0.5)*voxel_size;
        drawPoint(target, settings, position, voxel_color, node_level);
    }
}


__device__
void drawAllPoints(
	const CRenderTarget& target,
    const CRenderingSettings settings,
	COctreeNode* node
){
    auto block = cg::this_thread_block();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    CChunk* cur_points = node->points;

    while(cur_points){
        for(uint32_t i = thread_id; i < cur_points->size; i += nb_threads_per_block){
            const CPoint& point = cur_points->points[i];
            drawPoint(target, settings, point.position, point.color);
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
    const CAABB& aabb
){

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
void kernel_visibility_pass(
	CRenderTarget target,
    CRenderingSettings settings
){
	auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    for(uint32_t node_index = thread_id; node_index < globalVariables.curNbNodes; node_index += nb_threads){
        COctreeNode* node = globalVariables.packedNodes[node_index];

        if(settings.debug_lod_to_render != -1){
            continue;
        }

        const CAABB& aabb = globalVariables.relationshipMap[node->aabb_index].aabb;
        if(isLargerThanMinSpanning(target, settings, aabb)){
            node->flags |= (0x01 << CFlagIsLarge);
        }
    }

    // // Select the active visibility cache
    // CIdAABB* vis_cache = globalVariables.isUsingSecondRenderingBuffer
    //     ? globalVariables.visibilityCache2
    //     : globalVariables.visibilityCache;
    // uint32_t vis_cache_size = globalVariables.isUsingSecondRenderingBuffer
    //     ? globalVariables.visibilityCacheCurrentSize2
    //     : globalVariables.visibilityCacheCurrentSize;

    // for(uint32_t node_index = thread_id; node_index < vis_cache_size; node_index += nb_threads){
    //     const CIdAABB& id = vis_cache[node_index];
    //     globalVariables.setFlag(id, CFlagIsInVisibilityCache);
    // }
}



/// Run on "NB SMs" blocks of size min("Max threads per SM", "Max block dim")
extern "C" __global__
void kernel_replace_unloaded_nodes(
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

        CChunk* cur_voxels = node->voxels;
        const CAABB& aabb = globalVariables.relationshipMap[node->aabb_index].aabb;
        vec3 voxel_size = (aabb.maxs - aabb.mins) / float(OocSimLodSettings::GRID_SIZE_PER_DIMENSION);

        uint32_t depth = globalVariables.octreeDepth;
        float color_factor = float(node->level) / float(max(depth, 1));
        color_factor = clamp(color_factor, 0.0f, 1.0f);
        uint32_t min_level_color = 0xffffff00; // cyan
        uint32_t max_level_color = 0xff00ffff; // yellow
        uint32_t color = linearGradient(color_factor, min_level_color, max_level_color);   

        while(cur_voxels){
            for(uint32_t i = thread_id; i < cur_voxels->size; i += nb_threads_per_block){
                const CPoint& voxel = cur_voxels->points[i];

                // Check if a subchild containing the voxel is already drawn
                CNodePosition index = aabb.getNextChildIndex(voxel.position);
                CIdAABB child_id = globalVariables.relationshipMap[node->aabb_index].children[index];
                COctreeNode* child = node->children[index];
                bool can_render = true;

                while(child_id != CINVALID_ID){
                    // if(globalVariables.isInVisibilityCache(child_id)){
                    //     can_render = false;
                    //     break;
                    // }


                    index = globalVariables.relationshipMap[child_id].aabb.getNextChildIndex(voxel.position);
                    child_id = globalVariables.relationshipMap[child_id].children[index];

                    if(child){
                        bool child_has_enough_points = child->flags & (0x01 << CFlagHasEnoughPoints);
                        bool child_has_enough_voxels = child->flags & (0x01 << CFlagHasEnoughVoxels);
                        if(child_has_enough_points && child_has_enough_voxels){
                            can_render = false;
                            break;
                        }
                        child = child->children[index];
                    }

                }
                if(!can_render){continue;}

                uint32_t voxel_color = settings.use_voxels_debug_color ? color : voxel.color;
                uint32_t nb_points_per_axis = min(8, depth + 1 - node->level);

                drawPoint(target, settings, voxel.position, voxel_color, node->level);
            }
            
            cur_voxels = cur_voxels->next;
        }
    }
}







/// Run on "NB SMs" blocks of size min("Max threads per SM", "Max block dim")
extern "C" __global__
void kernel_draw_visibility_cache(
	CRenderTarget target,
    CRenderingSettings settings
){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    if(settings.debug_lod_to_render != -1){return;}

    // Select active buffer
    CPoint* rendered_points = globalVariables.isUsingSecondRenderingBuffer
        ? globalVariables.renderedPoints2 : globalVariables.renderedPoints;
    uint32_t nb_points = globalVariables.isUsingSecondRenderingBuffer
        ? globalVariables.nbRenderedPoints2 : globalVariables.nbRenderedPoints;
    CPoint* rendered_voxels = globalVariables.isUsingSecondRenderingBuffer
        ? globalVariables.renderedVoxels2 : globalVariables.renderedVoxels;
    uint32_t nb_voxels = globalVariables.isUsingSecondRenderingBuffer
        ? globalVariables.nbRenderedVoxels2 : globalVariables.nbRenderedVoxels;
    // CIdAABB* rendered_voxels_nodes = globalVariables.isUsingSecondRenderingBuffer
    //     ? globalVariables.renderedVoxelsNodes2 : globalVariables.renderedVoxelsNodes;

    // Render points
    for(uint32_t point_id = thread_id; point_id < nb_points; point_id += nb_threads){
        const CPoint& point = rendered_points[point_id];
        drawPoint(target, settings, point.position,
            settings.use_voxels_debug_color ? 0xff00ffff : point.color
        );
    }

    // Render voxels
    for(uint32_t voxel_id = thread_id; voxel_id < nb_voxels; voxel_id += nb_threads){
        const CPoint& voxel = rendered_voxels[voxel_id];
        // const CIdAABB& node_id = rendered_voxels_nodes[voxel_id];

        // const CAABB& node_aabb = globalVariables.relationshipMap[node_id].aabb;
        // const CNodePosition next_child_pos = node_aabb.getNextChildIndex(voxel.position);
        // const CIdAABB& child_index = globalVariables.relationshipMap[node_id].children[next_child_pos];
        // if(child_index == CINVALID_ID){continue;}

        // // Only render the voxel if the corresponding child is not present
        // if((child_index != CINVALID_ID) && globalVariables.isInVisibilityCache(child_index)){
        //     continue;
        // }

        drawPoint(target, settings, voxel.position, settings.use_voxels_debug_color ? 0xffff00ff : voxel.color);
    }
}






/// Run on "NB SMs" blocks of size min("Max threads per SM", "Max block dim")
extern "C" __global__
void kernel_draw_octree_large(
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

        if(!(node->flags & (0x01 << CFlagIsLarge))){continue;}

        drawAllPoints(target, settings, node);

        // Update flags
        if(thread_id == 0){
            for(uint32_t i=0; i<8; i++){
                COctreeNode* child = node->children[i];
                if(!child){continue;}
                if(child->flags & (0x01 << CFlagIsLarge)){continue;}
                child->flags |= (0x01 << CFlagIsCut);
            }
        }
    }
}




__device__
void drawAllVoxels(
	const CRenderTarget& target,
	CRenderingSettings settings,
    COctreeNode* node
){
    auto block = cg::this_thread_block();
    uint32_t thread_id = block.thread_rank();
    uint32_t nb_threads_per_block = block.num_threads();

    CChunk* cur_voxels = node->voxels;

    uint32_t depth = globalVariables.octreeDepth;
    float color_factor = float(node->level) / float(max(depth, 1));
    color_factor = clamp(color_factor, 0.0f, 1.0f);
    uint32_t min_level_color = 0xffffff00; // cyan
    uint32_t max_level_color = 0xff00ffff; // yellow
    uint32_t color = linearGradient(color_factor, min_level_color, max_level_color);   

    while(cur_voxels){
        for(uint32_t i = thread_id; i < cur_voxels->size; i += nb_threads_per_block){
            const CPoint& voxel = cur_voxels->points[i];
            uint32_t voxel_color = settings.use_voxels_debug_color ? color : voxel.color;
            drawPoint(target, settings, voxel.position, voxel_color,
                node->level + 1
            );
        }
        
        cur_voxels = cur_voxels->next;
    }
}






/// Run on "NB SMs" blocks of size min("Max threads per SM", "Max block dim")
extern "C" __global__
void kernel_draw_octree_small(
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

        if(settings.debug_lod_to_render != -1){
            if(node->level <= settings.debug_lod_to_render){
                drawAllVoxels(target, settings, node);
                drawAllPoints(target, settings, node);
            }
        } else {
            bool is_minimal_draw = (node->level == 0) && !(node->flags & (0x01 << CFlagIsLarge));
            if((node->flags & (0x01 << CFlagIsCut)) || is_minimal_draw){
                drawAllVoxels(target, settings, node);
                drawAllPoints(target, settings, node);
            }
        }

        __syncthreads();
        if(thread_id == 0){
            __nv_atomic_and(&node->flags, ~(1u << CFlagIsLarge), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_and(&node->flags, ~(1u << CFlagIsCut), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        }
    }

    // // Also unflag the nodes from the cache
    // CIdAABB* vis_cache = globalVariables.isUsingSecondRenderingBuffer
    //     ? globalVariables.visibilityCache2 : globalVariables.visibilityCache;
    // uint32_t vis_cache_size = globalVariables.isUsingSecondRenderingBuffer
    //     ? globalVariables.visibilityCacheCurrentSize2 : globalVariables.visibilityCacheCurrentSize;

    // uint32_t first_point = block_id * nb_threads_per_block + thread_id;
    // uint32_t step = nb_blocks * nb_threads_per_block;
    // for(uint32_t node_index = first_point; node_index < vis_cache_size; node_index += step){
    //     const CIdAABB& id = vis_cache[node_index];
    //     globalVariables.unsetFlagSync(id, CFlagIsInVisibilityCache);
    // }
}




















__device__
float getEdlShadingFactor(uint64_t* colorbuffer, int width, int height, float depth, int x, int y, int distance){
	auto getNeighborDepth = [&](int x, int y) -> float{

		if(x < 0 || x >= width) return INFINITY;
		if(y < 0 || y >= height) return INFINITY;

		int pixelID = x + width * y;
		uint64_t pixel = colorbuffer[pixelID];

		float d = __uint_as_float(pixel >> 32);

		return d;
	};

	float sum = 0.0f;
	int numSamples = 8;
	for(int i = 0; i < numSamples; i++){
		float u = 2.0f * 3.1415f * float(i) / float(numSamples);
		float dx = float(distance) * cos(u);
		float dy = float(distance) * sin(u);
		
		sum += max(log2f(depth) - log2f(getNeighborDepth(x + dx, y + dy)), 0.0f);
	}

	// float response = sum / 4.0f;
	float response = sum / float(numSamples);
	float edlStrength = 0.9f;
	float shade = exp(-response * 300.0f * edlStrength);
	shade = clamp(shade, 0.3f, 1.0f);

	shade = shade * 0.8f + 0.2f;

	return shade;
}



extern "C" __global__
void kernel_resolve_colorbuffer_to_screenshot(
	CRenderTarget source,
	uint32_t* screenshot,
	bool enableEDL,
	int windowWidth,
	int windowHeight,
	uint32_t backgroundColor
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int x = grid.thread_index().x;
	int y = grid.thread_index().y;
	int pixelID = x + source.width * y;

	if(x >= source.width) return;
	if(y >= source.height) return;

	uint64_t pixel = source.colorbuffers[0][pixelID];
	float depth = __uint_as_float(pixel >> 32);
	uint32_t color = pixel & 0xffffffff;

	float edl = 1.0f;
	float ssao = 1.0f;

	if(enableEDL){
		int supersamplingFactor = source.width / windowWidth;
		edl = getEdlShadingFactor(source.colorbuffers[0], source.width, source.height, depth, x, y, supersamplingFactor);
	}

	if(isinf(depth)) color = backgroundColor;

	float shade = edl * ssao;
	uint8_t* rgba = (uint8_t*)&color;
	rgba[0] = shade * float(rgba[0]);
	rgba[1] = shade * float(rgba[1]);
	rgba[2] = shade * float(rgba[2]);
	rgba[3] = 255;

	// surf2Dwrite(color, gl_desktop, x * 4, y);
	screenshot[pixelID] = color;	
}





extern "C" __global__
void kernel_unpack_resolved_pyramid_to_tensor(
    uint32_t* resolved_level0,
    uint32_t* resolved_level1,
    uint32_t* resolved_level2,
    uint32_t* resolved_level3,
    uint64_t* framebuffer,
    float*    out_tensor,
    uint32_t  base_width,
    uint32_t  base_height,
    uint32_t  nb_levels,
    uint64_t* level_fb_offsets,
    uint64_t* level_tensor_offsets
){
    uint32_t thread_id     = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_threads = blockDim.x * gridDim.x;

    uint32_t* resolved[4] = {
        resolved_level0, resolved_level1,
        resolved_level2, resolved_level3
    };

    for(uint32_t level = 0; level < nb_levels; level++){
        uint32_t w = base_width  >> level;
        uint32_t h = base_height >> level;
        uint64_t levelPixels = uint64_t(w) * h;
        uint64_t tensor_off  = level_tensor_offsets[level];
        uint64_t fb_off      = level_fb_offsets[level];
        uint64_t stride_c    = levelPixels;

        for(uint64_t i = thread_id; i < levelPixels; i += total_threads){
            uint32_t rgba = resolved[level][i];

            // kernel_resolve_colorbuffer_to_screenshot writes via
            // uint8_t* rgba = (uint8_t*)&color, so byte0=R, byte1=G, byte2=B
            // which is exactly what PIL reads from the PNG — matches training
            float r = float( rgba        & 0xff) / 255.0f;
            float g = float((rgba >>  8) & 0xff) / 255.0f;
            float b = float((rgba >> 16) & 0xff) / 255.0f;

            uint64_t fb = framebuffer[fb_off + i];
            float lod   = float(uint8_t(fb >> 56)) / 255.0f;

            out_tensor[tensor_off + 0 * stride_c + i] = r;
            out_tensor[tensor_off + 1 * stride_c + i] = g;
            out_tensor[tensor_off + 2 * stride_c + i] = b;
            out_tensor[tensor_off + 3 * stride_c + i] = lod;
            out_tensor[tensor_off + 4 * stride_c + i] = 0.0f;
        }
    }
}


extern "C" __global__
void kernel_pack_tensor_to_colorbuffer(
    float*    tensor,
    uint64_t* colorbuffer,
    uint32_t  width,
    uint32_t  height
){
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t numPixels = uint64_t(width) * height;
    if(i >= numPixels) return;

    float r_f = __saturatef(tensor[0 * numPixels + i]);
    float g_f = __saturatef(tensor[1 * numPixels + i]);
    float b_f = __saturatef(tensor[2 * numPixels + i]);

    uint8_t r = uint8_t(r_f * 255.0f);
    uint8_t g = uint8_t(g_f * 255.0f);
    uint8_t b = uint8_t(b_f * 255.0f);

    // Pack to match what kernel_resolve_colorbuffer_to_screenshot reads:
    // it does uint8_t* rgba = (uint8_t*)&color; rgba[0]=R, rgba[1]=G, rgba[2]=B
    // So byte0=R -> uint32 bit0..7 = R
    uint32_t color = uint32_t(r)
                   | (uint32_t(g) << 8)
                   | (uint32_t(b) << 16)
                   | (0xffu       << 24);

    // Write depth=1.0 so EDL and resolve treat this as a valid pixel
    uint32_t depth_bits = __float_as_uint(1.0f);
    colorbuffer[i] = (uint64_t(depth_bits) << 32) | uint64_t(color);
}




















__device__
struct CPlane {
    vec3 normal;
    float constant;

    CPlane(){}
    CPlane(float x, float y, float z, float w){
        float normal_length = length(vec3{x, y, z});
        normal = vec3{x, y, z} / normal_length;
        constant = w / normal_length;
    }
};

__device__
struct CFrustum {
    CPlane planes[6] = {};

    CFrustum(const mat4& view_proj){
        const mat4& transpose = view_proj;
        float m_00 = transpose[0][0];
        float m_01 = transpose[0][1];
        float m_02 = transpose[0][2];
        float m_03 = transpose[0][3];
        float m_10 = transpose[1][0];
        float m_11 = transpose[1][1];
        float m_12 = transpose[1][2];
        float m_13 = transpose[1][3];
        float m_20 = transpose[2][0];
        float m_21 = transpose[2][1];
        float m_22 = transpose[2][2];
        float m_23 = transpose[2][3];
        float m_30 = transpose[3][0];
        float m_31 = transpose[3][1];
        float m_32 = transpose[3][2];
        float m_33 = transpose[3][3];

        planes[0] = CPlane(m_03 - m_00, m_13 - m_10, m_23 - m_20, m_33 - m_30);
        planes[1] = CPlane(m_03 + m_00, m_13 + m_10, m_23 + m_20, m_33 + m_30);
        planes[2] = CPlane(m_03 + m_01, m_13 + m_11, m_23 + m_21, m_33 + m_31);
        planes[3] = CPlane(m_03 - m_01, m_13 - m_11, m_23 - m_21, m_33 - m_31);
        planes[4] = CPlane(m_03 - m_02, m_13 - m_12, m_23 - m_22, m_33 - m_32);
        // planes[5] = CPlane(m_03 + m_02, m_13 + m_12, m_23 + m_22, m_33 + m_32);
        planes[5] = CPlane(m_02, m_12, m_22, m_32); // Near (z >= 0, Vulkan)
    }

    /// Checks if a node intersects a frustum
    bool doesIntersect(const CAABB& aabb, const vec3& camera_pos) const {
        if(camera_pos.x >= aabb.mins.x && camera_pos.x <= aabb.maxs.x &&
            camera_pos.y >= aabb.mins.y && camera_pos.y <= aabb.maxs.y &&
            camera_pos.z >= aabb.mins.z && camera_pos.z <= aabb.maxs.z){
            return true;
        }

        for(uint32_t i = 0; i < 6; i++){
            vec3 vector = {
                planes[i].normal.x > 0.0 ? aabb.maxs.x : aabb.mins.x,
                planes[i].normal.y > 0.0 ? aabb.maxs.y : aabb.mins.y,
                planes[i].normal.z > 0.0 ? aabb.maxs.z : aabb.mins.z
            };

            float d = dot(planes[i].normal, vector) + planes[i].constant;
            if(d < 0){return false;}
        }

        return true;
    }
};






extern "C" __global__
void kernel_get_renderable_nodes_part_1_visibility(
	CRenderTarget target,
    CRenderingSettings settings
){
	auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    uint32_t nb_nodes = globalVariables.totalNbNodes;

    globalVariables.totalNbNodesForVisibility = nb_nodes;
    globalVariables.nbNodesExchangedVisPoints = 0;
    globalVariables.nbNodesExchangedVisVoxels = 0;

    CFrustum frustum = CFrustum(target.proj * target.view);

    // Get all visible nodes
    for(uint32_t node_index = thread_id; node_index < nb_nodes; node_index += nb_threads){
        // Unset old flags
        globalVariables.unsetFlag(node_index, CFlagIsVisibleHost);
        globalVariables.unsetFlag(node_index, CFlagIsLargeHost);
        globalVariables.unsetFlag(node_index, CFlagIsCutHost);
        globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize[node_index] = 0.;
        globalVariables.exchangedAABBIndicesVisVoxelsScreenSpaceSize[node_index] = 0.;

        CAABB aabb = globalVariables.relationshipMap[node_index].aabb;
        if(frustum.doesIntersect(aabb, target.camera_pos)){
            globalVariables.setFlag(node_index, CFlagIsVisibleHost);
        }
    }
}

extern "C" __global__
void kernel_get_renderable_nodes_part_2_flagging_large(
	CRenderTarget target,
    CRenderingSettings settings
){
	auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    uint32_t nb_nodes = globalVariables.totalNbNodesForVisibility;

    // Flag all large nodes
    for(uint32_t node_index = thread_id; node_index < nb_nodes; node_index += nb_threads){
        if(!globalVariables.getFlag(node_index, CFlagIsVisibleHost)){continue;}
        CAABB aabb = globalVariables.relationshipMap[node_index].aabb;
        if(isLargerThanMinSpanning(target, settings, aabb)){
            globalVariables.setFlag(node_index, CFlagIsLargeHost);
        }
    }
}


extern "C" __global__
void kernel_get_renderable_nodes_part_3_large_nodes(
	CRenderTarget target,
    CRenderingSettings settings
){
	auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    uint32_t nb_nodes = globalVariables.totalNbNodesForVisibility;

    // Flag nodes just above cut as points loadable
    for(uint32_t node_index = thread_id; node_index < nb_nodes; node_index += nb_threads){
        if(!globalVariables.getFlag(node_index, CFlagIsVisibleHost)){continue;}
        if(!globalVariables.getFlag(node_index, CFlagIsLargeHost)){continue;}

        uint32_t buffer_id = __nv_atomic_fetch_add(&globalVariables.nbNodesExchangedVisPoints, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        globalVariables.exchangedAABBIndicesVisPointsTmp[buffer_id] = node_index;

        // Compute and store screen-space size
        const CAABB& aabb = globalVariables.relationshipMap[node_index].aabb;
        float dx, dy;
        getScreenSpaceSize(target, aabb, dx, dy);
        globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize[buffer_id] = dx * dy;


        CIdAABB* children = globalVariables.relationshipMap[node_index].children;
        CIdAABB children_tmp[8] = {
            children[0], children[1], children[2], children[3],
            children[4], children[5], children[6], children[7]
        };
        for(uint32_t i=0; i<8; i++){
            if(children_tmp[i] == CINVALID_ID){continue;}
            if(globalVariables.getFlag(children_tmp[i], CFlagIsLargeHost)){continue;}
            globalVariables.setFlag(children_tmp[i], CFlagIsCutHost);
        }
    }
}

extern "C" __global__
void kernel_get_renderable_nodes_part_4_small_nodes(
	CRenderTarget target,
    CRenderingSettings settings
){
	auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    uint32_t nb_nodes = globalVariables.totalNbNodesForVisibility;

    // Flag nodes just below cut as voxels and points loadable
    for(uint32_t node_index = thread_id; node_index < nb_nodes; node_index += nb_threads){
        if(globalVariables.getFlag(node_index, CFlagIsVisibleHost)
            && globalVariables.getFlag(node_index, CFlagIsCutHost)
        ){
            // Compute and store screen-space size
            const CAABB& aabb = globalVariables.relationshipMap[node_index].aabb;
            float dx, dy;
            getScreenSpaceSize(target, aabb, dx, dy);
            float screen_space_size = dx * dy;

            uint32_t buffer_id = __nv_atomic_fetch_add(&globalVariables.nbNodesExchangedVisPoints, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            globalVariables.exchangedAABBIndicesVisPointsTmp[buffer_id] = node_index;
            globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize[buffer_id] = screen_space_size;

            uint32_t buffer_id = __nv_atomic_fetch_add(&globalVariables.nbNodesExchangedVisVoxels, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            globalVariables.exchangedAABBIndicesVisVoxelsTmp[buffer_id] = node_index;
            globalVariables.exchangedAABBIndicesVisVoxelsScreenSpaceSize[buffer_id] = screen_space_size;
        }
        
    }
}


extern "C" __global__
void kernel_get_renderable_nodes_part_5_reorder(
	CRenderTarget target,
    CRenderingSettings settings
){
    auto grid = cg::this_grid();
    uint32_t thread_id = grid.thread_rank();
    uint32_t nb_threads = grid.num_threads();

    uint32_t max_nb_exchanged_nodes = globalVariables.visibilityCacheSize;
    uint32_t nb_points = globalVariables.nbNodesExchangedVisPoints;
    uint32_t nb_voxels = globalVariables.nbNodesExchangedVisVoxels;

    // --- Bitonic sort (descending) on Points buffer ---
    // Pads logically to next power-of-two; out-of-range indices are treated as -inf
    for (uint32_t k = 2; k <= nb_points * 2; k <<= 1) {
        for (uint32_t j = k >> 1; j >= 1; j >>= 1) {
            for (uint32_t i = thread_id; i < nb_points; i += nb_threads) {
                uint32_t l = i ^ j;
                if (l > i && l < nb_points) {
                    // Ascending in index space → descending in value (biggest first)
                    bool swap = ((i & k) == 0)
                        ? (globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize[i] <
                           globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize[l])
                        : (globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize[i] >
                           globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize[l]);
                    if (swap) {
                        // Swap screen space sizes
                        float tmp_size = globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize[i];
                        globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize[i] =
                            globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize[l];
                        globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize[l] = tmp_size;
                        // Swap indices
                        uint32_t tmp_idx = globalVariables.exchangedAABBIndicesVisPointsTmp[i];
                        globalVariables.exchangedAABBIndicesVisPointsTmp[i] =
                            globalVariables.exchangedAABBIndicesVisPointsTmp[l];
                        globalVariables.exchangedAABBIndicesVisPointsTmp[l] = tmp_idx;
                    }
                }
            }
            grid.sync();
        }
    }

    // --- Bitonic sort (descending) on Voxels buffer ---
    for (uint32_t k = 2; k <= nb_voxels * 2; k <<= 1) {
        for (uint32_t j = k >> 1; j >= 1; j >>= 1) {
            for (uint32_t i = thread_id; i < nb_voxels; i += nb_threads) {
                uint32_t l = i ^ j;
                if (l > i && l < nb_voxels) {
                    bool swap = ((i & k) == 0)
                        ? (globalVariables.exchangedAABBIndicesVisVoxelsScreenSpaceSize[i] <
                           globalVariables.exchangedAABBIndicesVisVoxelsScreenSpaceSize[l])
                        : (globalVariables.exchangedAABBIndicesVisVoxelsScreenSpaceSize[i] >
                           globalVariables.exchangedAABBIndicesVisVoxelsScreenSpaceSize[l]);
                    if (swap) {
                        float tmp_size = globalVariables.exchangedAABBIndicesVisVoxelsScreenSpaceSize[i];
                        globalVariables.exchangedAABBIndicesVisVoxelsScreenSpaceSize[i] =
                            globalVariables.exchangedAABBIndicesVisVoxelsScreenSpaceSize[l];
                        globalVariables.exchangedAABBIndicesVisVoxelsScreenSpaceSize[l] = tmp_size;

                        uint32_t tmp_idx = globalVariables.exchangedAABBIndicesVisVoxelsTmp[i];
                        globalVariables.exchangedAABBIndicesVisVoxelsTmp[i] =
                            globalVariables.exchangedAABBIndicesVisVoxelsTmp[l];
                        globalVariables.exchangedAABBIndicesVisVoxelsTmp[l] = tmp_idx;
                    }
                }
            }
            grid.sync();
        }
    }

    // --- Copy top X entries into output buffers ---
    for (uint32_t i = thread_id; i < max_nb_exchanged_nodes; i += nb_threads) {
        globalVariables.exchangedAABBIndicesVisPoints[i] =
            (i < nb_points) ? globalVariables.exchangedAABBIndicesVisPointsTmp[i] : CINVALID_ID;
        globalVariables.exchangedAABBIndicesVisVoxels[i] =
            (i < nb_voxels) ? globalVariables.exchangedAABBIndicesVisVoxelsTmp[i] : CINVALID_ID;
    }
}