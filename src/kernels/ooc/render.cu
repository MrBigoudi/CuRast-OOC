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
            // For dataset creation
            if(i % settings.draw_every_x_points != 0){continue;}

            const CPoint& point = cur_points->points[i];
            drawPoint(target, settings, point.position, point.color);
        }
        cur_points = cur_points->next;
    }
}






__device__
struct CScreenRect {
    float x0, y0, x1, y1;   // unclipped screen-space bounds (pixels)
    bool crosses_near;      // at least one corner is behind / on the near plane
};

__device__ constexpr float VISIBILITY_NEAR_EPS      = 1e-4f;

__device__
CScreenRect projectAABB(const CRenderTarget& target, const CAABB& aabb){
    const mat4 transform = target.proj * target.view;
    const float W = float(target.width);
    const float H = float(target.height);

    CScreenRect r = {INFINITY, INFINITY, -INFINITY, -INFINITY, false};

    #pragma unroll
    for(uint32_t c = 0; c < 8; c++){
        vec4 p = {
            (c & 1) ? aabb.maxs.x : aabb.mins.x,
            (c & 2) ? aabb.maxs.y : aabb.mins.y,
            (c & 4) ? aabb.maxs.z : aabb.mins.z,
            1.0f
        };
        vec4 q = transform * p;
        if(q.w <= VISIBILITY_NEAR_EPS){
            r.crosses_near = true;
            continue;
        }
        float sx = (q.x / q.w * 0.5f + 0.5f) * W;
        float sy = (q.y / q.w * 0.5f + 0.5f) * H;
        r.x0 = min(r.x0, sx);  r.x1 = max(r.x1, sx);
        r.y0 = min(r.y0, sy);  r.y1 = max(r.y1, sy);
    }
    return r;
}

__device__ __forceinline__
bool isCameraInside(const CRenderTarget& target, const CAABB& aabb){
    const vec3 cam = target.camera_pos;
    return cam.x > aabb.mins.x && cam.x < aabb.maxs.x
        && cam.y > aabb.mins.y && cam.y < aabb.maxs.y
        && cam.z > aabb.mins.z && cam.z < aabb.maxs.z;
}






__device__
bool isLargerThanMinSpanning(
    const CRenderTarget& target,
	CRenderingSettings settings,
    const CAABB& aabb
){

    if(isCameraInside(target, aabb)){return true;}

    CScreenRect r = projectAABB(target, aabb);
    if(r.crosses_near){return true;}   // right in front of the camera: always large

    float dx = r.x1 - r.x0;
    float dy = r.y1 - r.y0;
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
            // // TODO: to remove
            // __nv_atomic_add(&globalVariables.curNbVisNode, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
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

    // // TODO: to remove
    // if(block_id == 0 && thread_id == 0){
    //     printf("nb large nodes = %d\n", globalVariables.curNbVisNode);
    //     globalVariables.curNbVisNode = 0;
    // }

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
        // For dataset creation
        if(point_id % settings.draw_every_x_points != 0){continue;}

        const CPoint& point = rendered_points[point_id];
        drawPoint(target, settings, point.position,
            settings.use_voxels_debug_color ? 0xff00ffff : point.color
        );
    }

    // Render voxels
    for(uint32_t voxel_id = thread_id; voxel_id < nb_voxels; voxel_id += nb_threads){
        // For dataset creation
        if(voxel_id % settings.draw_every_x_points != 0){continue;}

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
                // // TODO: to remove
                // if(child->voxels_counter > 0){
                //     __nv_atomic_add(&globalVariables.curNbVisNode, child->voxels_counter, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                // }
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
            // For dataset creation
            if(i % settings.draw_every_x_points != 0){continue;}

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

    // // TODO: to remove
    // if(block_id == 0 && thread_id == 0){
    //     printf("nb rendered voxels = %d\n", globalVariables.curNbVisNode);
    //     globalVariables.curNbVisNode = 0;
    // }

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
    uint64_t* colorbuffer,           // depth in the high 32 bits
    float*    out_tensor,
    uint32_t  base_width,
    uint32_t  base_height,
    uint32_t  nb_levels,
    uint32_t  nb_channels,           // == NeuralNet::config.level_channels()
    uint32_t  flip_y,                // 1 = match stbi_flip_vertically_on_write(1)
    uint64_t* level_cb_offsets,
    uint64_t* level_tensor_offsets
){
    uint64_t thread_id     = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t total_threads = uint64_t(blockDim.x) * gridDim.x;

    uint32_t* resolved[4] = { resolved_level0, resolved_level1, resolved_level2, resolved_level3 };

    for(uint32_t level = 0; level < nb_levels; level++){
        uint32_t w = base_width  >> level;
        uint32_t h = base_height >> level;
        uint64_t pixels     = uint64_t(w) * h;
        uint64_t tensor_off = level_tensor_offsets[level];
        uint64_t cb_off     = level_cb_offsets[level];

        for(uint64_t i = thread_id; i < pixels; i += total_threads){
            uint32_t x = uint32_t(i % w);
            uint32_t y = uint32_t(i / w);
            uint64_t src = uint64_t(flip_y ? (h - 1 - y) : y) * w + x;

            uint32_t rgba = resolved[level][src];
            out_tensor[tensor_off + 0 * pixels + i] = float( rgba        & 0xff) / 255.0f;
            out_tensor[tensor_off + 1 * pixels + i] = float((rgba >>  8) & 0xff) / 255.0f;
            out_tensor[tensor_off + 2 * pixels + i] = float((rgba >> 16) & 0xff) / 255.0f;

            if(nb_channels > 3){
                // Replicates kernel_resolve_depthbuffer_to_screenshot + PIL "L"
                uint64_t px         = colorbuffer[cb_off + src];
                uint32_t depth_bits = uint32_t(px >> 32);
                bool     is_hole    = isinf(__uint_as_float(depth_bits));
                uint8_t  depth_b    = is_hole ? 0 : uint8_t(depth_bits);
                out_tensor[tensor_off + 3 * pixels + i] = float(depth_b) / 255.0f;
            }
        }
    }
}

extern "C" __global__
void kernel_pack_tensor_to_colorbuffer(
    float*    tensor,
    uint64_t* colorbuffer,
    uint32_t  width,
    uint32_t  height,
    uint32_t  flip_y
){
    uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t numPixels = uint64_t(width) * height;
    if(i >= numPixels) return;

    uint8_t r = uint8_t(__saturatef(tensor[0 * numPixels + i]) * 255.0f);
    uint8_t g = uint8_t(__saturatef(tensor[1 * numPixels + i]) * 255.0f);
    uint8_t b = uint8_t(__saturatef(tensor[2 * numPixels + i]) * 255.0f);
    uint32_t color = uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (0xffu << 24);

    uint32_t x = uint32_t(i % width);
    uint32_t y = uint32_t(i / width);
    uint64_t dst = uint64_t(flip_y ? (height - 1 - y) : y) * width + x;

    colorbuffer[dst] = (uint64_t(__float_as_uint(1.0f)) << 32) | uint64_t(color);
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



__device__ constexpr float VISIBILITY_HYSTERESIS_BONUS = 1.15f;
__device__ constexpr float VISIBILITY_CENTER_WEIGHT = 0.5f;   // 0 = no centre bias, 1 = corners score 0

__device__
float getVisibilityPriority(const CRenderTarget& target, const CAABB& aabb){
    const float W = float(target.width);
    const float H = float(target.height);
    const float full_screen = W * H;

    if(isCameraInside(target, aabb)){ return full_screen; }

    CScreenRect r = projectAABB(target, aabb);
    if(r.crosses_near){ return full_screen; }

    // Clip to the viewport
    float x0 = clamp(r.x0, 0.0f, W), x1 = clamp(r.x1, 0.0f, W);
    float y0 = clamp(r.y0, 0.0f, H), y1 = clamp(r.y1, 0.0f, H);
    float area = max(x1 - x0, 0.0f) * max(y1 - y0, 0.0f);
    if(area <= 0.0f){ return 0.0f; }

    // Centre weighting on the clipped rectangle's centre
    vec2 c = {0.5f * (x0 + x1) - 0.5f * W, 0.5f * (y0 + y1) - 0.5f * H};
    float r_norm = clamp(length(c) / (0.5f * length(vec2{W, H})), 0.0f, 1.0f);
    float score = area * (1.0f - VISIBILITY_CENTER_WEIGHT * r_norm);

    return isfinite(score) ? score : 0.0f;
}



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
            // TODO: to remove
            // __nv_atomic_add(&globalVariables.curNbVisNode, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
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
            // TODO: to remove
            // __nv_atomic_add(&globalVariables.curNbVisNode, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
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

        uint32_t nb_points = globalVariables.relationshipMap[node_index].points_stored;
        CIdAABB* children = globalVariables.relationshipMap[node_index].children;

        if(nb_points > 0){
            uint32_t buffer_id = __nv_atomic_fetch_add(&globalVariables.nbNodesExchangedVisPoints, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            globalVariables.exchangedAABBIndicesVisPointsTmp[buffer_id] = node_index;

            const CAABB& aabb = globalVariables.relationshipMap[node_index].aabb;
            float priority = getVisibilityPriority(target, aabb);
            if(globalVariables.getFlag(node_index, CFlagWasVisibleHost)){
                priority *= VISIBILITY_HYSTERESIS_BONUS;
            }
            globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize[buffer_id] = priority;
        }

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
            uint32_t nb_voxels = globalVariables.relationshipMap[node_index].voxels_stored; 
            uint32_t nb_points = globalVariables.relationshipMap[node_index].points_stored; 
            float screen_space_size = getVisibilityPriority(target, aabb);
            if(globalVariables.getFlag(node_index, CFlagWasVisibleHost)){
                screen_space_size *= VISIBILITY_HYSTERESIS_BONUS;
            }

            if(nb_points > 0){
                uint32_t points_buffer_id = __nv_atomic_fetch_add(&globalVariables.nbNodesExchangedVisPoints, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                globalVariables.exchangedAABBIndicesVisPointsTmp[points_buffer_id] = node_index;
                globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize[points_buffer_id] = screen_space_size;
            }

            if(nb_voxels > 0){
                uint32_t voxels_buffer_id = __nv_atomic_fetch_add(&globalVariables.nbNodesExchangedVisVoxels, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                globalVariables.exchangedAABBIndicesVisVoxelsTmp[voxels_buffer_id] = node_index;
                globalVariables.exchangedAABBIndicesVisVoxelsScreenSpaceSize[voxels_buffer_id] = screen_space_size;
            }
        }
        
    }
}


__device__ __forceinline__
uint32_t nextPow2(uint32_t n){
    return n <= 1 ? n : (1u << (32 - __clz(n - 1)));
}

__device__
void bitonicSortDescending(
    cg::grid_group& grid,
    float* keys, CIdAABB* ids,
    uint32_t n, uint32_t capacity
){
    const uint32_t thread_id  = grid.thread_rank();
    const uint32_t nb_threads = grid.num_threads();

    uint32_t n2 = nextPow2(n);

#ifdef ASSERT_ENABLED
    if(n2 > capacity){
        if(thread_id == 0){
            printf("ERROR: bitonic sort needs %u slots, buffers only have %u\n", n2, capacity);
        }
        customAssert();
    }
#endif
    n2 = min(n2, capacity);   // uniform across threads, so grid.sync() counts stay equal

    // Pad with -inf (sorted to the end in descending order)
    for(uint32_t i = n + thread_id; i < n2; i += nb_threads){
        keys[i] = -INFINITY;
        ids[i]  = CINVALID_ID;
    }
    grid.sync();

    for(uint32_t k = 2; k <= n2; k <<= 1){
        for(uint32_t j = k >> 1; j > 0; j >>= 1){
            for(uint32_t i = thread_id; i < n2; i += nb_threads){
                uint32_t l = i ^ j;
                if(l <= i){continue;}

                float a = keys[i];
                float b = keys[l];
                // (i & k) == 0 -> descending block, else ascending block;
                // the final pass (k == n2) is fully descending
                bool swap = ((i & k) == 0) ? (a < b) : (a > b);
                if(swap){
                    keys[i] = b;
                    keys[l] = a;
                    CIdAABB tmp = ids[i];
                    ids[i] = ids[l];
                    ids[l] = tmp;
                }
            }
            grid.sync();
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

    uint32_t max_nb_exchanged_nodes = globalVariables.visibilityCacheSize / 2;
    uint32_t nb_points = globalVariables.nbNodesExchangedVisPoints;
    uint32_t nb_voxels = globalVariables.nbNodesExchangedVisVoxels;

    uint32_t sort_capacity = globalVariables.maxNbConcurrentNodes;

    // Clear previous "was visible" flags
    for(uint32_t i = thread_id; i < max_nb_exchanged_nodes; i += nb_threads){
        CIdAABB old_points_id = globalVariables.exchangedAABBIndicesVisPoints[i];
        if(old_points_id != CINVALID_ID){
            globalVariables.unsetFlagSync(old_points_id, CFlagWasVisibleHost);
        }
        CIdAABB old_voxels_id = globalVariables.exchangedAABBIndicesVisVoxels[i];
        if(old_voxels_id != CINVALID_ID){
            globalVariables.unsetFlagSync(old_voxels_id, CFlagWasVisibleHost);
        }
    }
    // Make sure every unset lands before any set below (the sorts may do zero syncs)
    grid.sync();

    bitonicSortDescending(grid,
        globalVariables.exchangedAABBIndicesVisPointsScreenSpaceSize,
        globalVariables.exchangedAABBIndicesVisPointsTmp,
        nb_points, sort_capacity
    );
    bitonicSortDescending(grid,
        globalVariables.exchangedAABBIndicesVisVoxelsScreenSpaceSize,
        globalVariables.exchangedAABBIndicesVisVoxelsTmp,
        nb_voxels, sort_capacity
    );

    // Copy the top entries into the output buffers
    for(uint32_t i = thread_id; i < max_nb_exchanged_nodes; i += nb_threads){
        CIdAABB new_points_id = (i < nb_points) ? globalVariables.exchangedAABBIndicesVisPointsTmp[i] : CINVALID_ID;
        CIdAABB new_voxels_id = (i < nb_voxels) ? globalVariables.exchangedAABBIndicesVisVoxelsTmp[i] : CINVALID_ID;

        globalVariables.exchangedAABBIndicesVisPoints[i] = new_points_id;
        globalVariables.exchangedAABBIndicesVisVoxels[i] = new_voxels_id;

        if(new_points_id != CINVALID_ID){
            globalVariables.setFlagSync(new_points_id, CFlagWasVisibleHost);
        }
        if(new_voxels_id != CINVALID_ID){
            globalVariables.setFlagSync(new_voxels_id, CFlagWasVisibleHost);
        }
    }
}