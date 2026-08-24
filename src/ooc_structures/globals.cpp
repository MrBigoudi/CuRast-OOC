#include "globals.h"

#include "allocator.h"
#include "structureUpdate.h"
#include "visibility.h"
#include "outOfCore.h"
#include "gpuVersion.h"

bool Point::operator==(const Point& rhs) const {
    if(position != rhs.position){return false;}
    for(uint32_t channel = 0; channel < 4; channel++){
        if(color[channel] != rhs.color[channel]){return false;}
    }
    return true;
}

vector<vec3> PointBatch::getPositions() const {
    vector<vec3> res = {};
    for(Point& point : *points){
        res.push_back(point.position);
    }
    return res;
}

vector<uint32_t> PointBatch::getColors() const {
    vector<uint32_t> res = {};
    for(Point& point : *points){
        uint32_t color = (uint32_t)point.color[0]
            | ((uint32_t)point.color[1] << 8)
            | ((uint32_t)point.color[2] << 16)
            | (0xFFu << 24)
        ;
        res.push_back(color);
    }
    return res;
}

vec3 AABB::getPointNormalizedCoordinates(const vec3& position) const {
    return vec3(
        (position.x - mins.x) / (maxs.x - mins.x),
        (position.y - mins.y) / (maxs.y - mins.y),
        (position.z - mins.z) / (maxs.z - mins.z)
    );
}

vec3 AABB::getPointWorldCoordinates(const vec3& normalized_position) const {
    return vec3(
        normalized_position.x * (maxs.x - mins.x) + mins.x,
        normalized_position.y * (maxs.y - mins.y) + mins.y,
        normalized_position.z * (maxs.z - mins.z) + mins.z
    ); 
}



bool AABB::contains(const vec3& position) const {
    return position.x > mins.x && position.x < maxs.x
        && position.y > mins.y && position.y < maxs.y
        && position.z > mins.z && position.z < maxs.z
    ;
}

bool AABB::isParentOf(const AABB& aabb) const {
    vec3 sizes = aabb.getSize() * 0.5f;
    vec3 centroid = aabb.getCentroid();
    vec3 positions[7] = {
        centroid,
        centroid - vec3(sizes.x, 0, 0),
        centroid + vec3(sizes.x, 0, 0),
        centroid - vec3(0, sizes.y, 0),
        centroid + vec3(0, sizes.y, 0),
        centroid - vec3(0, 0, sizes.y),
        centroid + vec3(0, 0, sizes.z)
    };
    
    for(const vec3& point : positions){
        if(!contains(point)){return false;}
    }
    return true;
}


vec3 AABB::getCentroid() const {
    return 0.5f*(mins + maxs);
}

vec3 AABB::getSize() const {
    return maxs - mins;
}

void AABB::extend(const NodePosition& position) {
    vec3 size = getSize();
    switch (position) {
        case FrontTopLeft:
            maxs.x += size.x;
            mins.y -= size.y;
            mins.z -= size.z;
            break;
        case FrontTopRight:
            mins.x -= size.x;
            mins.y -= size.y;
            mins.z -= size.z;
            break;
        case FrontBottomLeft:
            maxs.x += size.x;
            maxs.y += size.y;
            mins.z -= size.z;
            break;
        case FrontBottomRight:
            mins.x -= size.x;
            maxs.y += size.y;
            mins.z -= size.z;
            break;
        case BackTopLeft:
            maxs.x += size.x;
            mins.y -= size.y;
            maxs.z += size.z;
            break;
        case BackTopRight:
            mins.x -= size.x;
            mins.y -= size.y;
            maxs.z += size.z;
            break;
        case BackBottomLeft:
            maxs.x += size.x;
            maxs.y += size.y;
            maxs.z += size.z;
            break;
        case BackBottomRight:
            mins.x -= size.x;
            maxs.y += size.y;
            maxs.z += size.z;
            break;
    }
}

void AABB::shrink(const NodePosition& position) {
    vec3 size = getSize()*0.5f;
    switch (position) {
        case FrontTopLeft:
            maxs.x -= size.x;
            mins.y += size.y;
            mins.z += size.z;
            break;
        case FrontTopRight:
            mins.x += size.x;
            mins.y += size.y;
            mins.z += size.z;
            break;
        case FrontBottomLeft:
            maxs.x -= size.x;
            maxs.y -= size.y;
            mins.z += size.z;
            break;
        case FrontBottomRight:
            mins.x += size.x;
            maxs.y -= size.y;
            mins.z += size.z;
            break;
        case BackTopLeft:
            maxs.x -= size.x;
            mins.y += size.y;
            maxs.z -= size.z;
            break;
        case BackTopRight:
            mins.x += size.x;
            mins.y += size.y;
            maxs.z -= size.z;
            break;
        case BackBottomLeft:
            maxs.x -= size.x;
            maxs.y -= size.y;
            maxs.z -= size.z;
            break;
        case BackBottomRight:
            mins.x += size.x;
            maxs.y -= size.y;
            maxs.z -= size.z;
            break;
    }
}

NodePosition AABB::getNextChildIndex(const vec3& position) const {
    vec3 normalized_coordinates = getPointNormalizedCoordinates(position);
    bool is_front = normalized_coordinates.z >= 0.5f;
    bool is_top = normalized_coordinates.y >= 0.5f;
    bool is_right = normalized_coordinates.x >= 0.5f;
    if(is_right){
        if(is_top){
            if(is_front){
                return NodePosition::FrontTopRight;
            } else {
                return NodePosition::BackTopRight;
            }
        } else {
            if(is_front){
                return NodePosition::FrontBottomRight;
            } else {
                return NodePosition::BackBottomRight;
            }
        }
    } else {
        if(is_top){
            if(is_front){
                return NodePosition::FrontTopLeft;
            } else {
                return NodePosition::BackTopLeft;
            }
        } else {
            if(is_front){
                return NodePosition::FrontBottomLeft;
            } else {
                return NodePosition::BackBottomLeft;
            }
        }
    }
}


void updateNodePosition(NodePosition& position){
    switch(position){
        case FrontTopLeft:
            position = FrontTopRight;
            break;
        case FrontTopRight:
            position = FrontBottomLeft;
            break;
        case FrontBottomLeft:
            position = FrontBottomRight;
            break;
        case FrontBottomRight:
            position = BackTopLeft;
            break;
        case BackTopLeft:
            position = BackTopRight;
            break;
        case BackTopRight:
            position = BackBottomLeft;
            break;
        case BackBottomLeft:
            position = BackBottomRight;
            break;
        case BackBottomRight:
            position = FrontTopLeft;
            break;
    }
}


uint32_t OccupancyGrid::getNbFilledEntries() const {
    uint32_t cpt = 0;
    for(uint32_t i=0; i<OocSimLodSettings::GRID_SIZE / 32; i++){
        for(uint32_t j=0; j<32; j++){
            if(values[i] & (1u << j)){
                cpt++;
            }
        }
    }
    return cpt;
}
/// Return a pair (word_index, bit_index)
OccupancyGrid::GridIndex OccupancyGrid::getCellIndices(const AABB& aabb, const Point& point) {
    vec3 normalized_coordinates = aabb.getPointNormalizedCoordinates(point.position);
    uint32_t grid_x = clamp(
        uint32_t(floor(OocSimLodSettings::GRID_SIZE_PER_DIMENSION * normalized_coordinates.x)), 
        0u, 
        OocSimLodSettings::GRID_SIZE_PER_DIMENSION - 1u
    );
    uint32_t grid_y = clamp(
        uint32_t(floor(OocSimLodSettings::GRID_SIZE_PER_DIMENSION * normalized_coordinates.y)), 
        0u, 
        OocSimLodSettings::GRID_SIZE_PER_DIMENSION - 1u
    );
    uint32_t grid_z = clamp(
        uint32_t(floor(OocSimLodSettings::GRID_SIZE_PER_DIMENSION * normalized_coordinates.z)), 
        0u, 
        OocSimLodSettings::GRID_SIZE_PER_DIMENSION - 1u
    );
    uint32_t index = grid_x + OocSimLodSettings::GRID_SIZE_PER_DIMENSION * (grid_y + OocSimLodSettings::GRID_SIZE_PER_DIMENSION * grid_z);
    uint32_t word_index = index >> 5u;
    uint32_t bit_index = index & 31u;
    return GridIndex(word_index, bit_index, glm::uvec3(grid_x, grid_y, grid_z));
}
bool OccupancyGrid::isCellOcupied(const GridIndex& index) const {
    return (values[index.word].load() & (1u << index.bit)) != 0;
}
void OccupancyGrid::markCellAsFilled(const GridIndex& index){
    uint32_t old_value = values[index.word].load();
    uint32_t new_value = old_value |= (1u << index.bit);
    values[index.word].store(new_value);
}
vec3 OccupancyGrid::getCellCentroid(const AABB& aabb, const GridIndex& index) {
    vec3 world_grid_size = aabb.getSize() / float(OocSimLodSettings::GRID_SIZE_PER_DIMENSION);
	return aabb.mins + world_grid_size * vec3(index.grid.x, index.grid.y, index.grid.z) + 0.5f * world_grid_size;
}


uint32_t OctreeNode::getNbPoints() const {
    uint32_t res = 0;
    Chunk* point_chunk = points;
    while(point_chunk){
        res += point_chunk->size;
        point_chunk = point_chunk->next;
    }
    return res;
}

uint32_t OctreeNode::getNbChunks() const {
    uint32_t res = 0;
    Chunk* point_chunk = points;
    while(point_chunk){
        res++;
        point_chunk = point_chunk->next;
    }
    Chunk* voxel_chunk = voxels;
    while(voxel_chunk){
        res++;
        voxel_chunk = voxel_chunk->next;
    }
    return res;
}

uint32_t OctreeNode::getNbVoxels() const {
    uint32_t res = 0;
    Chunk* voxel_chunk = voxels;
    while(voxel_chunk){
        res += voxel_chunk->size;
        voxel_chunk = voxel_chunk->next;
    }
    return res;
}

uint32_t OctreeNode::getDepth() const {
    uint32_t max_level = 0;
    
    std::function<uint32_t(const OctreeNode*)> rec = [&](const OctreeNode* cur_node) -> uint32_t {
        if(!cur_node){return 0;}
        uint32_t max_children_depth = 0;
        for(uint32_t i=0; i<8; i++){
            uint32_t child_depth = rec(cur_node->children[i]);
            if(child_depth > max_children_depth){
                max_children_depth = child_depth;
            }
        }
        return 1+max_children_depth;
    };

    return rec(this);
}

uint32_t OctreeNode::getNbNodes(OctreeNode* root){
    uint32_t nb_nodes = 0;
    std::function<void(const OctreeNode*)> recursion = [&](const OctreeNode* cur_node){
        if(!cur_node){return;}
        nb_nodes++;
        for(uint32_t child = 0; child < 8; child++){
            recursion(cur_node->children[child]);
        }
    };
    recursion(root);
    return nb_nodes;
}

uint32_t OctreeNode::getNbGrids(OctreeNode* root){
    uint32_t nb_occupancy_grids = 0;
    std::function<void(const OctreeNode*)> recursion = [&](const OctreeNode* cur_node){
        if(!cur_node){return;}
        if(cur_node->occupancy){nb_occupancy_grids++;}
        for(uint32_t child = 0; child < 8; child++){
            recursion(cur_node->children[child]);
        }
    };
    recursion(root);
    return nb_occupancy_grids;
}

uint32_t OctreeNode::getNbChunks(OctreeNode* root){
    uint32_t nb_chunks = 0;
    std::function<void(const OctreeNode*)> recursion = [&](const OctreeNode* cur_node){
        if(!cur_node){return;}
        nb_chunks += cur_node->getNbChunks();
        for(uint32_t child = 0; child < 8; child++){
            recursion(cur_node->children[child]);
        }
    };
    recursion(root);
    return nb_chunks;
}

void OctreeNode::displayMemInfo() const {
    uint32_t nb_nodes = 0;
    uint32_t nb_chunks = 0;
    uint32_t nb_points = 0;
    uint32_t nb_voxels = 0;
    uint32_t nb_occupancy_grids = 0;

    std::function<void(const OctreeNode*)> recursion = [&](const OctreeNode* cur_node){
        if(!cur_node){return;}
        nb_nodes++;
        nb_chunks += cur_node->getNbChunks();
        nb_points += cur_node->getNbPoints();
        nb_voxels += cur_node->getNbVoxels();
        if(cur_node->occupancy){nb_occupancy_grids++;}

        for(uint32_t child = 0; child < 8; child++){
            recursion(cur_node->children[child]);
        }
    };

    recursion(this);
    println("nb nodes: {}, nb chunks: {}, nb points: {}, nb voxels: {}, nb grids: {}",
        nb_nodes, nb_chunks, nb_points, nb_voxels, nb_occupancy_grids
    );
}

void OctreeNode::display(uint32_t id, uint32_t level, bool node_only) const {
    println("level: {}, id: {}, aabb_index: {}, counter: {}, "
        "updated: {}, nbPoints: {}, nbVoxels: {}, "
        "visibility: {}, children visibility: 0b{}{}{}{}{}{}{}{}, "
        "points location: 0b{}{}{}{}{}{}{}{}, children: 0b{}{}{}{}{}{}{}{}",

        level, id, aabb_index, counter.load(), updated, getNbPoints(), getNbVoxels(), is_visible,
        uint8_t(bool(children_visibility & 0x01 << 0)),
        uint8_t(bool(children_visibility & 0x01 << 1)),
        uint8_t(bool(children_visibility & 0x01 << 2)),
        uint8_t(bool(children_visibility & 0x01 << 3)),
        uint8_t(bool(children_visibility & 0x01 << 4)),
        uint8_t(bool(children_visibility & 0x01 << 5)),
        uint8_t(bool(children_visibility & 0x01 << 6)),
        uint8_t(bool(children_visibility & 0x01 << 7)),
        uint8_t(bool(children_ids & 0x01 << 0)),
        uint8_t(bool(children_ids & 0x01 << 1)),
        uint8_t(bool(children_ids & 0x01 << 2)),
        uint8_t(bool(children_ids & 0x01 << 3)),
        uint8_t(bool(children_ids & 0x01 << 4)),
        uint8_t(bool(children_ids & 0x01 << 5)),
        uint8_t(bool(children_ids & 0x01 << 6)),
        uint8_t(bool(children_ids & 0x01 << 7)),
        uint8_t(children[0] != nullptr), 
        uint8_t(children[1] != nullptr), 
        uint8_t(children[2] != nullptr), 
        uint8_t(children[3] != nullptr),
        uint8_t(children[4] != nullptr), 
        uint8_t(children[5] != nullptr), 
        uint8_t(children[6] != nullptr), 
        uint8_t(children[7] != nullptr)
    );
    const AABB& aabb = GlobalVariables::getAABB(aabb_index);
    println("    aabb: mins = ({}, {}, {}), maxs = ({}, {}, {})",
        aabb.mins.x, aabb.mins.y, aabb.mins.z,
        aabb.maxs.x, aabb.maxs.y, aabb.maxs.z
    );
    if(!node_only){
        for(size_t i=0; i<8; i++){
            if(children[i]){
                children[i]->display(i, level+1);
            }
        }
    }
};

bool OctreeNode::operator==(const OctreeNode& rhs) const {

    std::function<bool(const OctreeNode*, const OctreeNode*)> recursion = 
        [&](const OctreeNode* cur_lhs, const OctreeNode* cur_rhs) -> bool
    {    
        if(cur_lhs->counter != cur_rhs->counter){
            println("OctreeNode::operator==: Wrong counter");
            return false;
        }
        if(cur_lhs->children_ids != cur_rhs->children_ids){
            println("OctreeNode::operator==: Wrong children ids");
            return false;
        }

       if(cur_lhs->aabb_index != cur_rhs->aabb_index){
            println("OctreeNode::operator==: Wrong aabbs");
            return false;
        }
        
        if(!cur_lhs->points && cur_rhs->points){
            println("OctreeNode::operator==: Wrong points, should be empty");
            return false;
        }
        if(cur_lhs->points){
            if(!cur_rhs->points){
                println("OctreeNode::operator==: Wrong points, should not be empty");
                return false;
            }
            const Chunk& lhs_points = *cur_lhs->points;
            const Chunk& rhs_points = *cur_rhs->points;
            if(lhs_points != rhs_points){
                println("OctreeNode::operator==: Wrong points");
                return false;
            }
        }
        
        if(!cur_lhs->voxels && cur_rhs->voxels){
            println("OctreeNode::operator==: Wrong voxels, should be empty");
            return false;
        }
        if(cur_lhs->voxels){
            if(!cur_rhs->voxels){
                println("OctreeNode::operator==: Wrong voxels, should not be empty");
                return false;
            }
            const Chunk& lhs_voxels = *cur_lhs->voxels;
            const Chunk& rhs_voxels = *cur_rhs->voxels;
            if(lhs_voxels != rhs_voxels){
                println("OctreeNode::operator==: Wrong voxels");
                return false;
            }
        }

        if(!cur_lhs->occupancy && cur_rhs->occupancy){
            println("OctreeNode::operator==: Wrong occupancy grid, should be empty");
            return false;
        }
        if(cur_lhs->occupancy){
            if(!cur_rhs->occupancy){
                println("OctreeNode::operator==: Wrong occupancy grid, should not be empty");
                return false;
            }
            const OccupancyGrid& lhs_grid = *cur_lhs->occupancy;
            const OccupancyGrid& rhs_grid = *cur_rhs->occupancy;
            if(lhs_grid != rhs_grid){
                println("OctreeNode::operator==: Wrong occupancy grid");
                return false;
            }
        }

        for(uint32_t i=0; i<8; i++){
            if(!cur_lhs->children[i] && !cur_rhs->children[i]){continue;}
            if(cur_lhs->children[i] && !cur_rhs->children[i]){
                println("OctreeNode::operator==: Wrong child, should not be empty");
                return false;
            }
            if(!cur_lhs->children[i] && cur_rhs->children[i]){
                println("OctreeNode::operator==: Wrong child, should be empty");
                return false;
            }
            const OctreeNode* lhs_child = cur_lhs->children[i];
            const OctreeNode* rhs_child = cur_rhs->children[i];
            if(!recursion(lhs_child, rhs_child)){
                println("OctreeNode::operator==: Wrong child");
                return false;
            }
        }

        return true;
    };

    return recursion(this, &rhs);
}


bool Chunk::operator==(const Chunk& rhs) const{
    const Chunk* lhs_chunk = this;
    const Chunk* rhs_chunk = &rhs;
    while(lhs_chunk){
        if(!rhs_chunk){return false;}
        if(lhs_chunk->size != rhs_chunk->size){return false;}
        for(uint32_t i=0; i<lhs_chunk->size; i++){
            const Point& lhs_point = lhs_chunk->points[i];
            const Point& rhs_point = rhs_chunk->points[i];
            if(lhs_point != rhs_point){return false;}
        }
        lhs_chunk = lhs_chunk->next;
        rhs_chunk = rhs_chunk->next;
    }
    if(rhs_chunk){return false;}
    return true;
}

void OctreeNode::rebuildOccupancy() {
    // Rebuild occupancy
    if(voxels){
        // occupancy = new OccupancyGrid();
        occupancy = MemoryAllocator::newOccupancyGrid();

        const Chunk* cur_chunk = voxels;
        while(cur_chunk){
            for(std::uint32_t point_id=0; point_id<cur_chunk->size; point_id++){
                const Point& point = cur_chunk->points[point_id];

                // Sample voxel occupancy grid at this location
                const AABB& aabb = GlobalVariables::getAABB(aabb_index);
                vec3 normalized_coordinates = aabb.getPointNormalizedCoordinates(point.position);
                uint32_t grid_x = clamp(
                    uint32_t(floor(OocSimLodSettings::GRID_SIZE_PER_DIMENSION * normalized_coordinates.x)), 
                    0u, 
                    OocSimLodSettings::GRID_SIZE_PER_DIMENSION - 1u
                );
                uint32_t grid_y = clamp(
                    uint32_t(floor(OocSimLodSettings::GRID_SIZE_PER_DIMENSION * normalized_coordinates.y)), 
                    0u, 
                    OocSimLodSettings::GRID_SIZE_PER_DIMENSION - 1u
                );
                uint32_t grid_z = clamp(
                    uint32_t(floor(OocSimLodSettings::GRID_SIZE_PER_DIMENSION * normalized_coordinates.z)), 
                    0u, 
                    OocSimLodSettings::GRID_SIZE_PER_DIMENSION - 1u
                );
                uint32_t index = grid_x + OocSimLodSettings::GRID_SIZE_PER_DIMENSION * (grid_y + OocSimLodSettings::GRID_SIZE_PER_DIMENSION * grid_z);
                uint32_t word_index = index >> 5u;
                uint32_t bit_index = index & 31u;
                occupancy->values[word_index] |= (1u << bit_index);
            }
            cur_chunk = cur_chunk->next;
        }
    }
}





///////////////////////////////////////////////////////////////////////////////
////////////////////////// GLOBAL EXTERNAL VARIABLES //////////////////////////
///////////////////////////////////////////////////////////////////////////////

std::string GlobalVariables::getSimLodOctreeName(bool generate_new_name){
    if(generate_new_name){
        simLodOctreeCounter++;
    }
    std::string name = format("MainOctreeSimLOD_{}", simLodOctreeCounter);
    // println("Octree name: {}", name);
    return name;
}

void GlobalVariables::init(CuRast* instance, CUcontext* context){
    /// The maps
    allOctreesRefCounter.init(1024);

    /// The queue of batches
    batchesQueue = std::deque<std::shared_ptr<PointBatch>>(OocSimLodSettings::BATCHES_LIST_SIZE, nullptr);
    batchesQueueMutexes = std::deque<std::mutex>(OocSimLodSettings::BATCHES_LIST_SIZE);
    /// The buffer of spilled points
    spilledPoints = std::make_shared<vector<Point>>(vector<Point>());
    /// The buffer of spilling nodes
    spillingNodes = std::make_shared<vector<OctreeNode*>>(vector<OctreeNode*>());

    /// The backlog buffer for new voxels
    backlogVoxels = std::make_shared<vector<Point>>(vector<Point>());
    /// The backlog buffer for the nodes corresponding to the new voxels
    backlogVoxelsNodes = std::make_shared<vector<OctreeNode*>>(vector<OctreeNode*>());

    updatesCache = std::make_shared<LRUCache>("updates cache", OocSimLodSettings::LRU_UPDATES_CACHE_SIZE);
    visibilityCache = std::make_shared<LRUCache>("visibility cache", OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE);
    cpuCache = std::make_shared<CPUFallbackCache>(OocSimLodSettings::LRU_CPU_CACHE_SIZE);

    for(auto& memory : batchedMemories){
        memory.init(instance, context);
    }
    
    // Initialise the allocator
    MemoryAllocator::init();

    cudaDeviceSynchronize();
}

std::string GlobalVariables::formatMemSize(uint64_t size_bytes, uint32_t pad){
    uint64_t nb_bytes = size_bytes;
    uint64_t nb_tbs = uint64_t(floor(nb_bytes / 1'024 / 1'024 / 1'024 / 1'024));
    nb_tbs = clamp(nb_tbs, uint64_t(0), uint64_t(999));
    nb_bytes -= nb_tbs * 1'024 * 1'024 * 1'024 * 1'024;
    uint64_t nb_gbs = uint64_t(floor(nb_bytes / 1'024 / 1'024 / 1'024));
    nb_gbs = clamp(nb_gbs, uint64_t(0), uint64_t(999));
    nb_bytes -= nb_gbs * 1'024 * 1'024 * 1'024;
    uint64_t nb_mbs = uint64_t(floor(nb_bytes / 1'024 / 1'024));
    nb_mbs = clamp(nb_mbs, uint64_t(0), uint64_t(999));
    nb_bytes -= nb_mbs * 1'024 * 1'024;
    uint64_t nb_kbs = uint64_t(floor(nb_bytes / 1'024));
    nb_kbs = clamp(nb_kbs, uint64_t(0), uint64_t(999));
    nb_bytes -= nb_kbs * 1'024;
    nb_bytes = clamp(nb_bytes, uint64_t(0), uint64_t(999));

    std::string pad_tbs = std::string(max(int32_t(pad - 20), 0), ' ');
    std::string pad_gbs = std::string(max(int32_t(pad - 15), 0), ' ');
    std::string pad_mbs = std::string(max(int32_t(pad - 10), 0), ' ');
    std::string pad_kbs = std::string(max(int32_t(pad - 5), 0), ' ');
    std::string pad_bs  = std::string(max(int32_t(pad), 0), ' ');
    if(nb_tbs){return format("{}{:3L}T {:3L}G {:3L}M {:3L}k {:3L}b", pad_tbs, nb_tbs, nb_gbs, nb_mbs, nb_kbs, nb_bytes);}
    if(nb_gbs){return format("{}{:3L}G {:3L}M {:3L}k {:3L}b", pad_gbs, nb_gbs, nb_mbs, nb_kbs, nb_bytes);}
    if(nb_mbs){return format("{}{:3L}M {:3L}k {:3L}b", pad_mbs, nb_mbs, nb_kbs, nb_bytes);}
    if(nb_kbs){return format("{}{:3L}k {:3L}b", pad_kbs, nb_kbs, nb_bytes);}
    return format("{}{:3L}b", pad_bs, nb_bytes);
}


void GlobalVariables::destroy(CuRast* instance, CUcontext* context){
    println("Begin destroy");

    mainLoopIsTerminating = true;

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    cudaDeviceSynchronize();

    OocSimLodSettings::display();
    displayTimings();
    displayBuffers();

    cudaDeviceSynchronize();
    for(auto& memory : batchedMemories){
        memory.destroy();
    }

    cudaDeviceSynchronize();
    freeOctreesOnGPU(CuRast::instance);

    cudaDeviceSynchronize();
    freeOctreesOnCPU();

    std::lock_guard<std::mutex> lock1(isUpdatingMtx);
    std::lock_guard<std::mutex> lock2(LRUCache::caches_sync_mtx);
    displayCpuMemoryUsage();
    println("GPU Memory Usage:\n{}", getGpuMemoryUsage());
}

IdAABB GlobalVariables::createNewAABB(const AABB& aabb){
    IdAABB id = IdAABB(allAABBs.size());
    if(id == INVALID_ID){
        println("ERROR: reached the maximum number of nodes that can be created");
        throw(EXIT_FAILURE);
    }
    allAABBs.push_back(aabb);
    aabbRelationshipMap->insertOrReplace(id, {
        INVALID_ID, INVALID_ID, INVALID_ID, INVALID_ID,
        INVALID_ID, INVALID_ID, INVALID_ID, INVALID_ID
    });
    return id;
}
const AABB& GlobalVariables::getAABB(const IdAABB& id){
    return allAABBs[id];
}

// This must be correctly synchronised
void GlobalVariables::swapAABBsMaps() {
    // We copy in a new pointer to synchronise with the main thread
    // In the main thread, we are using getting an access to the previous shared pointer which is automatically
    // destroyed on need
    aabbRelationshipMapCpy = std::make_shared<AABBRelationshipMap>(
        AABBRelationshipMap(*aabbRelationshipMap)
    );
}

void GlobalVariables::displayCpuMemoryUsage(){
    MemoryAllocator::displayInfo();

    println("Memory pre-allocated for CPU-GPU transfers:");
    uint64_t total_size = 0;
    for(auto& batch : batchedMemories){
        total_size += batch.memory_size;
    }
    println("    - CPU side: {}", formatMemSize(total_size));
    println("    - GPU side: {}", formatMemSize(total_size));

    if(mainOctree){
        println("MainOctree info:");
        printf("    - "); mainOctree->displayMemInfo();
    }
    if(mainOctreeCpy){
        println("MainOctreeCpy info:");
        printf("    - "); mainOctreeCpy->displayMemInfo();
    }

    println("\nUpdates cache info:");
    println("    - capacity: {}, size: {}", 
        updatesCache->CACHE_SIZE, 
        updatesCache->getSize()
    );
    println("Visibility cache info:");
    println("    - capacity: {}, size: {}", 
        visibilityCache->CACHE_SIZE, 
        visibilityCache->getSize()
    );

    println("\nTotal real usage: ");
    uint32_t real_nb_nodes = OctreeNode::getNbNodes(mainOctree) 
        + OctreeNode::getNbNodes(mainOctreeCpy)
    ;
    uint32_t real_nb_grids = OctreeNode::getNbGrids(mainOctree) 
        + OctreeNode::getNbGrids(mainOctreeCpy)
    ;
    uint32_t real_nb_chunks = OctreeNode::getNbChunks(mainOctree) 
        + OctreeNode::getNbChunks(mainOctreeCpy)
    ;
    uint32_t real_nb_octrees = allOctreesRefCounter.size;

    println("    - nb nodes = {}, lost nodes = {}", real_nb_nodes,
        MemoryAllocator::nodesAllocator->nb_allocated_elements.load() - real_nb_nodes
    );
    println("    - nb grids = {}, lost grids = {}", real_nb_grids,
        MemoryAllocator::gridsAllocator->nb_allocated_elements.load() - real_nb_grids
    );
    println("    - nb chunks = {}, lost chunks = {}", real_nb_chunks,
        MemoryAllocator::chunksAllocator->nb_allocated_elements.load() - real_nb_chunks
    );
    println("    - nb octrees = {}, lost octrees = {}", 
        real_nb_octrees, real_nb_octrees - 1
    );

    allOctreesRefCounter.mapWithKey([&](OctreeNode* node, uint32_t& counter){
        printf("    - counter[%p] = %u\n", node, counter);
    });
    println("\n");
}

std::string GlobalVariables::getGpuMemoryUsage() {
    cudaDeviceSynchronize();

    // https://forums.developer.nvidia.com/t/best-way-to-report-memory-consumption-in-cuda/21042
    uint64_t free_byte, total_byte = 0;
    double free_db, total_db, used_db = 0.;

    CURuntime::assertCudaSuccess(cuMemGetInfo(&free_byte, &total_byte));
    free_db = (double)free_byte; total_db = (double)total_byte; used_db = total_db - free_db;
    free_db /= (1024 * 1024); total_db /= (1024 * 1024); used_db /= (1024 * 1024);
    return format("    Total: {:L} Mb\n    InUse: {:L} Mb\n    Available: {:L} Mb",
        total_db, used_db, free_db
    );
}


std::vector<OctreeNode*> GlobalVariables::getAllNodes(OctreeNode* root){
    uint32_t guessed_nb_nodes = OocSimLodSettings::LRU_UPDATES_CACHE_SIZE + OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE;
    std::vector<OctreeNode*> res = {};
    res.reserve(guessed_nb_nodes);

    std::function<void(OctreeNode*)> recursion = [&](OctreeNode* cur_node){
        if(!cur_node){return;}
        res.push_back(cur_node);
        for(uint32_t child = 0; child < 8; child++){
            recursion(cur_node->children[child]);
        }
    };
    
    recursion(root);
    return res;
}

std::vector<std::pair<OctreeNode*, uint8_t>> GlobalVariables::getAllNodesWithLevel(OctreeNode* root, uint8_t initial_level){
    uint32_t guessed_nb_nodes = OocSimLodSettings::LRU_UPDATES_CACHE_SIZE + OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE;
    std::vector<std::pair<OctreeNode*, uint8_t>> res = {};
    res.reserve(guessed_nb_nodes);

    std::function<void(OctreeNode*, uint8_t)> recursion = [&](OctreeNode* cur_node, uint8_t level){
        if(!cur_node){return;}
        res.push_back({cur_node, level});
        for(uint32_t child = 0; child < 8; child++){
            recursion(cur_node->children[child], level + 1);
        }
    };
    
    recursion(root, initial_level);
    return res;
}

std::vector<OctreeNode*> GlobalVariables::getAllPartialLeaves(OctreeNode* root){
    uint32_t guessed_nb_nodes = OocSimLodSettings::LRU_UPDATES_CACHE_SIZE + OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE;
    std::vector<OctreeNode*> res = {};
    res.reserve(guessed_nb_nodes);

    std::function<void(OctreeNode*)> recursion = [&](OctreeNode* cur_node){
        if(!cur_node){return;}

        bool is_complete = true;
        for(uint32_t child = 0; child < 8; child++){
            if(!cur_node->children[child]){
                is_complete = false;
            } else {
                recursion(cur_node->children[child]);
            }
        }

        if(!is_complete){
            res.push_back(cur_node);
        }
    };
    
    recursion(root);
    return res;
}

std::vector<std::pair<OctreeNode*, uint8_t>> GlobalVariables::getAllPartialLeavesWithLevels(OctreeNode* root, uint8_t initial_level){
    uint32_t guessed_nb_nodes = OocSimLodSettings::LRU_UPDATES_CACHE_SIZE + OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE;
    std::vector<std::pair<OctreeNode*, uint8_t>> res = {};
    res.reserve(guessed_nb_nodes);

    std::function<void(OctreeNode*, uint8_t)> recursion = [&](OctreeNode* cur_node, uint8_t level){
        if(!cur_node){return;}

        bool is_complete = true;
        for(uint32_t child = 0; child < 8; child++){
            if(!cur_node->children[child]){
                is_complete = false;
            } else {
                recursion(cur_node->children[child], level + 1);
            }
        }

        if(!is_complete){
            res.push_back({cur_node, level});
        }
    };
    
    recursion(root, initial_level);
    return res;
}

std::vector<OctreeNode*> GlobalVariables::getAllPartialNodes(OctreeNode* root){
    uint32_t guessed_nb_nodes = OocSimLodSettings::LRU_UPDATES_CACHE_SIZE + OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE;
    std::vector<OctreeNode*> res = {};
    res.reserve(guessed_nb_nodes);

    std::function<void(OctreeNode*)> recursion = [&](OctreeNode* cur_node){
        if(!cur_node){return;}

        bool is_not_empty = false;
        for(uint32_t child = 0; child < 8; child++){
            if(cur_node->children[child]){
                is_not_empty = true;
                recursion(cur_node->children[child]);
            }
        }

        if(is_not_empty){
            res.push_back(cur_node);
        }
    };
    
    recursion(root);
    return res;
}

std::vector<std::pair<OctreeNode*, uint8_t>> GlobalVariables::getAllPartialNodesWithLevels(OctreeNode* root, uint8_t initial_level){
    uint32_t guessed_nb_nodes = OocSimLodSettings::LRU_UPDATES_CACHE_SIZE + OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE;
    std::vector<std::pair<OctreeNode*, uint8_t>> res = {};
    res.reserve(guessed_nb_nodes);

    std::function<void(OctreeNode*, uint8_t)> recursion = [&](OctreeNode* cur_node, uint8_t level){
        if(!cur_node){return;}

        bool is_not_empty = false;
        for(uint32_t child = 0; child < 8; child++){
            if(cur_node->children[child]){
                is_not_empty = true;
                recursion(cur_node->children[child], level + 1);
            }
        }

        if(is_not_empty){
            res.push_back({cur_node, level});
        }
    };
    
    recursion(root, initial_level);
    return res;
}







std::shared_ptr<Timing> Timing::addTiming(string name, bool start_now, uint32_t level){
    if(!OocSimLodSettings::MEASURE_TIMINGS){
        return std::make_shared<Timing>(name, start_now, level);
    }
    std::shared_ptr<Timing> new_timing = std::make_shared<Timing>(name, start_now, level);
    std::lock_guard<std::mutex> lock(timingsMtx);
    timingsList.push_back(new_timing);
    return timingsList.back();
}

void GlobalVariables::displayTimings(){
    if(!OocSimLodSettings::MEASURE_TIMINGS){return;}
	println("///////////////////////////////////////////////////");
	println("///////////////////// Timings /////////////////////");
	println("///////////////////////////////////////////////////\n");
	for (auto& timing : Timing::timingsList){
		uint64_t us = timing->duration.count();
		uint64_t s = uint64_t(us / 1'000'000);
		uint64_t ms = uint64_t((us - (s*1'000'000)) / 1'000);
		us = us - (s*1'000'000) - (ms*1'000);

        string tab = std::string(4*timing->level, ' ');

		println("{}- {}: {}s, {}ms, {}us",
			tab, timing->name, s, ms, us 
		);
	}
    println("\n///////////////////////////////////////////////////");
	println("///////////////////////////////////////////////////");
	println("///////////////////////////////////////////////////\n");
}

void GlobalVariables::displayBuffers(){
	println("///////////////////////////////////////////////////");
	println("///////////////////// Buffers /////////////////////");
	println("///////////////////////////////////////////////////\n");
	
    uint32_t nb_empty = 0;
    uint32_t nb_to_load = 0;
    uint32_t nb_loaded = 0;
    uint32_t nb_inserted = 0;
    uint32_t nb_to_remove = 0;
    for(uint32_t i=0; i<OocSimLodSettings::BATCHES_LIST_SIZE; i++){
        if(!batchesQueue[i]){
            nb_empty++;
            continue;
        }
        switch(batchesQueue[i]->state){
            case Empty:
                println("Error: there should not be a batch with an Empty state...");
                break;
            case ToLoad:
                nb_to_load++;
                break;
            case Loaded:
                nb_loaded++;
                break;
            case Inserted:
                nb_inserted++;
                break;
            case ToRemove:
                nb_to_remove++;
                break;
            break;
        }
    }
    println("- batches: {} empty, {} to load, {} loaded, {} inserted, {} to remove", nb_empty, nb_to_load, nb_loaded, nb_inserted, nb_to_remove);
    println("- spilledPoints: {} elements", spilledPoints->size());
    println("- spillingNodes: {} elements", spillingNodes->size());
    println("- backlogVoxels: {} elements", backlogVoxels->size());
    println("- backlogVoxelsNodes: {} elements", backlogVoxelsNodes->size());

    println("\n///////////////////////////////////////////////////");
	println("///////////////////////////////////////////////////");
	println("///////////////////////////////////////////////////\n");
}











///////////////////////////////////////////////////////////////////////////////
/////////////////////////// LRU CACHING SHENANIGANS ///////////////////////////
///////////////////////////////////////////////////////////////////////////////


std::optional<IdAABB> LRUCache::add(const IdAABB& aabb_index){
    CDoubleLinkedList<IdAABB>::Iterator** it = cache_map.find(aabb_index);

    // If the AABB was already in cache, remove its old version from the list
    if(it){
        cache.moveBegin(*it);
        return nullopt;
    }

    std::optional<IdAABB> old_aabb = nullopt;

    // If the cache is full, remove the last node
    if(cache_map.size >= CACHE_SIZE){
        old_aabb = *cache.back();
        cache.popBack();
        if(!cache_map.erase(old_aabb.value())){
            println("Erasing an LRU entry should always work at this point: old_aabb.value() = {}", old_aabb.value());
            throw(EXIT_FAILURE);
        }
    }

    // Insert the new node at the front of the list
    cache.pushFront(aabb_index);
    cache_map[aabb_index] = cache.begin();

    return old_aabb;
}

bool LRUCache::contains(const IdAABB& aabb_index) {
    return cache_map.contains(aabb_index);
}

uint32_t LRUCache::getSize() const {
    return cache_map.size;
}


void LRUCache::display() {
    std::string pad = std::string(max(int32_t(name.size())-2, 0), '/');
    println("////////////////////////////////////////////////{}", pad);
	println("////////////////////// {} //////////////////////", name);
	println("////////////////////////////////////////////////{}\n", pad);
    uint32_t index = 0;

    CDoubleLinkedList<IdAABB>::Iterator* list_it = cache.begin();
    while(list_it){
        const IdAABB& aabb_index = list_it->value;
        const AABB& aabb = GlobalVariables::getAABB(aabb_index);
        std::string output = format("mins = ({}, {}, {}), maxs = ({}, {}, {})",
            aabb.mins.x, 
            aabb.mins.y, 
            aabb.mins.z, 
            aabb.maxs.x, 
            aabb.maxs.y, 
            aabb.maxs.z
        );
        println("- [ {} ]: {}", index, output);
        index++;
        list_it = list_it->next;
    }
	println("\n////////////////////////////////////////////////{}", pad);
    println("////////////////////////////////////////////////{}", pad);
	println("////////////////////////////////////////////////{}\n", pad);
}


///////////////////////////////////////////////////////////////////////////////
////////////////////////// MEMORY BATCHING ALLOCATOR //////////////////////////
///////////////////////////////////////////////////////////////////////////////


void BatchedMemory::init(CuRast* instance, CUcontext* context){
    // Sanity check
    if(sizeof(CChunk) != sizeof(Chunk)
        || sizeof(COctreeNode) != sizeof(OctreeNode)
        || alignof(CChunk) != alignof(Chunk)
        || alignof(COctreeNode) != alignof(OctreeNode)
        || sizeof(CUdeviceptr) != sizeof(CChunk*)
        || sizeof(CUdeviceptr) != sizeof(COctreeNode*)
        || sizeof(CUdeviceptr) != sizeof(Chunk*)
        || sizeof(CUdeviceptr) != sizeof(OctreeNode*)
        || sizeof(CChunk*) != sizeof(Chunk*)
        || sizeof(COctreeNode*) != sizeof(OctreeNode*)
    ){
        println("Sizes don't match");
        throw(EXIT_FAILURE);
    }

    uint64_t max_nb_nodes = OocSimLodSettings::LRU_UPDATES_CACHE_SIZE
        + OocSimLodSettings::LRU_VISIBILITY_CACHE_SIZE
    ;
    uint64_t max_nb_chunks = 
        float(OocSimLodSettings::MAX_POINTS_PER_LEAF) / float(OocSimLodSettings::NB_POINTS_PER_CHUNK)
        * max_nb_nodes
    ;
    memory_size = max_nb_nodes * sizeof(OctreeNode) + max_nb_chunks * sizeof(Chunk);

    // GPU allocation
    CUresult cuda_status = cuMemAlloc(&gpu_allocated_memory, memory_size);
    CURuntime::assertCudaSuccess(cuda_status);

    // CPU allocation
    allocated_memory = (char*)malloc(memory_size);

    reset();
}

void BatchedMemory::reset(){
    next_space_pointer = 0;
    srcs.clear();
    dsts.clear();
    sizes.clear();
}

void BatchedMemory::destroy(){
    // CPU free
    free(allocated_memory);

    // GPU free
    // TODO: fix this by creating a cleaning routine on GlobalVariables
    CUresult cuda_status = cuMemFree(gpu_allocated_memory);
    CURuntime::assertCudaSuccess(cuda_status);
}


void BatchedMemory::copyMemory(CUcontext* context, CUstream* stream) {
    size_t batch_size = dsts.size();

    CUmemcpyAttributes attributes = CUmemcpyAttributes();
    attributes.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    attributes.srcLocHint.type = CU_MEM_LOCATION_TYPE_HOST;
    attributes.dstLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
    size_t attributes_idxs = 0;
    size_t nb_attributes = 1;

	CUresult cuda_status = CUDA_SUCCESS;
    cuda_status = cuMemcpyBatchAsync((CUdeviceptr*)dsts.data(), (CUdeviceptr*)srcs.data(), 
        (size_t*)sizes.data(), batch_size,
        &attributes, &attributes_idxs, nb_attributes, *stream
    );
    CURuntime::assertCudaSuccess(cuda_status);

    if(!OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
        cudaDeviceSynchronize();
    }
}

void BatchedMemory::display() const {
    for(uint32_t i=0; i<srcs.size(); i++){
        char* src = (char*)srcs[i];
        if(sizes[i] == sizeof(CChunk)){
            println("Chunk: size = {}", ((CChunk*)src)->size);
        } else if(sizes[i] == sizeof(COctreeNode)){
            COctreeNode* node = (COctreeNode*)src;
            println("level: {}, counter: {}, children visibility: 0b{}{}{}{}{}{}{}{}, points location: 0b{}{}{}{}{}{}{}{}, children: 0b{}{}{}{}{}{}{}{}",
                node->level, node->points_counter,
                uint8_t(bool(node->children_visibility & 0x01 << 0)),
                uint8_t(bool(node->children_visibility & 0x01 << 1)),
                uint8_t(bool(node->children_visibility & 0x01 << 2)),
                uint8_t(bool(node->children_visibility & 0x01 << 3)),
                uint8_t(bool(node->children_visibility & 0x01 << 4)),
                uint8_t(bool(node->children_visibility & 0x01 << 5)),
                uint8_t(bool(node->children_visibility & 0x01 << 6)),
                uint8_t(bool(node->children_visibility & 0x01 << 7)),
                uint8_t(bool(node->children_ids & 0x01 << 0)),
                uint8_t(bool(node->children_ids & 0x01 << 1)),
                uint8_t(bool(node->children_ids & 0x01 << 2)),
                uint8_t(bool(node->children_ids & 0x01 << 3)),
                uint8_t(bool(node->children_ids & 0x01 << 4)),
                uint8_t(bool(node->children_ids & 0x01 << 5)),
                uint8_t(bool(node->children_ids & 0x01 << 6)),
                uint8_t(bool(node->children_ids & 0x01 << 7)),
                uint8_t(node->children[0] != nullptr), 
                uint8_t(node->children[1] != nullptr), 
                uint8_t(node->children[2] != nullptr), 
                uint8_t(node->children[3] != nullptr),
                uint8_t(node->children[4] != nullptr), 
                uint8_t(node->children[5] != nullptr), 
                uint8_t(node->children[6] != nullptr), 
                uint8_t(node->children[7] != nullptr)
            );
        }
    }
}











///////////////////////////////////////////////////////////////////////////////
/////////////////////// CUDA UNIFIED MEMORY SHENANIGANS ///////////////////////
///////////////////////////////////////////////////////////////////////////////

std::vector<std::shared_ptr<std::vector<vec3>>> unified_positions = {};
std::vector<std::shared_ptr<std::vector<uint32_t>>> unified_colors = {};

CFullOctreeUnifiedBuilder unifiedOctreeBuilder = {};
void CFullOctreeUnifiedBuilder::update() {
    nodes.clear();
    num_nodes = 0;
    max_lod_level = 0;

    std::function<void(OctreeNode*, uint8_t)> recursion = [&](
        OctreeNode* cur_node, uint8_t level
    ) -> void {

        if(cur_node){
            max_lod_level = std::max(max_lod_level, uint32_t(level));
            num_nodes++;
            cur_node->level = level;
            cur_node->is_large = false;
            cur_node->is_visible = false;
            cur_node->is_cut = false;
            nodes.push_back(cur_node);

            for(uint32_t child = 0; child < 8; child++){
                recursion(cur_node->children[child], level+1);
            }
        }
    };

    recursion(GlobalVariables::mainOctree, 0);
}


CFullOctreeUnified CFullOctreeUnifiedBuilder::build() {
    CFullOctreeUnified res = {};
    res.nodes = (COctreeNodeUnified**)nodes.data();
    res.num_nodes = num_nodes;
    res.max_lod_level = max_lod_level;
    res.world = mat4(1.f);
    return res;
}



void GlobalVariables::updateGPU(CuRast* instance, CUcontext* context, View* view,
    std::optional<OctreeNode*>& octree_ref,
    std::shared_ptr<AABBRelationshipMap> relationship_map_ref)
{
    if(mainLoopIsTerminating){return;}
    if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL || elapsedFrames > OocSimLodSettings::NUMBER_OF_FRAMES_BETWEEN_DATA_EXCHANGE){
        elapsedFrames = 0;

        // Skip the update if too many octrees are in the scene
        if(GlobalVariables::nbOctreesInScene >= GlobalVariables::batchedMemories.size()){
            return;
        }
        
        {
            auto lock = OocSimLodSettings::IS_RUNNING_IN_PARALLEL
                ? std::unique_lock<std::mutex>(GlobalVariables::isUpdatingMtx) 
                : std::unique_lock<std::mutex>();
                
            if(octree_ref.has_value() && octree_ref.value()){
                allOctreesRefCounter[octree_ref.value()]--;
            }
            octree_ref = std::optional<OctreeNode*>(mainOctree);
            if(octree_ref.has_value() && octree_ref.value()){
                allOctreesRefCounter[octree_ref.value()]++;
            }
            relationship_map_ref = aabbRelationshipMapCpy;
        }


        bool has_scene_changed = Visibility::updateVisibilityCache(VKRenderer::view.view, VKRenderer::view.proj,
            *octree_ref, relationship_map_ref
        );
        if(has_scene_changed){
            loadOctreeOnGPU(instance, context,
                *octree_ref, relationship_map_ref
            );
        }

    }
}


void GlobalVariables::swapOctrees(){
    if(!mainOctree || !mainOctreeCpy){return;}

    allOctreesRefCounter[mainOctree]--;

    // OctreeNode* new_octree = MemoryAllocator::newOctreeNodeCpy(mainOctreeCpy);
    // {
    //     // Augment mainOctreeCpy
    //     std::function<void(OctreeNode*, OctreeNode*)> recursion = [&](OctreeNode* cur_node, OctreeNode* old_node){
    //         if(!old_node){return;}

    //         for(uint32_t child_id = 0; child_id < 8; child_id++){
    //             OctreeNode* cur_child = cur_node->children[child_id];
    //             OctreeNode* old_child = old_node->children[child_id];

    //             // If the new octree misses one child, add it
    //             if(old_child && !cur_child){
    //                 OctreeNode* new_child = MemoryAllocator::newOctreeNodeCpy(old_child, true);
    //                 cur_node->children[child_id] = new_child;
    //                 recursion(new_child, cur_child);
    //             } else {
    //                 recursion(cur_child, old_child);
    //             }
    //         }

    //     };
    //     std::lock_guard<std::mutex> lock_send(LRUCache::caches_sync_mtx);
    //     recursion(new_octree, mainOctree);
    // }
    // mainOctree = new_octree;

    // // For now, the mainOctree after visibility updates is way bigger than the one after updates update
    // // Also, the one after updates update never loads the node in the visibility update tree
    // mainOctree = MemoryAllocator::newOctreeNodeCpy(mainOctreeCpy);
    mainOctree = MemoryAllocator::newOctreeNodePartialCpy(mainOctreeCpy, mainOctree);
    allOctreesRefCounter[mainOctree] = 1;
}


void GlobalVariables::freeOctreesOnCPU(){
    if(allOctreesRefCounter.size <= 1){
        return;
    }

    // // https://stackoverflow.com/questions/15662412/how-to-remove-multiple-items-from-unordered-map-while-iterating-over-it
    // for (auto it = allOctreesRefCounter.begin(); it != allOctreesRefCounter.end();) {
    //     OctreeNode* node = it->first;
    //     uint32_t counter = it->second;
    //     if(counter == 0) {
    //         MemoryAllocator::delOctreeNode(node);
    //         it = allOctreesRefCounter.erase(it);
    //     } else {
    //         it++;
    //     }
    // }

    std::vector<OctreeNode*> to_erase = {};
    allOctreesRefCounter.mapWithKey([&](OctreeNode* node, uint32_t counter){
        if(counter == 0){
            to_erase.push_back(node);            
        }
    });
    for(OctreeNode* node : to_erase){
        allOctreesRefCounter.erase(node);
        MemoryAllocator::delOctreeNode(node);
    }
}