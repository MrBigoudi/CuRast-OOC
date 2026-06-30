#include "globals.h"

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


uint32_t OctreeNode::getNbPoints() const {
    uint32_t res = 0;
    Chunk* point_chunk = points;
    while(point_chunk){
        res += point_chunk->size;
        point_chunk = point_chunk->next;
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

void OctreeNode::display(uint32_t id, uint32_t level, bool node_only) const {
    println("level: {}, id: {}, counter: {}, "
        "updated: {}, nbPoints: {}, nbVoxels: {}, "
        "visibility: {}, children visibility: 0b{}{}{}{}{}{}{}{}, "
        "points location: 0b{}{}{}{}{}{}{}{}, children: 0b{}{}{}{}{}{}{}{}",

        level, id, counter, updated, getNbPoints(), getNbVoxels(), is_visible,
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

       if(cur_lhs->aabb != cur_rhs->aabb){
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



///////////////////////////////////////////////////////////////////////////////
////////////////////////// GLOBAL EXTERNAL VARIABLES //////////////////////////
///////////////////////////////////////////////////////////////////////////////

std::deque<std::shared_ptr<PointBatch>> GlobalVariables::batchesQueue = {};
std::deque<std::mutex> GlobalVariables::batchesQueueMutexes = {};
std::shared_ptr<vector<Point>> GlobalVariables::spilledPoints = {};
std::shared_ptr<vector<OctreeNode*>> GlobalVariables::spillingNodes = {};
std::shared_ptr<vector<Point>> GlobalVariables::backlogVoxels = {};
std::shared_ptr<vector<OctreeNode*>> GlobalVariables::backlogVoxelsNodes = {};


uint32_t GlobalVariables::elapsedFrames = 0;
uint64_t GlobalVariables::nbPoints = 0;
bool GlobalVariables::mainLoopIsTerminating = false;
std::mutex GlobalVariables::mainLoopIsTerminatingMtx;
uint64_t GlobalVariables::simLodOctreeCounter = 0;

/// Variables tracking when the octree can be sent to GPU
/// Initialized as not ready to be sent
std::binary_semaphore GlobalVariables::octreeReadyToBeSent{0};
/// Initialized as ready to be updated
std::binary_semaphore GlobalVariables::octreeReadyToBeUpdated{1};
/// Initialized as not being sent
std::binary_semaphore GlobalVariables::octreeNotBeingSent{1};

std::mutex GlobalVariables::isUpdatingMtx;

/// The queue of batches
std::mutex GlobalVariables::updateSceneMutex;

/// The main octree
std::shared_ptr<OctreeNode> GlobalVariables::mainOctree = nullptr;
OctreeNode* GlobalVariables::mainOctreeCpy = nullptr;

/// The LRU cache for the nodes
std::shared_ptr<LRUCache> GlobalVariables::updatesCache = nullptr;
std::shared_ptr<LRUCache> GlobalVariables::visibilityCache = nullptr;
std::unordered_map<AABB, std::array<std::optional<AABB>, 8>, AABB::Hash> GlobalVariables::aabbRelationshipMap = {};
std::mutex GlobalVariables::aabbRelationshipMapMtx;
std::unordered_map<AABB, AABB, AABB::Hash> GlobalVariables::aabbParentMap = {};
std::unordered_map<AABB, std::mutex, AABB::Hash> GlobalVariables::aabbMutexMap = {};

/// The global allocated memory (for batches)
BatchedMemory GlobalVariables::batchedMemory = {};

std::string GlobalVariables::getSimLodOctreeName(bool generate_new_name){
    if(generate_new_name){
        simLodOctreeCounter++;
    }
    std::string name = format("MainOctreeSimLOD_{}", simLodOctreeCounter);
    // println("Octree name: {}", name);
    return name;
}

void GlobalVariables::init(CuRast* instance, CUcontext* context){
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

    batchedMemory.init(instance, context);
}


void GlobalVariables::destroy(CuRast* instance, CUcontext* context){
    cudaDeviceSynchronize();

    {
		std::lock_guard<std::mutex> lock(GlobalVariables::mainLoopIsTerminatingMtx);
		GlobalVariables::mainLoopIsTerminating = true;
		// Destroy temporary folder
		std::filesystem::remove_all(OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY);
	}

    cudaDeviceSynchronize();
    batchedMemory.destroy();

    cudaDeviceSynchronize();
    if(mainOctreeCpy){
		delete(mainOctreeCpy);
		mainOctreeCpy = nullptr;
	}
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




vector<std::shared_ptr<Timing>> Timing::timingsList = {};
std::mutex Timing::timingsMtx;

std::shared_ptr<Timing> Timing::addTiming(string name, bool start_now, uint32_t level){
    if(!OocSimLodSettings::MEASURE_TIMINGS){
        return std::make_shared<Timing>(name, start_now, level);
    }
    std::shared_ptr<Timing> new_timing = std::make_shared<Timing>(name, start_now, level);
    std::lock_guard<std::mutex> lock(timingsMtx);
    timingsList.push_back(new_timing);
    return timingsList.back();
}

void displayTimings(){
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

void displayBuffers(){
	println("///////////////////////////////////////////////////");
	println("///////////////////// Buffers /////////////////////");
	println("///////////////////////////////////////////////////\n");
	
    uint32_t nb_empty = 0;
    uint32_t nb_to_load = 0;
    uint32_t nb_loaded = 0;
    uint32_t nb_inserted = 0;
    uint32_t nb_to_remove = 0;
    for(uint32_t i=0; i<OocSimLodSettings::BATCHES_LIST_SIZE; i++){
        if(!GlobalVariables::batchesQueue[i]){
            nb_empty++;
            continue;
        }
        switch(GlobalVariables::batchesQueue[i]->state){
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
    println("- spilledPoints: {} elements", (*GlobalVariables::spilledPoints).size());
    println("- spillingNodes: {} elements", (*GlobalVariables::spillingNodes).size());
    println("- backlogVoxels: {} elements", (*GlobalVariables::backlogVoxels).size());
    println("- backlogVoxelsNodes: {} elements", (*GlobalVariables::backlogVoxelsNodes).size());

    println("\n///////////////////////////////////////////////////");
	println("///////////////////////////////////////////////////");
	println("///////////////////////////////////////////////////\n");
}











///////////////////////////////////////////////////////////////////////////////
/////////////////////////// LRU CACHING SHENANIGANS ///////////////////////////
///////////////////////////////////////////////////////////////////////////////



std::mutex LRUCache::stored_set_mtx;
std::unordered_set<AABB, AABB::Hash> LRUCache::stored_set = {};
std::mutex LRUCache::caches_sync_mtx;

std::optional<AABB> LRUCache::add(const AABB& aabb){
    bool already_in_cache = false;

    // Reset every counters if needed
    if(counter == UINT64_MAX){
        println("Cache counter reseting");
        for(uint32_t cache_id = 0; cache_id < CACHE_SIZE; cache_id++){
            if(cache[cache_id]){
                CacheEntry& entry = cache[cache_id].value();
                entry.first = 0;
                if(entry.second == aabb){
                    entry.first = 1;
                    already_in_cache = true;
                }
            }
        }
        counter = 0;
    }
    counter++;
    if(already_in_cache){return nullopt;}

    // Check if already in cache
    uint32_t new_id = 0;
    uint64_t min_counter = UINT64_MAX;
    for(uint32_t cache_id = 0; cache_id < CACHE_SIZE; cache_id++){
        if(cache[cache_id]){
            CacheEntry& entry = cache[cache_id].value();

            // Check if already in cache
            if(entry.second == aabb){
                entry.first = counter;
                return nullopt;
            }

            // Check if smallest counter
            if(entry.first < min_counter){
                min_counter = entry.first;
                new_id = cache_id;
            }
        } else {
            // Found empty space
            cache[cache_id] = {counter, aabb};
            cache_map[aabb] = cache_id;
            return nullopt;
        }
    }

    // If not in cache, create new entry
    const AABB old_entry = cache[new_id]->second;
    cache[new_id] = {counter, aabb};
    cache_map[aabb] = new_id;
    cache_map.erase(old_entry);

    return std::optional<AABB>(old_entry);
}

std::optional<uint32_t> LRUCache::getIndex(const AABB& aabb) {
    return cache_map.contains(aabb) ? std::optional<uint32_t>(cache_map.at(aabb)) : nullopt;
}

bool LRUCache::contains(const AABB& aabb ) {
    return getIndex(aabb).has_value();
}

uint32_t LRUCache::getSize() const {
    uint32_t nb_elements = 0;
    for(auto& entry : cache){
        nb_elements += uint32_t(entry.has_value());
    }
    return nb_elements;
}


void LRUCache::display() {
    std::string pad = std::string(max(int32_t(name.size())-2, 0), '/');
    println("////////////////////////////////////////////////{}", pad);
	println("////////////////////// {} //////////////////////", name);
	println("////////////////////////////////////////////////{}\n", pad);
	for(const std::optional<CacheEntry>& entry : cache){
        if(entry.has_value()){
            std::string output = format("mins = ({}, {}, {}), maxs = ({}, {}, {})",
                entry->second.mins.x, 
                entry->second.mins.y, 
                entry->second.mins.z, 
                entry->second.maxs.x, 
                entry->second.maxs.y, 
                entry->second.maxs.z
            );
            println("- [ {} ]: {}", entry->first, output);
        } else {
            // println("- [ null ]");
        }
    }
	println("\n////////////////////////////////////////////////{}", pad);
    println("////////////////////////////////////////////////{}", pad);
	println("////////////////////////////////////////////////{}\n", pad);
}

void LRUCache::displayStored(){
    std::lock_guard<std::mutex> lock(stored_set_mtx);

    println("//////////////////////////////////////////////////////////");
	println("////////////////////// Stored Nodes //////////////////////");
	println("//////////////////////////////////////////////////////////\n");
	for(const AABB& aabb : stored_set){
        std::string output = format("mins = ({}, {}, {}), maxs = ({}, {}, {})",
            aabb.mins.x, 
            aabb.mins.y, 
            aabb.mins.z, 
            aabb.maxs.x, 
            aabb.maxs.y, 
            aabb.maxs.z
        );
        println("- {}", output);
    }
	println("\n//////////////////////////////////////////////////////////");
	println("//////////////////////////////////////////////////////////");
	println("//////////////////////////////////////////////////////////\n");
}


bool LRUCache::hasBeenStored(const AABB& aabb){
    std::lock_guard<std::mutex> lock(stored_set_mtx);
    return stored_set.contains(aabb);
}

void LRUCache::mark(const AABB& aabb){
    std::lock_guard<std::mutex> lock(stored_set_mtx);

    // // TODO: to remove
    // {
    //     std::string output = format("mins = ({}, {}, {}), maxs = ({}, {}, {})",
    //         aabb.mins.x, 
    //         aabb.mins.y, 
    //         aabb.mins.z, 
    //         aabb.maxs.x, 
    //         aabb.maxs.y, 
    //         aabb.maxs.z
    //     );
    //     if(stored_set.contains(aabb)){
    //         println("AABB: {} was already stored", output);
    //     } else {
    //         println("AABB: {} has been stored", output);
    //     }
    // }

    stored_set.insert(aabb);
}

void LRUCache::unmark(const AABB& aabb){
    std::lock_guard<std::mutex> lock(stored_set_mtx);

    // // TODO: to remove
    // {
    //     static std::unordered_set<AABB, AABB::Hash> tmp_loaded = {};
    //     std::string output = format("mins = ({}, {}, {}), maxs = ({}, {}, {})",
    //         aabb.mins.x, 
    //         aabb.mins.y, 
    //         aabb.mins.z, 
    //         aabb.maxs.x, 
    //         aabb.maxs.y, 
    //         aabb.maxs.z
    //     );
    //     if(tmp_loaded.contains(aabb)){
    //         println("AABB: {} was already loaded", output);
    //     } else {
    //         println("AABB: {} has been loaded", output);
    //     }
    //     tmp_loaded.insert(aabb);
    // }

    stored_set.erase(aabb);
}

bool LRUCache::isInACache(const AABB& aabb){
    return GlobalVariables::updatesCache->contains(aabb) 
        || GlobalVariables::visibilityCache->contains(aabb)
        // TODO: add other caches if necessary
    ;
}
bool LRUCache::isInAllCaches(const AABB& aabb){
    return GlobalVariables::updatesCache->contains(aabb) 
        && GlobalVariables::visibilityCache->contains(aabb)
        // TODO: add other caches if necessary
    ;
}


bool LRUCache::sanityCheck(const OctreeNode* root_node) {
    std::lock_guard<std::mutex> lock(caches_sync_mtx);
    if(!contains(root_node->aabb)){
        println("ERROR: cache should always contain the root node");
        return false;
    }

    std::unordered_set<AABB, AABB::Hash> correct = {};
    std::function<void(const AABB&)> recursion = [&](const AABB& cur_aabb){
        if(contains(cur_aabb)){
            correct.insert(cur_aabb);
            for(uint32_t child_id = 0; child_id < 8; child_id++){
                // TODO: temporary code
                if(GlobalVariables::aabbRelationshipMap[cur_aabb][child_id].has_value()){
                    recursion(GlobalVariables::aabbRelationshipMap[cur_aabb][child_id].value());
                }
            }
        }
    };
    recursion(root_node->aabb);

    uint32_t expected = getSize();
    uint32_t found = correct.size();
    if(found != expected){
        println("ERROR: invalid elements in cache, expected {} elements from the octree, found {}",
            expected, found
        );
        std::unordered_set<AABB, AABB::Hash> incorrect = {};
        for(auto& [aabb, id] : cache_map){
            if(!correct.contains(aabb)){
                incorrect.insert(aabb);
            }
        }
        assert(incorrect.size() + found == expected);
        println("Correct elements in cache:");
        for(const AABB& aabb : correct){
            println("    mins = ({}, {}, {}), maxs = ({}, {}, {})",
                aabb.mins.x, aabb.mins.y, aabb.mins.z,
                aabb.maxs.x, aabb.maxs.y, aabb.maxs.z
            );
        }
        println("");
        println("Incorrect elements in cache:");
        for(const AABB& aabb : incorrect){
            println("    mins = ({}, {}, {}), maxs = ({}, {}, {})",
                aabb.mins.x, aabb.mins.y, aabb.mins.z,
                aabb.maxs.x, aabb.maxs.y, aabb.maxs.z
            );
        }
        return false;
    };

    return true;
}

bool LRUCache::sanityCheckStored(const OctreeNode* root_node) {
    std::lock_guard<std::mutex> lock(stored_set_mtx);

    std::unordered_set<AABB, AABB::Hash> roots = {};
    std::function<void(const AABB&)> roots_recursion = [&](const AABB& cur_aabb){
        if(!stored_set.contains(cur_aabb)){
            for(uint32_t child_id = 0; child_id < 8; child_id++){
                // TODO: temporary code
                if(GlobalVariables::aabbRelationshipMap[cur_aabb][child_id].has_value()){
                    roots_recursion(GlobalVariables::aabbRelationshipMap[cur_aabb][child_id].value());
                }
            }
        } else {
            roots.insert(cur_aabb);
        }
    };
    roots_recursion(root_node->aabb);

    std::unordered_set<AABB, AABB::Hash> correct = {};
    std::function<void(const AABB&)> recursion = [&](const AABB& cur_aabb){
        if(stored_set.contains(cur_aabb)){
            correct.insert(cur_aabb);
            for(uint32_t child_id = 0; child_id < 8; child_id++){
                AABB child_aabb = AABB(cur_aabb);
                child_aabb.shrink((NodePosition)child_id);
                recursion(child_aabb);
            }
        }
    };
    for(const AABB& root : roots){
        recursion(root);
    }

    uint32_t expected = stored_set.size();
    uint32_t found = correct.size();
    if(found != expected){
        println("ERROR: invalid elements in stored cache, expected {} elements from the octree, found {} correct elements",
            expected, found
        );

        println("Roots found:");
        for(const AABB& aabb : roots){
            println("    mins = ({}, {}, {}), maxs = ({}, {}, {})",
                aabb.mins.x, aabb.mins.y, aabb.mins.z,
                aabb.maxs.x, aabb.maxs.y, aabb.maxs.z
            );
        }

        std::unordered_set<AABB, AABB::Hash> incorrect = {};
        for(const AABB& aabb : stored_set){
            if(!correct.contains(aabb)){
                incorrect.insert(aabb);
            }
        }
        assert(incorrect.size() + found == expected);
        println("Correct elements in stored cache:");
        for(const AABB& aabb : correct){
            println("    mins = ({}, {}, {}), maxs = ({}, {}, {})",
                aabb.mins.x, aabb.mins.y, aabb.mins.z,
                aabb.maxs.x, aabb.maxs.y, aabb.maxs.z
            );
        }
        println("");
        println("Incorrect elements in stored cache:");
        for(const AABB& aabb : incorrect){
            println("    mins = ({}, {}, {}), maxs = ({}, {}, {})",
                aabb.mins.x, aabb.mins.y, aabb.mins.z,
                aabb.maxs.x, aabb.maxs.y, aabb.maxs.z
            );
        }
        return false;
    };

    return true;
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
        exit(EXIT_FAILURE);
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
            println("level: {}, counter: {}, updated: {}, visibility: {}, children visibility: 0b{}{}{}{}{}{}{}{}, points location: 0b{}{}{}{}{}{}{}{}, children: 0b{}{}{}{}{}{}{}{}",
                node->level, node->counter, node->updated, node->is_visible,
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

    recursion(GlobalVariables::mainOctree.get(), 0);
}


CFullOctreeUnified CFullOctreeUnifiedBuilder::build() {
    CFullOctreeUnified res = {};
    res.nodes = (COctreeNodeUnified**)nodes.data();
    res.num_nodes = num_nodes;
    res.max_lod_level = max_lod_level;
    res.world = mat4(1.f);
    return res;
}