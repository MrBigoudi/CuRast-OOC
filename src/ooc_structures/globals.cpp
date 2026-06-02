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
    return position.x >= mins.x && position.x <= maxs.x
        && position.y >= mins.y && position.y <= maxs.y
        && position.z >= mins.z && position.z <= maxs.z
    ;
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
    shared_ptr<Chunk> point_chunk = points;
    while(point_chunk){
        res += point_chunk->size;
        point_chunk = point_chunk->next;
    }
    return res;
}

uint32_t OctreeNode::getNbVoxels() const {
    uint32_t res = 0;
    shared_ptr<Chunk> voxel_chunk = voxels;
    while(voxel_chunk){
        res += voxel_chunk->size;
        voxel_chunk = voxel_chunk->next;
    }
    return res;
}

void OctreeNode::display(uint32_t id, uint32_t level, bool node_only) const {
    println("id: {}, level: {}, counter: {}, nbPoints: {}, nbVoxels: {}, points location: 0b{}{}{}{}{}{}{}{}, children: 0b{}{}{}{}{}{}{}{}",
        id, level, counter, getNbPoints(), getNbVoxels(),
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
    if(!node_only){
        for(size_t i=0; i<8; i++){
            if(children[i]){
                children[i]->display(i, level+1);
            }
        }
    }
};

bool OctreeNode::operator==(const OctreeNode& rhs) const {

    std::function<bool(const OctreeNode&, const OctreeNode&)> recursion = [&](const OctreeNode& cur_lhs, const OctreeNode& cur_rhs) -> bool{
        // if(!)
        
        if(cur_lhs.counter != cur_rhs.counter){return false;}
        if(cur_lhs.children_ids != cur_rhs.children_ids){return false;}
        
        const Chunk& lhs_points = *cur_lhs.points.get();
        const Chunk& rhs_points = *cur_rhs.points.get();
        if(lhs_points != rhs_points){return false;}

        const Chunk& lhs_voxels = *cur_lhs.voxels.get();
        const Chunk& rhs_voxels = *cur_rhs.voxels.get();
        if(lhs_voxels != rhs_voxels){return false;}

        return true;
    };

    return recursion(*this, rhs);
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
        lhs_chunk = lhs_chunk->next.get();
        rhs_chunk = rhs_chunk->next.get();
    }
    if(rhs_chunk){return false;}
    return true;
}



std::string getSimLodOctreeName(bool generate_new_name){
    if(generate_new_name){simLodOctreeCounter++;}
    return format("{}_{}", simLodOctreeName, simLodOctreeCounter);
}


///////////////////////////////////////////////////////////////////////////////
////////////////////////////// GLOBAL VARIABLES ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

vector<std::shared_ptr<Timing>> timingsList = {};

/// Variables tracking when the octree can be sent to GPU
/// Initialized as not ready to be sent
std::binary_semaphore octreeReadyToBeSent{0};
/// Initialized as ready to be updated
std::binary_semaphore octreeReadyToBeUpdated{1};

/// The batches that still need to be read
std::deque<PointBatch> batchesToLoad = {};
mutex batchesToLoadMutex;
/// The batches that are already loaded
std::deque<PointBatch> batchesLoaded = {};
mutex batchesLoadedMutex;
/// The batches that have already been inserted in the octree
std::deque<PointBatch> batchesInserted = {};
mutex batchesInsertedMutex;

/// The main octree
std::shared_ptr<OctreeNode> mainOctree = std::make_shared<OctreeNode>();
/// The main bounding box
std::shared_ptr<AABB> mainAABB = nullptr;

/// The buffer of spilled points
std::shared_ptr<vector<Point>> spilledPoints = std::make_shared<vector<Point>>(vector<Point>());
/// The buffer of spilling nodes
std::shared_ptr<vector<OctreeNode*>> spillingNodes = std::make_shared<vector<OctreeNode*>>(vector<OctreeNode*>());

/// The backlog buffer for new voxels
std::shared_ptr<vector<Point>> backlogVoxels = std::make_shared<vector<Point>>(vector<Point>());
/// The backlog buffer for the nodes corresponding to the new voxels
std::shared_ptr<vector<OctreeNode*>> backlogVoxelsNodes = std::make_shared<vector<OctreeNode*>>(vector<OctreeNode*>());


std::shared_ptr<Timing> addTiming(string name, bool start_now, uint32_t level){
    std::shared_ptr<Timing> new_timing = std::make_shared<Timing>(name, start_now, level);
    std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    timingsList.push_back(new_timing);
    return timingsList.back();
}

void displayTimings(){
	println("///////////////////////////////////////////////////");
	println("///////////////////// Timings /////////////////////");
	println("///////////////////////////////////////////////////\n");
	for (auto& timing : timingsList){
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
};

void displayBuffers(){
	println("///////////////////////////////////////////////////");
	println("///////////////////// Buffers /////////////////////");
	println("///////////////////////////////////////////////////\n");
	
    println("- batchesToLoad: {} elements", batchesToLoad.size());
    println("- batchesLoaded: {} elements", batchesLoaded.size());
    println("- batchesInserted: {} elements", batchesInserted.size());
    println("- spilledPoints: {} elements", (*spilledPoints).size());
    println("- spillingNodes: {} elements", (*spillingNodes).size());
    println("- backlogVoxels: {} elements", (*backlogVoxels).size());
    println("- backlogVoxelsNodes: {} elements", (*backlogVoxelsNodes).size());

    println("\n///////////////////////////////////////////////////");
	println("///////////////////////////////////////////////////");
	println("///////////////////////////////////////////////////\n");
};
