#include "globals.h"

uint32_t NbLoadedClouds = 0;
vector<Timing> timingsList = {};

void displayTimings(){
	println("///////////////////////////////////////////////////");
	println("///////////////////// Timings /////////////////////");
	println("///////////////////////////////////////////////////\n");
	for (Timing& timing : timingsList){
		uint64_t us = timing.duration.count();
		uint64_t s = uint64_t(us / 1'000'000);
		uint64_t ms = uint64_t((us - (s*1'000'000)) / 1'000);
		us = us - (s*1'000'000) - (ms*1'000);

        string tab = std::string(4*timing.level, ' ');

		println("{}- {}:{}s, {}ms, {}us",
			tab, timing.name, s, ms, us 
		);
	}
    println("\n///////////////////////////////////////////////////");
	println("///////////////////////////////////////////////////");
	println("///////////////////////////////////////////////////\n");
};

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
    println("id: {}, level: {}, counter: {}, nbPoints: {}, nbVoxels: {}, children: 0b{}{}{}{}{}{}{}{}",
        // id, level, counter.load(), getNbPoints(), getNbVoxels(),
        id, level, counter, getNbPoints(), getNbVoxels(),
        uint8_t(bool(children_ids & 0x01 << 0)),
        uint8_t(bool(children_ids & 0x01 << 1)),
        uint8_t(bool(children_ids & 0x01 << 2)),
        uint8_t(bool(children_ids & 0x01 << 3)),
        uint8_t(bool(children_ids & 0x01 << 4)),
        uint8_t(bool(children_ids & 0x01 << 5)),
        uint8_t(bool(children_ids & 0x01 << 6)),
        uint8_t(bool(children_ids & 0x01 << 7))
    );
    if(!node_only){
        for(size_t i=0; i<8; i++){
            if(children[i]){
                children[i]->display(i, level+1);
            }
        }
    }
};