#include "visibility.h"

Plane::Plane(float x, float y, float z, float w){
    float normal_length = glm::length(vec3{x, y, z});
    normal = vec3{x, y, z} / normal_length;
    constant = w / normal_length;
}


Frustum::Frustum(const mat4& view_proj){

    // const mat4 transpose = glm::transpose(view_proj);
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

    planes[0] = Plane(m_03 - m_00, m_13 - m_10, m_23 - m_20, m_33 - m_30);
    planes[1] = Plane(m_03 + m_00, m_13 + m_10, m_23 + m_20, m_33 + m_30);
    planes[2] = Plane(m_03 + m_01, m_13 + m_11, m_23 + m_21, m_33 + m_31);
    planes[3] = Plane(m_03 - m_01, m_13 - m_11, m_23 - m_21, m_33 - m_31);
    planes[4] = Plane(m_03 - m_02, m_13 - m_12, m_23 - m_22, m_33 - m_32);
    planes[5] = Plane(m_03 + m_02, m_13 + m_12, m_23 + m_22, m_33 + m_32);
}

bool Frustum::doesIntersect(const AABB& aabb) const {
    for(uint32_t i = 0; i < 6; i++){
		vec3 vector = {
		    planes[i].normal.x > 0.0 ? aabb.maxs.x : aabb.mins.x,
		    planes[i].normal.y > 0.0 ? aabb.maxs.y : aabb.mins.y,
		    planes[i].normal.z > 0.0 ? aabb.maxs.z : aabb.mins.z
        };

		float d = glm::dot(planes[i].normal, vector) + planes[i].constant;
		if(d < 0){return false;}
	}

	return true;
}

void Frustum::display() const {
    println("Frustum:");
    for(uint32_t i=0; i<6; i++){
        println("Plane: normal = ({}, {}, {}), constant = {}",
            planes[i].normal.x, planes[i].normal.y, planes[i].normal.z, planes[i].constant
        );
    }
    println();
}


/// Get a list of all visible nodes that are either loaded or in cache
std::unordered_set<AABB, AABB::Hash> getVisibleNodes(const Frustum& frustum){
    std::unordered_set<AABB, AABB::Hash> res = {};
    res.reserve(LRU_CACHE_SIZE);

    {
        std::lock_guard<std::mutex> lock_cache(updatesCache.mtx);
        for(const auto& [aabb, _id] : updatesCache.cache_map){
            if(frustum.doesIntersect(aabb)){
                res.insert(aabb);
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock_cache(LRUCache::stored_set_mtx);
        for(const AABB& aabb : LRUCache::stored_set){
            if(frustum.doesIntersect(aabb)){
                res.insert(aabb);
            }
        }
    }

    return res;
}


void updateVisibilityCache(const mat4& view, const mat4& proj){
    Frustum frustum = Frustum(proj * view);
    // frustum.display();

    std::unordered_set<AABB, AABB::Hash> visible_nodes = getVisibleNodes(frustum);
    
    // Flag the visible nodes
    std::function<void(OctreeNode*)> recursion = [&](OctreeNode* cur_node){
        if(!CuRastSettings::freezeVisibleNodes){
            cur_node->is_visible = visible_nodes.contains(*cur_node->aabb);
        }

        for(uint32_t child=0; child<8; child++){
            if(cur_node->children[child]){
                recursion(cur_node->children[child]);
            }
        }
    };

    if(mainOctree){
        recursion(mainOctree.get());
    }
}