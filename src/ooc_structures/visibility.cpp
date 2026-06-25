#include "visibility.h"

#include "outOfCore.h"


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
    println("");
}


/// Get a list of all visible nodes that are either loaded or in cache
std::unordered_set<AABB, AABB::Hash> getVisibleNodes(const Frustum& frustum){
    std::unordered_set<AABB, AABB::Hash> res = {};

    {
        std::lock_guard<std::mutex> lock_cache(updatesCache->mtx);
        for(const auto& [aabb, _id] : updatesCache->cache_map){
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


/// Order the visible nodes from furthest to closest
/// All nodes in the list must appear only once
/// All nodes in the list must have their parent in the list
/// All nodes in the list must have their parent marked as closest
std::vector<AABB> orderNodes(
    const AABB& root_node,
    const std::unordered_set<AABB, AABB::Hash>& visible_nodes,
    const vec3& camera_pos
){
    uint32_t size = visible_nodes.size();

    // Get root node
    if(!visible_nodes.empty() && !visible_nodes.contains(root_node)){
        println("ERROR: the root node should always be marked as visible");
        exit(EXIT_FAILURE);
    }

    // println("before ordering: vis cache size = {}, updates cache size = {}, stored nodes = {}, nb_nodes = {}", 
    //     visibilityCache->getSize(), updatesCache->getSize(), LRUCache::stored_set.size(), size
    // );

    // // TODO: can we do better than O(n2) ?
    // std::vector<AABB> sorted = std::vector<AABB>(visible_nodes.begin(), visible_nodes.end());
    // // Sort the visible nodes by furthest to camera position
    // auto comparison_dist = [&](const AABB& a, const AABB& b) -> bool {
    //     float dist_a = glm::length(a.getCentroid() - camera_pos);
    //     float dist_b = glm::length(b.getCentroid() - camera_pos);
    //     return dist_a > dist_b;
    // };
    // std::sort(sorted.begin(), sorted.end(), comparison_dist);
    // std::vector<AABB> res = sorted;
    // for(uint32_t i=0; i<size; i++){
    //     uint32_t last_child_index = 0;

    //     do {
    //         last_child_index = i;
    //         AABB current_node = res[i];

    //         // Try to find a child marked as closer
    //         for(uint32_t j=i+1; j<size; j++){
    //             // If is a child of current node
    //             if(current_node.isParentOf(res[j])){
    //                 last_child_index = j;
    //             }
    //         }

    //         // Update the array by swapping elements up to the child found
    //         for(uint32_t j=i; j<last_child_index; j++){
    //             res[j] = res[j+1];
    //         }
    //         res[last_child_index] = current_node;
    //     } while(last_child_index != i);
    // }


    // TODO: better than O(n2) but less accurate
    // Build a temporary octree
    struct Node {
        const AABB aabb;
        float dist = INFINITY;
        std::vector<std::shared_ptr<Node>> children = {};

        Node(const AABB& aabb, float dist): aabb(aabb), dist(dist) {
            children.reserve(8);
        }
    };
    std::vector<AABB> res = {};
    res.reserve(size);
    std::function<void(std::shared_ptr<Node>)> recursion = [&](std::shared_ptr<Node> cur_node){
        for(uint32_t child_id = 0; child_id < 8; child_id++){
            // AABB child_aabb = AABB(cur_node->aabb);
            // child_aabb.shrink((NodePosition)child_id);

            // TODO: temporary code
            AABB child_aabb = {};
            {
                std::lock_guard<std::mutex> lock(aabb_relationship_map_mtx);
                if(!aabb_relationship_map[cur_node->aabb][child_id].has_value()){
                    continue;
                }
                child_aabb = aabb_relationship_map[cur_node->aabb][child_id].value();
            }

            if(visible_nodes.contains(child_aabb)){
                float dist = glm::length(child_aabb.getCentroid() - camera_pos);
                cur_node->children.push_back(std::make_shared<Node>(child_aabb, dist));
            }
        }

        std::sort(cur_node->children.begin(), cur_node->children.end(), 
        [](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
            return a->dist > b->dist;
        });
        // println("children:");
        // for(auto& child : cur_node->children){
        //     println("    .mins = ({}, {}, {}), .maxs = ({}, {}, {})",
        //         child->aabb.mins.x, child->aabb.mins.y, child->aabb.mins.z,
        //         child->aabb.maxs.x, child->aabb.maxs.y, child->aabb.maxs.z
        //     );
        // }

        for(auto child : cur_node->children){
            recursion(child);
        }

        res.push_back(cur_node->aabb);
    };
    float root_dist = glm::length(root_node.getCentroid() - camera_pos);
    std::shared_ptr<Node> root = std::make_shared<Node>(root_node, root_dist);
    recursion(root);

    // println("after ordering: vis cache size = {}, updates cache size = {}, stored nodes = {}, nb_nodes = {}", 
    //     visibilityCache->getSize(), updatesCache->getSize(), LRUCache::stored_set.size(), res.size()
    // );

    return res;
}



/// Fill the visibility cache with the ordered nodes
void fillVisibilityCache(const std::vector<AABB>& nodes, OctreeNode* root_octree){
    // println("before add: vis cache size = {}, updates cache size = {}, stored nodes = {}, nb visible nodes = {}, total nb nodes = {}", 
    //     visibilityCache->getSize(), updatesCache->getSize(), LRUCache::stored_set.size(), nodes.size(), aabb_relationship_map.size()
    // );
    
    // uint32_t first_index = 0;
    uint32_t first_index = uint32_t(max(int32_t(nodes.size()) - int32_t(LRU_VISIBILITY_CACHE_SIZE), 0));
    uint32_t last_index = min(first_index + LRU_VISIBILITY_CACHE_SIZE, uint32_t(nodes.size()));

    for(uint32_t i = first_index; i<last_index; i++){
        visibilityCache->add(nodes[i], true);
    }

    // println("after add: vis cache size = {}, updates cache size = {}, stored nodes = {}, nb visible nodes = {}, total nb nodes = {}", 
    //     visibilityCache->getSize(), updatesCache->getSize(), LRUCache::stored_set.size(), nodes.size(), aabb_relationship_map.size()
    // );

    // println("");
    // println("");
    // updatesCache->display(true);
    // println("");
    // println("");
    // visibilityCache->display(true);
    // println("");
    // println("");

    // LRUCache::displayStored();
    // // println("Octree before: ");
    // // root_octree->display();
    // println("");
    // println("");

    // println("//////////////////////////////////////////////////////////");
	// println("///////////////////// Ordered stored /////////////////////");
	// println("//////////////////////////////////////////////////////////\n");
	// for(const AABB& aabb : nodes){
    //     std::string output = format("mins = ({}, {}, {}), maxs = ({}, {}, {})",
    //         aabb.mins.x, 
    //         aabb.mins.y, 
    //         aabb.mins.z, 
    //         aabb.maxs.x, 
    //         aabb.maxs.y, 
    //         aabb.maxs.z
    //     );
    //     println("- {}", output);
    // }
	// println("\n//////////////////////////////////////////////////////////");
	// println("//////////////////////////////////////////////////////////");
	// println("//////////////////////////////////////////////////////////\n");

    // if(!updatesCache->sanityCheck(root_octree)){
    //     println("Sanity check failed for the updates cache");
    //     exit(EXIT_FAILURE);
    // }
    // if(!visibilityCache->sanityCheck(root_octree)){
    //     println("Sanity check failed for the visibility cache");
    //     exit(EXIT_FAILURE);
    // }
    

    // Remove all nodes that are not in any of the other caches
    // No need to store them as if they were not in the updates cache they were not updated since last load
    // Also load all the nodes that need to be loaded
    std::function<bool(OctreeNode*, uint32_t, uint32_t, bool*)> recursion = [&](OctreeNode* cur_node, uint32_t id, uint32_t level, bool* is_visible) -> bool {
        const AABB& aabb = *cur_node->aabb;
        bool in_vis_cache = visibilityCache->contains(aabb, true);
        
        if(!CuRastSettings::freezeVisibleNodes){
            *is_visible = in_vis_cache;
            cur_node->is_visible = in_vis_cache;
            cur_node->children_visibility = 0b00000000;
        }

        for(uint32_t child_id = 0; child_id < 8; child_id++){
            bool child_is_visible = false;

            if(cur_node->children[child_id]){
                if(recursion(cur_node->children[child_id], child_id, level+1, &child_is_visible)){
                    delete(cur_node->children[child_id]);
                    cur_node->children[child_id] = nullptr;
                }
            } else {
                // AABB child_aabb = AABB(aabb);
                // child_aabb.shrink((NodePosition)child_id);

                // TODO: temporary code
                AABB child_aabb = {};
                {
                    std::lock_guard<std::mutex> lock(aabb_relationship_map_mtx);
                    if(!aabb_relationship_map[*cur_node->aabb][child_id].has_value()){
                        continue;
                    }
                    child_aabb = aabb_relationship_map[*cur_node->aabb][child_id].value();
                }

                if(visibilityCache->contains(child_aabb, true) && LRUCache::hasBeenStored(child_aabb)){
                    cur_node->children[child_id] = loadOctree(child_aabb, true);
                    recursion(cur_node->children[child_id], child_id, level+1, &child_is_visible);
                }
            }

            cur_node->children_visibility |= (uint32_t(child_is_visible) << child_id);

        }

        return !in_vis_cache && !updatesCache->contains(aabb, true) ;
    };

    // std::lock_guard<std::mutex> lock_test(updatesCache->mtx);
    bool root_visible = false;
    recursion(root_octree, 0, 0, &root_visible);
    if(!root_visible){
        println("Root should alwasy be visible");
        exit(EXIT_FAILURE);
    }

    // println("before add: vis cache size = {}, updates cache size = {}, stored nodes = {}, nb visible nodes = {}, total nb nodes = {}", 
    //     visibilityCache->getSize(), updatesCache->getSize(), LRUCache::stored_set.size(), nodes.size(), aabb_relationship_map.size()
    // );
    // println("");
}


void updateVisibilityCache(const mat4& view, const mat4& proj){
    // TODO: just for debugging
    static bool was_freezed = false;
    bool just_freezed = false;
    if(!was_freezed && CuRastSettings::freezeVisibleNodes){
        was_freezed = true;
        just_freezed = true;
    }
    if(was_freezed && !CuRastSettings::freezeVisibleNodes){
        was_freezed = false;
    }


    std::lock_guard<std::mutex> lock(LRUCache::test_mtx);


    std::shared_ptr<OctreeNode> octree_ref = nullptr;
	if(CPU_PARALLELISED){
		std::lock_guard<std::mutex> lock_send(isUpdatingMtx);
		octree_ref = mainOctree;
	} else {
		octree_ref = mainOctree;
	}
	if(!octree_ref){return;}


    // // TODO: to remove
    // {
    //     uint32_t cpt = 0;
    //     for(auto& [aabb, _id] : updatesCache->cache_map){
    //         if(LRUCache::hasBeenStored(aabb)){
    //             cpt++;
    //         }
    //     }
    //     println("{} nodes are both stored and in cache", cpt);
    // }


    Frustum frustum = Frustum(proj * view);
    // frustum.display();

    std::unordered_set<AABB, AABB::Hash> visible_nodes = getVisibleNodes(frustum);

    // // TODO: just for debugging
    // println("visible nodes:");
    // for(const AABB& aabb : visible_nodes){
    //     println("    .mins = ({}, {}, {}), .maxs = ({}, {}, {})",
    //         aabb.mins.x, aabb.mins.y, aabb.mins.z,
    //         aabb.maxs.x, aabb.maxs.y, aabb.maxs.z
    //     );
    // }
    // println("");

    // {
    //     // TODO: to remove
    //     std::unordered_set<AABB, AABB::Hash> invisible_nodes = {};
    //     invisible_nodes.reserve(LRU_CACHE_SIZE);
    //     {
    //         std::lock_guard<std::mutex> lock_cache(updatesCache->mtx);
    //         for(const auto& [aabb, _id] : updatesCache->cache_map){
    //             if(!frustum.doesIntersect(aabb)){
    //                 invisible_nodes.insert(aabb);
    //             }
    //         }
    //     }
    //     {
    //         std::lock_guard<std::mutex> lock_cache(LRUCache::stored_set_mtx);
    //         for(const AABB& aabb : LRUCache::stored_set){
    //             if(!frustum.doesIntersect(aabb)){
    //                 invisible_nodes.insert(aabb);
    //             }
    //         }
    //     }
    //     println("not visible nodes:");
    //     for(const AABB& aabb : invisible_nodes){
    //         println("    .mins = ({}, {}, {}), .maxs = ({}, {}, {})",
    //             aabb.mins.x, aabb.mins.y, aabb.mins.z,
    //             aabb.maxs.x, aabb.maxs.y, aabb.maxs.z
    //         );
    //     }
    //     println("");
    // }


    vec3 cameraPos = vec3(glm::inverse(view) * vec4(0.0f, 0.0f, 0.0f, 1.0f));
    std::vector<AABB> ordered_nodes = orderNodes(*octree_ref->aabb, visible_nodes, cameraPos);

    // // TODO: just for debugging
    // if(just_freezed){
    //     println("ordered visible nodes:");
    //     for(const AABB& aabb : ordered_nodes){
    //         println("    .mins = ({}, {}, {}), .maxs = ({}, {}, {})",
    //             aabb.mins.x, aabb.mins.y, aabb.mins.z,
    //             aabb.maxs.x, aabb.maxs.y, aabb.maxs.z
    //         );
    //     }
    //     println("");
    // }


    std::lock_guard<std::mutex> lock_send(isUpdatingMtx);
    fillVisibilityCache(ordered_nodes, octree_ref.get());

    // if(just_freezed){
    //     updatesCache->display(true);
    //     visibilityCache->display(true);
    // }

    // exit(EXIT_FAILURE);
}