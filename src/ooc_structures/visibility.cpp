#include "visibility.h"

#include "outOfCore.h"
#include "allocator.h"


Plane::Plane(float x, float y, float z, float w){
    float normal_length = glm::length(vec3{x, y, z});
    normal = vec3{x, y, z} / normal_length;
    constant = w / normal_length;
}


Frustum::Frustum(const mat4& view_proj){
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
std::unordered_set<IdAABB> Visibility::getVisibleNodes(
    const Frustum& frustum,
    std::shared_ptr<AABBRelationshipMap> relationship_map_ref
){
    std::unordered_set<IdAABB> res = {};
    res.reserve(relationship_map_ref->size);

    // println("before get nodes: vis cache size = {}, updates cache size = {}, total nodes = {}, nb_nodes = {}", 
    //     GlobalVariables::visibilityCache->getSize(), GlobalVariables::updatesCache->getSize(), 
    //     relationship_map_ref->size(), res.size()
    // );

    relationship_map_ref->mapWithKey([&](IdAABB aabb_index, std::array<IdAABB, 8>& children){
        const AABB& aabb = GlobalVariables::getAABB(aabb_index);
        if(frustum.doesIntersect(aabb)){
            res.insert(aabb_index);
        }
    });

    // println("after get nodes: vis cache size = {}, updates cache size = {}, total nodes = {}, nb_nodes = {}\n", 
    //     GlobalVariables::visibilityCache->getSize(), GlobalVariables::updatesCache->getSize(), 
    //     relationship_map_ref->size(), res.size()
    // );

    return res;
}


/// Order the visible nodes from furthest to closest
/// All nodes in the list must appear only once
/// All nodes in the list must have their parent in the list
/// All nodes in the list must have their parent marked as closest
std::vector<IdAABB> Visibility::orderNodes(
    const IdAABB& root_node,
    const std::unordered_set<IdAABB>& visible_nodes,
    const vec3& camera_pos,
    std::shared_ptr<AABBRelationshipMap> relationship_map_ref
){
    uint32_t size = visible_nodes.size();

    // Get root node
    if(!visible_nodes.empty() && !visible_nodes.contains(root_node)){
        println("ERROR: the root node should always be marked as visible");
        throw(EXIT_FAILURE);
    }

    // TODO: better than O(n2) but less accurate
    // Build a temporary octree
    struct Node {
        const IdAABB aabb_index;
        float dist = INFINITY;
        std::vector<std::shared_ptr<Node>> children = {};

        Node(const IdAABB& aabb_index, float dist): aabb_index(aabb_index), dist(dist) {
            children.reserve(8);
        }
    };
    std::vector<IdAABB> res = {};
    res.reserve(size);

    // println("before ordering: vis cache size = {}, updates cache size = {}, total nodes = {}, nb_nodes = {}", 
    //     GlobalVariables::visibilityCache->getSize(), GlobalVariables::updatesCache->getSize(), 
    //     relationship_map_ref->size(), res.size()
    // );

    std::function<void(std::shared_ptr<Node>)> recursion = [&](std::shared_ptr<Node> cur_node){
        if(!relationship_map_ref->contains(cur_node->aabb_index)){
            println("WTFF");

            println("Relationship map: ");
            relationship_map_ref->mapWithKey([&](IdAABB id, std::array<IdAABB, 8>& children){
                println("    - [{}]: children: [{}, {}, {}, {}, {}, {}, {}, {}]",
                    id,
                    children[0] == INVALID_ID ? -1 : int32_t(children[0]),
                    children[1] == INVALID_ID ? -1 : int32_t(children[1]),
                    children[2] == INVALID_ID ? -1 : int32_t(children[2]),
                    children[3] == INVALID_ID ? -1 : int32_t(children[3]),
                    children[4] == INVALID_ID ? -1 : int32_t(children[4]),
                    children[5] == INVALID_ID ? -1 : int32_t(children[5]),
                    children[6] == INVALID_ID ? -1 : int32_t(children[6]),
                    children[7] == INVALID_ID ? -1 : int32_t(children[7])
                );
            });
            println("\n\n\n");

            println("All AABBs");
            for(uint32_t i=0; i<GlobalVariables::allAABBs.size(); i++){
                AABB& aabb = GlobalVariables::allAABBs[i];
                println("    - [{}]: .mins = ({}, {}, {}), .maxs = ({}, {}, {})", i,
                    aabb.mins.x, aabb.mins.y, aabb.mins.z,
                    aabb.maxs.x, aabb.maxs.y, aabb.maxs.z
                );
            }
            println("\n\n\n");
            
            throw(EXIT_FAILURE);
        }


        for(uint32_t child_id = 0; child_id < 8; child_id++){
            IdAABB child_aabb_index = (*relationship_map_ref)[cur_node->aabb_index][child_id];
            if(child_aabb_index == INVALID_ID){continue;}

            if(visible_nodes.contains(child_aabb_index)){
                const AABB& aabb = GlobalVariables::getAABB(child_aabb_index);
                float dist = glm::length(aabb.getCentroid() - camera_pos);
                cur_node->children.push_back(std::make_shared<Node>(child_aabb_index, dist));
            }
        }

        std::sort(cur_node->children.begin(), cur_node->children.end(), 
        [](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
            return a->dist > b->dist;
        });

        for(auto child : cur_node->children){
            recursion(child);
        }

        res.push_back(cur_node->aabb_index);
    };

    const AABB& root_aabb = GlobalVariables::getAABB(root_node);
    float root_dist = glm::length(root_aabb.getCentroid() - camera_pos);
    std::shared_ptr<Node> root = std::make_shared<Node>(root_node, root_dist);
    recursion(root);

    // println("after ordering: vis cache size = {}, updates cache size = {}, total nodes = {}, nb_nodes = {}\n", 
    //     GlobalVariables::visibilityCache->getSize(), GlobalVariables::updatesCache->getSize(), 
    //     relationship_map_ref->size(), res.size()
    // );

    return res;
}



/// Fill the visibility cache with the ordered nodes
void Visibility::fillVisibilityCache(
    const std::vector<IdAABB>& nodes, 
    OctreeNode* root_octree,
    std::shared_ptr<AABBRelationshipMap> relationship_map_ref
){
    std::lock_guard<std::mutex> lock(LRUCache::caches_sync_mtx);

    // println("before filling cache: vis cache size = {}, updates cache size = {}, total nodes = {}, nb_nodes = {}", 
    //     GlobalVariables::visibilityCache->getSize(), GlobalVariables::updatesCache->getSize(), 
    //     relationship_map_ref->size(), nodes.size()
    // );

    uint32_t first_index = uint32_t(max(int32_t(nodes.size()) - int32_t(GlobalVariables::visibilityCache->CACHE_SIZE), 0));
    uint32_t last_index = min(first_index + GlobalVariables::visibilityCache->CACHE_SIZE, uint32_t(nodes.size()));

    for(uint32_t i = first_index; i<last_index; i++){
        GlobalVariables::visibilityCache->add(nodes[i]);
    }

    // Remove all nodes that are not in any of the other caches
    // No need to store them as if they were not in the updates cache they were not updated since last load
    // Also load all the nodes that need to be loaded
    std::function<bool(OctreeNode*, uint32_t, uint32_t, bool*)> recursion = [&](OctreeNode* cur_node, uint32_t id, uint32_t level, bool* is_visible) -> bool {
        const IdAABB& aabb_index = cur_node->aabb_index;
        bool in_vis_cache = GlobalVariables::visibilityCache->contains(aabb_index);
        
        if(!CuRastSettings::freezeVisibleNodes){
            *is_visible = in_vis_cache;
            cur_node->is_visible = in_vis_cache;
            cur_node->children_visibility = 0b00000000;
        }

        if(!relationship_map_ref->contains(cur_node->aabb_index)){
            println("This should not happen, at this stage all nodes in the octree must be in the relationship_map");

            println("Cur node:");
            cur_node->display(id, level, true);

            println("Relationship map: ");
            relationship_map_ref->mapWithKey([&](IdAABB id, std::array<IdAABB, 8>& children){
                println("    - [{}]: children: [{}, {}, {}, {}, {}, {}, {}, {}]",
                    id,
                    children[0] == INVALID_ID ? -1 : int32_t(children[0]),
                    children[1] == INVALID_ID ? -1 : int32_t(children[1]),
                    children[2] == INVALID_ID ? -1 : int32_t(children[2]),
                    children[3] == INVALID_ID ? -1 : int32_t(children[3]),
                    children[4] == INVALID_ID ? -1 : int32_t(children[4]),
                    children[5] == INVALID_ID ? -1 : int32_t(children[5]),
                    children[6] == INVALID_ID ? -1 : int32_t(children[6]),
                    children[7] == INVALID_ID ? -1 : int32_t(children[7])
                );
            });
            println("\n\n\n");

            println("All AABBs");
            for(uint32_t i=0; i<GlobalVariables::allAABBs.size(); i++){
                AABB& aabb = GlobalVariables::allAABBs[i];
                println("    - [{}]: .mins = ({}, {}, {}), .maxs = ({}, {}, {})", i,
                    aabb.mins.x, aabb.mins.y, aabb.mins.z,
                    aabb.maxs.x, aabb.maxs.y, aabb.maxs.z
                );
            }
            println("\n\n\n");

            println("Current octree:");
            root_octree->display();
            
            throw(EXIT_FAILURE);
        }


        for(uint32_t child_id = 0; child_id < 8; child_id++){
            bool child_is_visible = false;

            // Check if the node was already in memory
            if(cur_node->children[child_id]){
                if(recursion(cur_node->children[child_id], child_id, level+1, &child_is_visible)){
                    // Remove the node if it is not on any of the caches
                    // delete(cur_node->children[child_id]);
                    MemoryAllocator::delOctreeNode(cur_node->children[child_id]);
                    cur_node->children[child_id] = nullptr;
                }
            } else {
                // If the node is not in memory, load it
                IdAABB child_aabb_index = (*relationship_map_ref)[cur_node->aabb_index][child_id];
                if(child_aabb_index == INVALID_ID){continue;}

                if(GlobalVariables::visibilityCache->contains(child_aabb_index)){
                    cur_node->children[child_id] = loadOctree(child_aabb_index);
                    recursion(cur_node->children[child_id], child_id, level+1, &child_is_visible);
                }
            }

            cur_node->children_visibility |= (uint32_t(child_is_visible) << child_id);

        }

        return !in_vis_cache && !GlobalVariables::updatesCache->contains(aabb_index);
    };

    bool root_visible = false;
    recursion(root_octree, 0, 0, &root_visible);
    if(!root_visible){
        println("Root should always be visible: ie, should always be in the cache");
        throw(EXIT_FAILURE);
    }

    // println("after filling cache: vis cache size = {}, updates cache size = {}, total nodes = {}, nb_nodes = {}\n\n", 
    //     GlobalVariables::visibilityCache->getSize(), GlobalVariables::updatesCache->getSize(), 
    //     relationship_map_ref->size(), nodes.size()
    // );
}


bool Visibility::updateVisibilityCache(
    const mat4& view, const mat4& proj, 
    OctreeNode* octree_ref,
    std::shared_ptr<AABBRelationshipMap> relationship_map_ref
){
    if(!octree_ref){return false;}
    
    // TODO: just for debugging
    {
        static bool was_freezed = false;
        bool just_freezed = false;
        if(!was_freezed && CuRastSettings::freezeVisibleNodes){
            was_freezed = true;
            just_freezed = true;
        }
        if(was_freezed && !CuRastSettings::freezeVisibleNodes){
            was_freezed = false;
        }
    }

    Frustum frustum = Frustum(proj * view);

    std::unordered_set<IdAABB> visible_nodes = getVisibleNodes(frustum, relationship_map_ref);

    vec3 cameraPos = vec3(glm::inverse(view) * vec4(0.0f, 0.0f, 0.0f, 1.0f));
    std::vector<IdAABB> ordered_nodes = orderNodes(octree_ref->aabb_index, visible_nodes, cameraPos, relationship_map_ref);

    static std::vector<IdAABB> oldOrderedNodes = {};
    if(ordered_nodes != oldOrderedNodes){
        oldOrderedNodes = ordered_nodes;
        fillVisibilityCache(ordered_nodes, octree_ref, relationship_map_ref);
        return true;
    }
    return false;
}