#pragma once

#include "globals.h"


struct Plane {
    vec3 normal;
    float constant;

    Plane(){}
    Plane(float x, float y, float z, float w);
};


struct Frustum {
    Plane planes[6] = {};

    Frustum(const mat4& view_proj);

    /// Checks if a node intersects a frustum
    bool doesIntersect(const AABB& aabb) const;

    void display() const;
};

struct Visibility {
    /// Get a list of all visible nodes that are either loaded or in cache
    static std::unordered_set<IdAABB> getVisibleNodes(
        const Frustum& frustum,
        std::shared_ptr<AABBRelationshipMap> relationship_map_ref
    );

    /// Order the visible nodes from furthest to closest
    /// All nodes in the list must appear only once
    /// All nodes in the list must have their parent in the list
    /// All nodes in the list must have their parent marked as closest
    static std::vector<IdAABB> orderNodes(
        const IdAABB& root_node,
        const std::unordered_set<IdAABB>& visible_nodes,
        const vec3& camera_pos,
        std::shared_ptr<AABBRelationshipMap> relationship_map_ref
    );

    /// Fill the visibility cache with the ordered nodes
    /// Also load and store the nodes according to the cache
    static void fillVisibilityCache(
        const std::vector<IdAABB>& nodes, 
        OctreeNode* root_octree,
        std::shared_ptr<AABBRelationshipMap> relationship_map_ref
    );


    /// Update visibility cache and the current octree by taking into account the visibility of each nodes
    /// Return false it the scene hasn't changed
    static bool updateVisibilityCache(
        const mat4& view, const mat4& proj, 
        OctreeNode* octree_ref,
        std::shared_ptr<AABBRelationshipMap> relationship_map_ref
    );
};