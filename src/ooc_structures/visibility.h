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

/// Get a list of all visible nodes that are either loaded or in cache
std::unordered_set<AABB, AABB::Hash> getVisibleNodes(const Frustum& frustum);

/// Order the visible nodes from furthest to closest
/// All nodes in the list must appear only once
/// All nodes in the list must have their parent in the list
/// All nodes in the list must have their parent marked as closest
std::vector<AABB> orderNodes(
    const AABB& root_node,
    const std::unordered_set<AABB, AABB::Hash>& visible_nodes,
    const vec3& camera_pos
);

/// Fill the visibility cache with the ordered nodes
/// Also load and store the nodes according to the cache
void fillVisibilityCache(const std::vector<AABB>& nodes, OctreeNode* root_octree);


/// Update visibility cache and the current octree by taking into account the visibility of each nodes
void updateVisibilityCache(const mat4& view, const mat4& proj);