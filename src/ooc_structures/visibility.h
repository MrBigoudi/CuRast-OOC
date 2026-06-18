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


/// Update visibility cache and the current octree by taking into account the visibility of each nodes
void updateVisibilityCache(const mat4& view, const mat4& proj);