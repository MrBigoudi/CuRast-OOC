#pragma once

#include "CuRast.h"
#include "globals.h"

/// Init the main octree
void initOctree(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<AABB>& main_aabb, 
    std::shared_ptr<vector<Point>>& points
);

/// Grow the octree
uint32_t growOctree(
    const std::shared_ptr<AABB>& main_aabb, 
    const std::shared_ptr<vector<Point>>& points
);

/// Bottom up update of the octree
/// Creates nb_new_levels inner nodes
void uptadeOctree(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<AABB>& main_aabb, 
    uint32_t nb_new_levels
);


/// TODO: temporary function
/// Load an octree to gpu memory
void loadOctreeOnGPU(CuRast* editor);
/// TODO: temporary function
/// Frees the unused octrees on gpu memory
void freeOctreesOnGPU(CuRast* editor, bool force_free = false);


/// Add new batches to the octree
void addPointBatches();
/// Asynchronously update the octree
void updateOctreeRoutine();
