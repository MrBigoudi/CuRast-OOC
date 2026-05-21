#pragma once

#include "CuRast.h"
#include "globals.h"

/// Initialized empty batches of points from las/laz files
/// Updates the global `batchesToLoad` variable
void initLoadPointBatches(string file);

/// Load points from the global `batchesToLoad` variable
void loadPointsInBatches();

/// Send points to CUDA memory
void loadBatchesOnGPU(CuRast* editor);

/// The main loop of the OOC update / rendering
void mainLoop();

/// Init the main octree
void initOctree(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<AABB>& main_aabb, 
    std::shared_ptr<vector<Point>>& points
);

/// Grow the octree
uint32_t growOctree(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<AABB>& main_aabb, 
    std::shared_ptr<vector<Point>>& points
);

/// Bottom up update of the octree
/// Creates nb_new_levels * 8 new nodes:
///     - (nb_new_levels * 7) empty leaves
///     - nb_new_levels inner nodes
void uptadeOctree(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<AABB>& main_aabb, 
    uint32_t nb_new_levels
);

/// SimLOD octree update
void simLodUpdate(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<AABB>& main_aabb, 
    std::shared_ptr<vector<Point>>& points
);
/// SimLOD counting pass
void simLodCount(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<AABB>& main_aabb, 
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<OctreeNode*>>& spilling_nodes
);
/// SimLOD splitting pass
void simLodSplit(
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<OctreeNode*>>& spilling_nodes
);
/// SimLOD voxel sampling pass
void simLodVoxelSampling(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<AABB>& main_aabb, 
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<Point>>& backlog_voxels,
    std::shared_ptr<vector<OctreeNode*>>& backlog_voxels_nodes
);
/// SimLOD point insertion pass
void simLodInsertion(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<AABB>& main_aabb, 
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<Point>>& backlog_voxels,
    std::shared_ptr<vector<OctreeNode*>>& backlog_voxels_nodes
);


// TODO: temporary function to load an octree to gpu memory
void loadOctree(CuRast* editor, const std::shared_ptr<OctreeNode>& main_root, const std::shared_ptr<AABB>& main_aabb);

// TODO: temporary function to load synchronously the point cloud
void loadPointcloud(string file, CuRast* editor);