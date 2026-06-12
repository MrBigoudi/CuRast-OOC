#pragma once

#include "globals.h"

/// SimLOD octree update
void simLodUpdate(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<vector<Point>>& points
);

/// SimLOD counting pass
void simLodCount(
    std::shared_ptr<OctreeNode>& main_root, 
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
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<Point>>& backlog_voxels,
    std::shared_ptr<vector<OctreeNode*>>& backlog_voxels_nodes
);

/// SimLOD point insertion pass
void simLodInsertion(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points,
    std::shared_ptr<vector<Point>>& backlog_voxels,
    std::shared_ptr<vector<OctreeNode*>>& backlog_voxels_nodes
);



/// SimLOD loading pass
void simLodLoad(
    std::shared_ptr<OctreeNode>& main_root, 
    std::shared_ptr<vector<Point>>& points,
    std::shared_ptr<vector<Point>>& spilled_points
);