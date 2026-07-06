#pragma once

#include "globals.h"

struct SimLod {
    /// SimLOD octree update
    static void update(
        OctreeNode* main_root, 
        std::shared_ptr<vector<Point>>& points,
        std::shared_ptr<AABBRelationshipMap> relationship_map_ref
    );

    /// SimLOD loading pass
    static void load(
        OctreeNode* main_root, 
        std::shared_ptr<vector<Point>>& points,
        std::shared_ptr<AABBRelationshipMap> relationship_map_ref
    );

    /// SimLOD counting pass
    static void count(
        OctreeNode* main_root, 
        std::shared_ptr<vector<Point>>& points,
        std::shared_ptr<vector<Point>>& spilled_points,
        std::shared_ptr<vector<OctreeNode*>>& spilling_nodes
    );

    /// SimLOD splitting pass
    static void split(
        std::shared_ptr<vector<Point>>& spilled_points,
        std::shared_ptr<vector<OctreeNode*>>& spilling_nodes,
        std::shared_ptr<AABBRelationshipMap> relationship_map_ref
    );

    /// SimLOD voxel sampling pass
    static void voxelSampling(
        OctreeNode* main_root, 
        std::shared_ptr<vector<Point>>& points,
        std::shared_ptr<vector<Point>>& spilled_points,
        std::shared_ptr<vector<Point>>& backlog_voxels,
        std::shared_ptr<vector<OctreeNode*>>& backlog_voxels_nodes
    );

    /// SimLOD point insertion pass
    static void insertion(
        OctreeNode* main_root, 
        std::shared_ptr<vector<Point>>& points,
        std::shared_ptr<vector<Point>>& spilled_points,
        std::shared_ptr<vector<Point>>& backlog_voxels,
        std::shared_ptr<vector<OctreeNode*>>& backlog_voxels_nodes
    );
};