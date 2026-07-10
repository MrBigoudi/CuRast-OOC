#pragma once

#include "CuRast.h"
#include "globals.h"

/// Init the main octree
OctreeNode* initOctree(
    std::shared_ptr<vector<Point>>& points
);

/// Grow the octree
uint32_t growOctree(
    OctreeNode* root_node, 
    const std::shared_ptr<vector<Point>>& points
);

/// Bottom up update of the octree
/// Creates nb_new_levels inner nodes
/// Returns the new root
OctreeNode* uptadeOctree(
    OctreeNode* main_root,
    uint32_t nb_new_levels
);


/// TODO: temporary function
/// Load an octree to gpu memory
void loadOctreeOnGPU(CuRast* editor, CUcontext* context, 
    OctreeNode* octree_ref,
    std::shared_ptr<AABBRelationshipMap> relationship_map_ref
);
void createCudaMemory(CuRast* editor, CUcontext* context, 
    OctreeNode* input_octree,
    std::shared_ptr<AABBRelationshipMap> relationship_map_ref
);
std::optional<CUdeviceptr> allocateChunks(
    std::shared_ptr<SNCOctree>& octree, 
    const Chunk* root, bool is_voxel_chunk = false
);

/// TODO: temporary function
/// Frees the unused octrees on gpu memory
void freeOctreesOnGPU(CuRast* editor);
/// Frees the last unused octree on gpu memory
/// If given a caller, only frees the memory when the caller is done loading
void freePreviousOctreeOnGPU(CuRast* editor, std::shared_ptr<SNCOctree> caller);


/// Add new batches to the octree
void addPointBatches();
/// Asynchronously update the octree
void updateOctreeRoutine();
