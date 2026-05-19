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
void initOctree(std::shared_ptr<OctreeNode> main_root, std::shared_ptr<AABB> main_aabb, std::shared_ptr<vector<Point>> points);
/// Grow the octree
uint32_t growOctree(std::shared_ptr<OctreeNode> main_root, std::shared_ptr<AABB> main_aabb, std::shared_ptr<vector<Point>> points);
/// Update the octree
void updateOctree();

// TODO: temporary function to load synchronously the point cloud
void loadPointcloud(string file, CuRast* editor);