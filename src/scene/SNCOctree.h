#pragma once

#include <string>
#include <vector>

#include "SceneNode.h"
#include "./kernels/HostDeviceInterface.h"

using std::string;
using std::vector;
using glm::ivec2;

struct SNCOctree : public SceneNode{

	vector<CUdeviceptr> cptr_nodes;
	vector<CUdeviceptr> cptr_aabbs;
    vector<CUdeviceptr> cptr_chunks;
    vector<CUdeviceptr> cptr_occupancy_grids;
	uint32_t num_nodes = 0;
	uint32_t max_lod_level = 0;
	
	SNCOctree(string name) : SceneNode(name){
		
	}

	~SNCOctree() {

	}

	uint64_t getGpuMemoryUsage() override {
		uint64_t total = 0;
		
		total += cptr_nodes.size() * sizeof(COctreeNode);
		total += cptr_aabbs.size() * sizeof(CAABB);
		total += cptr_chunks.size() * sizeof(CChunk);
		total += cptr_occupancy_grids.size() * sizeof(COccupancyGrid);

		return total;
	}


};