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
	uint32_t num_nodes = 0;
	uint32_t max_lod_level = 0;
	
	SNCOctree(string name) : SceneNode(name){
		
	}

	uint64_t getGpuMemoryUsage(){
		return 0;
	}

};