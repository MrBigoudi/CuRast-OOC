#pragma once

#include <string>
#include <vector>

#include "SceneNode.h"
#include "./kernels/HostDeviceInterface.h"

using std::string;
using std::vector;
using glm::ivec2;

struct SNCPoints : public SceneNode{

	CUdeviceptr cptr_positions;
	CUdeviceptr cptr_colors;
	void* ptr_positions;
	void* ptr_colors;
	uint64_t numPoints = 0;
	
	SNCPoints(string name) : SceneNode(name){
		
	}

	uint64_t getGpuMemoryUsage() override {
		uint64_t total = 0;
		
		total += numPoints * (sizeof(vec3) + sizeof(uint32_t));

		return total;
	}

};