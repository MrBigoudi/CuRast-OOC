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
		CUresult cuda_status = CUDA_SUCCESS;
		const char* name = nullptr;
		const char* desc = nullptr;
		
		for(CUdeviceptr& ptr : cptr_nodes){
			cuda_status = cuMemFree(ptr);
			if(cuda_status != CUDA_SUCCESS){
				cuGetErrorName(cuda_status, &name);
				cuGetErrorString(cuda_status, &desc);
				println(stderr, "Error: cuMemFree failed for cptr_nodes, {} ({}): {}\n ",
					int(cuda_status),
					name ? name : "unknown",
					desc ? desc : "unknown"
				);
			}
		}
		for(CUdeviceptr& ptr : cptr_aabbs){
			cuda_status = cuMemFree(ptr);
			if(cuda_status != CUDA_SUCCESS){
				cuGetErrorName(cuda_status, &name);
				cuGetErrorString(cuda_status, &desc);
				println(stderr, "Error: cuMemFree failed for cptr_aabbs, {} ({}): {}\n ",
					int(cuda_status),
					name ? name : "unknown",
					desc ? desc : "unknown"
				);
			}
		}
		for(CUdeviceptr& ptr : cptr_chunks){
			cuda_status = cuMemFree(ptr);
			if(cuda_status != CUDA_SUCCESS){
				cuGetErrorName(cuda_status, &name);
				cuGetErrorString(cuda_status, &desc);
				println(stderr, "Error: cuMemFree failed for cptr_chunks, {} ({}): {}\n ",
					int(cuda_status),
					name ? name : "unknown",
					desc ? desc : "unknown"
				);
			}
		}
		for(CUdeviceptr& ptr : cptr_occupancy_grids){
			cuda_status = cuMemFree(ptr);
			if(cuda_status != CUDA_SUCCESS){
				cuGetErrorName(cuda_status, &name);
				cuGetErrorString(cuda_status, &desc);
				println(stderr, "Error: cuMemFree failed for cptr_occupancy_grids, {} ({}): {}\n ",
					int(cuda_status),
					name ? name : "unknown",
					desc ? desc : "unknown"
				);
			}
		}
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