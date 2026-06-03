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

	CUdeviceptr nodes;
	CUdeviceptr aabbs;
	CUdeviceptr chunks;
	CUdeviceptr occupancy_grids;

	uint32_t num_nodes = 0;
	uint32_t max_lod_level = 0;
	
	SNCOctree(string name) : SceneNode(name){
		
	}

	~SNCOctree() {
		CUresult cuda_status = CUDA_SUCCESS;

		auto cudaCheck = [](CUresult result, string struct_name){
			const char* name = nullptr;
			const char* desc = nullptr;
			if(result != CUDA_SUCCESS){
				cuGetErrorName(result, &name);
				cuGetErrorString(result, &desc);
				println(stderr, "Error: cuMemFree failed for {}, {} ({}): {}\n ",
					struct_name,
					int(result),
					name ? name : "unknown",
					desc ? desc : "unknown"
				);
			}
		};
		
		for(CUdeviceptr& ptr : cptr_nodes){
			cuda_status = cuMemFree(ptr);
			cudaCheck(cuda_status, "cptr_nodes");
		}
		for(CUdeviceptr& ptr : cptr_aabbs){
			cuda_status = cuMemFree(ptr);
			cudaCheck(cuda_status, "cptr_aabbs");
		}
		for(CUdeviceptr& ptr : cptr_chunks){
			cuda_status = cuMemFree(ptr);
			cudaCheck(cuda_status, "cptr_chunks");
		}
		for(CUdeviceptr& ptr : cptr_occupancy_grids){
			cuda_status = cuMemFree(ptr);
			cudaCheck(cuda_status, "cptr_occupancy_grids");
		}

		cuda_status = cuMemFree(nodes);
		cudaCheck(cuda_status, "nodes");

		cuda_status = cuMemFree(aabbs);
		cudaCheck(cuda_status, "aabbs");

		cuda_status = cuMemFree(chunks);
		cudaCheck(cuda_status, "chunks");

		cuda_status = cuMemFree(occupancy_grids);
		cudaCheck(cuda_status, "occupancy_grids");
	}

	uint64_t getGpuMemoryUsage() override {
		uint64_t total = 0;
		
		total += cptr_nodes.size() * sizeof(COctreeNode);
		total += cptr_aabbs.size() * sizeof(CAABB);
		total += cptr_chunks.size() * sizeof(CChunk);
		total += cptr_occupancy_grids.size() * sizeof(COccupancyGrid);

		total += sizeof(nodes);
		total += sizeof(aabbs);
		total += sizeof(chunks);
		total += sizeof(occupancy_grids);

		return total;
	}


};