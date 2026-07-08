#pragma once

#include <string>
#include <vector>

#include "SceneNode.h"
#include "./kernels/HostDeviceInterface.h"

using std::string;
using std::vector;
using glm::ivec2;

struct SNCOctree : public SceneNode{
	CUdeviceptr nodes;
	CUdeviceptr aabbs;

	uint32_t max_lod_level = 0;

	uint32_t nb_nodes  = 0;
	uint64_t nb_chunks = 0;
	uint64_t nb_points = 0;
	uint64_t nb_voxels = 0;
	uint64_t nb_aabbs  = 0;

	uint64_t octree_id = 0;
	bool need_to_be_executed = false;

	CUstream stream;

	SNCOctree(string name, uint64_t octree_id) : SceneNode(name), octree_id(octree_id){}

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
		cuda_status = cuMemFreeAsync(nodes, stream);
		cudaCheck(cuda_status, "nodes");
		cuda_status = cuMemFreeAsync(aabbs, stream);
		cudaCheck(cuda_status, "aabbs");

		cuStreamDestroy(stream);
	}

	uint64_t getOverheadGPUMemoryUsage() {
		uint64_t total = 0;
		// List of nodes pointers
		total += nb_nodes * sizeof(CUdeviceptr);
		// List of AABBs pointers
		total += nb_aabbs * sizeof(CUdeviceptr);
		// Pointers to the structures + debug values
		total += sizeof(CFullOctree);
		return total;
	}

	uint64_t getGpuMemoryUsage() override {
		uint64_t total = getOverheadGPUMemoryUsage();
		// Actual chunks
		total += nb_chunks * sizeof(CChunk);
		// Actual nodes
		total += nb_nodes * sizeof(COctreeNode);
		return total;
	}

	bool isDoneLoadingToGpu() {
		// Have a peek at the stream
		// returns cudaSuccess if the stream is empty
		// returns cudaErrorNotReady if the stream is not empty
		CUresult status = cuStreamQuery(stream);

		switch (status) {
			case CUDA_SUCCESS:
				return true;
			case CUDA_ERROR_NOT_READY:
				return false;
			default:
				println("Error on Octree's cuda stream");
				const char* name = nullptr;
				const char* desc = nullptr;
				cuGetErrorName(status, &name);
				cuGetErrorString(status, &desc);

				println(stderr, "CUDA error {} ({}): {}\n ",
					int(status),
					name ? name : "unknown",
					desc ? desc : "unknown");
				exit(EXIT_FAILURE);
		};

		return false;
	}

	CFullOctree toFullOctree() const {
		return CFullOctree {
			.world = transform_global,
        	.nodes = (COctreeNode**)nodes,
			.aabbs = (CAABB*)aabbs,
        	.num_nodes = nb_nodes,
        	.max_lod_level = max_lod_level
		};
	}
};