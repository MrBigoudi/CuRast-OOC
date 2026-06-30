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
	CUdeviceptr nodes;

	uint32_t max_lod_level = 0;

	uint32_t nb_nodes = 0;
	uint64_t nb_chunks = 0;
	uint64_t nb_points = 0;
	uint64_t nb_voxels = 0;

	uint64_t octree_id = 0;
	bool need_to_be_executed = false;

	CUstream stream;

	SNCOctree(string name, uint64_t octree_id) : SceneNode(name), octree_id(octree_id){}

	uint64_t getGpuMemoryUsage() override {
		uint64_t total = 0;
		total += nb_nodes * sizeof(COctreeNode);
		total += sizeof(nodes);
		total += nb_chunks * sizeof(CChunk);
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
        	.num_nodes = nb_nodes,
        	.max_lod_level = max_lod_level
		};
	}
};