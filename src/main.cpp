// CuRast includes
#include <cstdio>
#include <format>
#include <print>
#include <filesystem>
#include <string>
#include <queue>
#include <vector>
#include <algorithm>
#include <execution>
#include <thread>

#include "unsuck.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "CudaModularProgram.h"
#include "CudaVulkanSharedMemory.h"
#include "VulkanCudaSharedMemory.h"
#include "jpeg/JPEGIndexer.h"

#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include "Runtime.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "json/json.hpp"
#include "CuRast.h"
#include "MappedFile.h"
#include "GLTFLoader.h"
#include "LargeGlbLoader.h"
#include "PlyLoader.h"
#include "laszip/laszip_api.h"


// Custom includes
#include "ooc_structures/loader.h"
#include "ooc_structures/settings.h"
#include "ooc_structures/gpuVersion.h"
#include "ooc_structures/neural.h"


CUcontext context;


void initCuda() {
	cuInit(0);
	
	CUctxCreateParams creation_params = {};
	cuDeviceGet(&CURuntime::device, 0);
	cuCtxCreate(&context, &creation_params, 0, CURuntime::device);
}



void fileDropCallback(vector<string> files){
	// All the dropped files together: their overview (one point per LAZ chunk of every
	// file) is loaded first, then their regular batches (see `createNewBatchesFromFiles')
	vector<string> point_cloud_files = {};
	for(const string& file : files){
		if(iEndsWith(file, ".las") || iEndsWith(file, ".laz")){
			point_cloud_files.push_back(file);
		}
	}
	if(!point_cloud_files.empty()){
		std::thread thread_init_batches([](vector<string> point_cloud_files){
			LoaderGpuVersion::createNewBatchesFromFiles(std::move(point_cloud_files));
		}, std::move(point_cloud_files));
		thread_init_batches.detach();
	}
}



void initApp(CuRast* instance, CUcontext* context) {
	// Clean previous run
	std::filesystem::remove_all(OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY);
	std::filesystem::create_directories(OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY);

	// Init CuRast
	initCuda();
	VKRenderer::init();
	CuRast::setup();

	// Init custom structures
	OocSimLodSettings::init();
	GpuVersion::init(instance, context);

	// Init custom callbacks
	VKRenderer::onFileDrop([&](vector<string> files){fileDropCallback(files);});
}



void update(CuRast* instance, CUcontext* context){
	GpuVersion::updateOctree(instance, context);
	GpuVersion::takeSingleScreenShot();
	GpuVersion::tryToCenterTarget();
	instance->update();

	DeviceState* state = instance->deviceState;
	double stage1_millies = double(state->nanotime_stage_1 - state->nanotime_start) / 1'000'000.0;
	double stage2_millies = double(state->nanotime_stage_2 - state->nanotime_stage_1) / 1'000'000.0;
	double stage3_millies = double(state->nanotime_stage_3 - state->nanotime_stage_2) / 1'000'000.0;
	Runtime::debugValues["stage 1"] = format("{:.3f}", stage1_millies);
	Runtime::debugValues["stage 2"] = format("{:.3f}", stage2_millies);
	Runtime::debugValues["stage 3"] = format("{:.3f}", stage3_millies);
}



void clean(std::vector<std::thread*> all_threads){
	// Destruction procedure
	GpuVersion::mainLoopIsTerminating = true;
	
	for(std::thread* thread : all_threads){
		if(thread){
			thread->join();
			delete(thread);
		}
	}

	cudaDeviceSynchronize();
	println("GPU Memory Usage before destroy:\n{}", GpuVersion::getGpuMemoryUsage());
	OocSimLodSettings::display();
	GpuVersion::destroy(CuRast::instance, &context);
	cudaDeviceSynchronize();

	std::filesystem::remove_all(OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY);
	VKRenderer::destroy();
	println("GPU Memory Usage final:\n{}", GpuVersion::getGpuMemoryUsage());
}



int main(int argc, char** argv){
	// Miscellaneous stuff
	Benchmarking::datasetPath = "./";
	for(int i = 1; i < argc - 1; i++){
		if(string(argv[i]) == "-b"){
			Benchmarking::datasetPath = argv[i + 1];
		}
	}
	std::locale::global(getSaneLocale());

	std::thread* thread_points_loader = nullptr;

	try {
		// Initialisation
		println("GPU Memory Usage Before initialisation:\n{}", GpuVersion::getGpuMemoryUsage());
		initApp(CuRast::instance, &context);
		thread_points_loader = new std::thread([&](){
			while(!GpuVersion::mainLoopIsTerminating){
				LoaderGpuVersion::loadingRoutine();
			}
		});
		println("GPU Memory Usage After initialisation:\n{}", GpuVersion::getGpuMemoryUsage());

		// Main loop
		VKRenderer::loop(
			[&]() { update(CuRast::instance, &context); },
			[&]() { CuRast::instance->render(); },
			[&]() { CuRast::instance->postFrame(); }
		);

		// Cleaning
		clean({thread_points_loader});
	} catch(int) {
		clean({thread_points_loader});
	}
}