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
#include <unordered_set>

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



bool isPointCloudFile(const string& file){
	return iEndsWith(file, ".las") || iEndsWith(file, ".laz");
}

/// The point cloud files among the dropped paths: the dropped .las / .laz files, plus every .las /
/// .laz file found recursively in the dropped folders (sorted per folder, each file once)
vector<string> collectPointCloudFiles(const vector<string>& dropped){
	vector<string> point_cloud_files = {};
	std::unordered_set<string> seen = {};   // same file dropped twice (directly and through its folder)

	auto add = [&](const std::filesystem::path& path){
		std::error_code ec;
		const std::filesystem::path key_path = std::filesystem::weakly_canonical(path, ec);
		const string key = ec ? path.lexically_normal().string() : key_path.string();
		if(seen.insert(key).second){
			point_cloud_files.push_back(path.string());
		}
	};

	for(const string& item : dropped){
		std::error_code ec;
		if(std::filesystem::is_directory(item, ec)){
			vector<std::filesystem::path> found = {};
			const auto options = std::filesystem::directory_options::skip_permission_denied;
			std::filesystem::recursive_directory_iterator it(item, options, ec);
			if(ec){
				println("ERROR: can't open the folder '{}': {}", item, ec.message());
				continue;
			}
			for(const std::filesystem::recursive_directory_iterator end; it != end; it.increment(ec)){
				if(ec){
					println("WARNING: in the folder '{}': {}", item, ec.message());
					ec.clear();
					continue;
				}
				std::error_code type_ec;
				if(!it->is_regular_file(type_ec) || type_ec){continue;}
				if(isPointCloudFile(it->path().string())){
					found.push_back(it->path());
				}
			}
			// Deterministic order: sorted paths (files of a same folder together)
			std::sort(found.begin(), found.end());
			println("Folder '{}': {} point cloud files found", item, found.size());
			for(const std::filesystem::path& path : found){ add(path); }
		} else if(isPointCloudFile(item)){
			add(item);
		}
	}
	return point_cloud_files;
}

void fileDropCallback(vector<string> files){
	// The dropped files and every .las / .laz file of the dropped folders (recursively), all
	// together: their overview (one point per LAZ chunk of every file) is loaded first, then their
	// regular batches (see `createNewBatchesFromFiles')
	// The folders are walked on the thread, not in the callback (large folders take a while)
	std::thread thread_init_batches([](vector<string> dropped){
		vector<string> point_cloud_files = collectPointCloudFiles(dropped);
		if(point_cloud_files.empty()){return;}
		LoaderGpuVersion::createNewBatchesFromFiles(std::move(point_cloud_files));
	}, std::move(files));
	thread_init_batches.detach();
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