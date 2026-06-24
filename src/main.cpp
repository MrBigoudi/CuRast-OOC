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

#include "ooc_structures/structureUpdate.h"
#include "ooc_structures/simLod.h"
#include "ooc_structures/loader.h"
#include "ooc_structures/outOfCore.h"
#include "ooc_structures/visibility.h"


using namespace std; // YOLO

CUcontext context;

mat4 flip = mat4(
	1.000,  0.000, 0.000, 0.000,
	0.000,  0.000, 1.000, 0.000,
	0.000, -1.000, 0.000, 0.000,
	0.000,  0.000, 0.000, 1.000);

void initCuda() {
	cuInit(0);
	
	CUctxCreateParams creation_params = {};
	cuDeviceGet(&CURuntime::device, 0);
	cuCtxCreate(&context, &creation_params, 0, CURuntime::device);
}




void initScene() {
	CuRast* editor = CuRast::instance;
	Scene& scene = editor->scene;

	// position: 124.54672426747658, -42.72048538939598, -12.2730454323992 
	Runtime::controls->yaw    = -5.179;
	Runtime::controls->pitch  = 0.108;
	Runtime::controls->radius = 142.656;
	Runtime::controls->target = { -2.859, 21.085, -5.387, };

	auto loadSponza = [=](){ 
		string file = "F:/resources/meshes/sponza-png_by_Ludicon.glb";

		static auto glb = largeGlb::load(file, context, {.compress = false});
		glb->glbNode->name = "Sponza";
		glb->glbNode->transform = flip * glb->glbNode->transform;
		scene.world->children.push_back(glb->glbNode);

		Runtime::controls->yaw    = -4.731;
		Runtime::controls->pitch  = 0.009;
		Runtime::controls->radius = 336.359;
		Runtime::controls->target = { -2.986, 32.881, 119.491, };
	};

	auto loadSponzaJPEG = [=](){ 
		std::string file = "./resources/meshes/Sponza_70.glb";

		static auto glb = largeGlb::load(file, context, {
			.skipUVs = false, 
			.compress = true,
			.useJpegTextures = true,
		});
		glb->glbNode->transform = flip * glb->glbNode->transform;
		scene.world->children.push_back(glb->glbNode);

		// position: 1.3305474790692626, 0.45402304811990946, 1.1192273142715552 
		Runtime::controls->yaw    = -4.646;
		Runtime::controls->pitch  = 0.024;
		Runtime::controls->radius = 8.993;
		Runtime::controls->target = { -7.642, -0.142, 1.105, };
	};

	auto loadCubeJpeg = [=](){ 
		std::string file = "./resources/meshes/Cube_70.glb";

		static auto glb = largeGlb::load(file, context, {
			.skipUVs = false, 
			.compress = true,
			.useJpegTextures = true,
		});
		glb->glbNode->transform = flip * glb->glbNode->transform;
		editor->scene.world->children.push_back(glb->glbNode);

		// position: 25.15382712255294, -20.17937489109018, 12.02894404906026 
		Runtime::controls->yaw    = -5.466;
		Runtime::controls->pitch  = -0.531;
		Runtime::controls->radius = 34.148;
		Runtime::controls->target = { 0.252, -0.030, 0.196};
	};

	auto loadHakone = [=](){
		string file = "./example_donaukanal_urania.glb";
		// string file = "F:/resources/meshes/hakone_lantern.glb";
		// string file = "F:/resources/meshes/hakone_1M.glb";

		static auto glb = largeGlb::load(file, context, {.skipUVs = false, .compress = false});
		editor->scene.world->children.push_back(glb->glbNode);

		// Overview
		Runtime::controls->yaw    = -7.070;
		Runtime::controls->pitch  = -0.515;
		Runtime::controls->radius = 37.564;
		Runtime::controls->target = { 25.607, -17.328, 8.340, };
	};

	auto loadHakoneInstances = [=](){
		string file = "./example_donaukanal_urania.glb";
		// string file = "./resources/meshes/donaukanal_urania_1M_jpeg80.glb";
		// string file = "F:/resources/meshes/hakone_lantern.glb";
		// string file = "F:/resources/meshes/hakone_lantern_optimized.glb";
		// string file = "F:/resources/meshes/hakone_lantern_3.glb";
		// string file = "F:/resources/meshes/hakone_1m.glb";
		// string file = "F:/resources/meshes/hakone_1m_optimized.glb";

		static auto glb = largeGlb::load(file, context, {.skipUVs = false, .compress = false});

		shared_ptr<SNTriangles> original = dynamic_pointer_cast<SNTriangles>(glb->glbNode->children[0]);

		for(int ix = 0; ix < 50; ix++)
		for(int iy = 0; iy < 60; iy++)
		{
			shared_ptr<SNTriangles> instance = make_shared<SNTriangles>("instance");
			instance->mesh = original->mesh;
			instance->texture = original->texture;
			instance->aabb = original->aabb;
			instance->transform = glm::translate(vec3{ix * 30.0f, iy * 30.0f, 0.0f});
			editor->scene.world->children.push_back(instance);
		}

		// position: -0.9794631786208647, -40.41708964196321, 21.421843904392993 
		Runtime::controls->yaw    = -7.070;
		Runtime::controls->pitch  = -0.515;
		Runtime::controls->radius = 37.564;
		Runtime::controls->target = { 25.607, -17.328, 8.340, };
	};

	auto loadSpot = [=](){ 
		std::string file = "F:/resources/meshes/spot.glb";

		static auto glb = largeGlb::load(file, context, {.skipUVs = false, .compress = false});
		editor->scene.world->children.push_back(glb->glbNode);

		// position: 1.1620903300458982, 1.4676847017158816, -0.27796041598897114 
		Runtime::controls->yaw    = -16.358;
		Runtime::controls->pitch  = -0.381;
		Runtime::controls->radius = 1.957;
		Runtime::controls->target = { -0.023, 0.022, 0.301, };
	};

	auto loadZorah = [=](){ 
		string file = "F:/resources/meshes/zorah_main_public.gltf/zorah_main_public.gltf";
		// string file = "F:/resources/meshes/zorah_main_public.gltf_optimized/zorah_main_public.gltf";
		
		// Mesh has no textures/uvs/normals
		CuRastSettings::displayAttribute = DisplayAttribute::NONE;

		static auto glb = largeGlb::load(file, context, {.skipUVs = true, .compress = true});
		glb->glbNode->transform = flip;
		editor->scene.world->children.push_back(glb->glbNode);

		// Let's remove some less appealing billboards
		vector<shared_ptr<SceneNode>> filtered;
		for(shared_ptr<SceneNode> node : glb->glbNode->children){
			if(node->name == "FogCard") continue;
			if(node->name == "Plane") continue;
			
			filtered.push_back(node);
		}
		glb->glbNode->children = filtered;

		// Overview
		Runtime::controls->yaw    = -16.537;
		Runtime::controls->pitch  = -0.472;
		Runtime::controls->radius = 97.539;
		Runtime::controls->target = { 17.436, -6.343, 3.689, };
		
		// Closeup
		Runtime::controls->yaw    = -17.344;
		Runtime::controls->pitch  = 0.073;
		Runtime::controls->radius = 10.893;
		Runtime::controls->target = { 44.294, 1.156, 6.458, };

	};

	auto createCube = [&](){
		static shared_ptr<SNTriangles> node = make_shared<SNTriangles>("node");
		node->texture = new Texture();
		node->mesh = new Mesh();

		{ // Create Default Texture
			int64_t textureWidth = 128;
			int64_t textureHeight = 128;
			vector<uint8_t> textureData = vector<uint8_t>(2 * 4 * textureWidth * textureHeight, 255);
			
			node->texture->width = textureWidth;
			node->texture->height = textureHeight;
			node->texture->data = (uint32_t*)MemoryManager::alloc(byteSizeOf(textureData), "default texture");

			cuMemcpyHtoDAsync(CUdeviceptr(node->texture->data), textureData.data(), byteSizeOf(textureData), 0);
		}

		node->mesh->isLoaded = true;
		node->mesh->name = "default mesh";
		node->mesh->numTriangles = 0;

		vector<vec3> positions = {
			vec3{0.0f, 0.0f, 0.0f},
			vec3{1.0f, 0.0f, 0.0f},
			vec3{1.0f, 1.0f, 0.0f},
		};
		vector<vec2> uvs = {
			vec2{0.0f, 0.0f},
			vec2{1.0f, 0.0f},
			vec2{1.0f, 1.0f},
		};
		vector<uint32_t> indices = {0, 1, 2};

		int numVertices = positions.size();
		int numTriangles = indices.size() / 3;
		node->mesh->cptr_position = MemoryManager::alloc(sizeof(vec3) * numVertices, "position");
		node->mesh->cptr_uv       = MemoryManager::alloc(sizeof(vec2) * numVertices, "uv");
		node->mesh->cptr_indices  = MemoryManager::alloc(sizeof(uint32_t) * 3 * numTriangles, "indices");
		node->aabb.extend(vec3{-1.0f, -1.0f, -1.0f});
		node->aabb.extend(vec3{1.0f, 1.0f, 1.0f});

		cuMemcpyHtoD(node->mesh->cptr_position, positions.data(), byteSizeOf(positions));
		cuMemcpyHtoD(node->mesh->cptr_uv, uvs.data(), byteSizeOf(uvs));
		cuMemcpyHtoD(node->mesh->cptr_indices, indices.data(), byteSizeOf(indices));

		node->mesh->numTriangles = indices.size() / 3;
		node->mesh->numVertices = positions.size();

		scene.world->children.push_back(node);

		// position: 0.8369693760957783, 0.05588397571280396, 0.02743282811653472 
		Runtime::controls->yaw    = -17.426;
		Runtime::controls->pitch  = -0.272;
		Runtime::controls->radius = 0.818;
		Runtime::controls->target = { 0.028, 0.172, -0.005, };
	};

	auto loadVenice = [=](){ 
		string file = "F:/resources/meshes/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/venice.gltf";
		// string file = "F:/resources/meshes/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/venice_optimized.gltf";
		
		CuRastSettings::displayAttribute = DisplayAttribute::TEXTURE;

		static auto glb = largeGlb::load(file, context, {
			.skipNormals = true, // We need that previous VRAM
			.skipVertexColors = true,
			.compress = true,
			.useJpegTextures = false,
			.imageDivisionFactor = 2
		});
		editor->scene.world->children.push_back(glb->glbNode);
		
		// Distant
		Runtime::controls->yaw    = -18.968;
		Runtime::controls->pitch  = -0.769;
		Runtime::controls->radius = 4180.978;
		Runtime::controls->target = { 352.960, -1134.931, -462.529, };
	};

	auto loadLion = [=]() {
		std::string file = "./lion.laz";

		// position: 4.283698075250294, -4.795270477550499, 6.400710991008715 
		Runtime::controls->yaw    = -5.582;
		Runtime::controls->pitch  = -0.294;
		Runtime::controls->radius = 5.584;
		Runtime::controls->target = { 0.679, -0.714, 5.163};

		if(!CPU_PARALLELISED){
			initLoadPointBatches(file);
			while(true){
				loadPointsInBatches();
				bool done = true;
				for(uint32_t i=0; i<BATCHES_QUEUE_SIZE; i++){
					if(batchesQueue[i] && batchesQueue[i]->state != BatchState::Loaded){
						done = false;
					}
				}
				if(done){break;}
			}
			while(true){
				addPointBatches();
				bool done = true;
				for(uint32_t i=0; i<BATCHES_QUEUE_SIZE; i++){
					if(batchesQueue[i] && batchesQueue[i]->state != BatchState::Inserted){
						done = false;
					}
				}
				if(done){break;}
			}
			while(true){
				loadBatchesOnGPU(CuRast::instance);
				bool done = true;
				for(uint32_t i=0; i<BATCHES_QUEUE_SIZE; i++){
					if(batchesQueue[i] && batchesQueue[i]->state != BatchState::ToRemove){
						done = false;
					}
				}
				if(done){break;}
			}
			clearUnusedBatches();
			loadOctreeOnGPU(CuRast::instance, &context);
		} else {
			std::thread thread_loadLion([&](std::string file){
				initLoadPointBatches(file);
			}, file);
			thread_loadLion.detach();
		}
	};
	

	// createCube();
	// loadZorah();
	// loadGraffiti();
	// loadHakone();
	// loadHakoneInstances();
	// loadXyzDragon();
	// loadSpot();
	// loadWietrznia();
	// loadSponza();
	// loadSponzaJPEG();
	// loadCubeJpeg();
	// loadPolygraphenewerkLeibzigInstances();
	// loadVenice();
	loadLion();

	// // TODO: to remove
	// {

	// 	glm::mat4 view = {
	// 		0.7640781, -0.18694586, 0.6174431, -0, 
	// 		0.6451238, 0.22141676, -0.7312933, 0, 
	// 		2.7755576e-17, 0.9570924, 0.2897829, -0, 
	// 		-0.0581906, -4.6564403, -8.021537, 1
	// 	};
	// 	glm::mat4 proj = {
	// 		0.9742786, 0, 0, 0, 
	// 		0, 1.7320509, 0, 0, 
	// 		0, 0, 0, -1, 
	// 		0, 0, 0.01, 0
	// 	};
	// 	updateVisibilityCache(view, proj);
				
	// 	println("First lion loaded");
	// 	std::shared_ptr<OctreeNode> first_octree = mainOctree;
	// 	std::shared_ptr<LRUCache> first_cache = updatesCache;
	// 	mainOctree = nullptr;
	// 	uint32_t correct_size = 1024;
	// 	updatesCache = std::make_shared<LRUCache>("updates cache", correct_size);
	// 	visibilityCache = std::make_shared<LRUCache>("visibility cache", LRU_VISIBILITY_CACHE_SIZE);
	// 	LRUCache::stored_set = {};
	// 	aabb_relationship_map.clear();
	// 	aabb_mutex_map.clear();
	// 	aabb_parent_map.clear();

	// 	loadLion();
	// 	updateVisibilityCache(view, proj);
	// 	println("Second lion loaded");

	// 	if(*mainOctree != *first_octree){
	// 		println("The two octrees should be similar: ");
	// 		println("Original octree: cache size = {}", LRU_UPDATES_CACHE_SIZE);
	// 		first_octree->display();
	// 		println("\n\n\n\n\n\n\n\n\n");
	// 		println("Correct octree: cache size = {}", correct_size);
	// 		mainOctree->display();
	// 		{
	// 			std::lock_guard<std::mutex> lock(mainLoopIsTerminatingMtx);
	// 			MAIN_LOOP_IS_TERMINATING = true;
	// 			// Destroy temporary folder
	// 			std::filesystem::remove_all(TEMPORARY_DIRECTORY);
	// 		}
	// 		exit(EXIT_FAILURE);
	// 	}
	// }


	// loadTemple();
}

void update(){

	if(Benchmarking::request_scenario){

		auto scenario = Benchmarking::request_scenario;

		string path = stringReplace(scenario->path, "DATASETPATH", Benchmarking::datasetPath);

		CuRastSettings::displayAttribute = scenario->attribute;

		static auto glb = largeGlb::load(path, context, {
			.skipUVs = scenario->skipUVs,
			.skipNormals = scenario->skipNormals,
			.compress = scenario->compress,
			.useJpegTextures = scenario->useJpegTextures,
			.imageDivisionFactor = scenario->imageDivisionFactor,
		});
		glb->glbNode->name = scenario->label;
		glb->glbNode->transform = scenario->transform * glb->glbNode->transform;

		vector<shared_ptr<SceneNode>> filtered;
		for(shared_ptr<SceneNode> node : glb->glbNode->children){

			bool accept = scenario->filter(node);
			
			if(accept){
				filtered.push_back(node);
			}
		}
		glb->glbNode->children = filtered;

		shared_ptr<SceneNode> original = glb->glbNode;

		function<shared_ptr<SceneNode>(shared_ptr<SceneNode>)> deepClone =
		[&deepClone](shared_ptr<SceneNode> node) -> shared_ptr<SceneNode> {

			shared_ptr<SceneNode> clone;

			shared_ptr<SNTriangles> tris = dynamic_pointer_cast<SNTriangles>(node);
			if(tris){
				auto triClone      = make_shared<SNTriangles>(tris->name);
				triClone->mesh     = tris->mesh;
				triClone->texture  = tris->texture;
				triClone->aabb     = tris->aabb;
				clone = triClone;
			}else{
				clone = make_shared<SceneNode>(node->name);
				clone->aabb = node->aabb;
			}

			clone->transform = node->transform;
			clone->visible   = node->visible;

			for(auto& child : node->children){
				clone->children.push_back(deepClone(child));
			}

			return clone;
		};

		for(int ix = 0; ix < scenario->instances_count.x; ix++)
		for(int iy = 0; iy < scenario->instances_count.y; iy++)
		{
			shared_ptr<SceneNode> clone = deepClone(original);
			clone->transform = glm::translate(vec3{ix * scenario->instances_spacing.x, iy * scenario->instances_spacing.y, 0.0f}) * clone->transform;
			CuRast::instance->scene.world->children.push_back(clone);
		}

		Runtime::controls->yaw    = scenario->view_overview.yaw;
		Runtime::controls->pitch  = scenario->view_overview.pitch;
		Runtime::controls->radius = scenario->view_overview.radius;
		Runtime::controls->target = scenario->view_overview.target;

		Benchmarking::active_scenario = Benchmarking::request_scenario;
		Benchmarking::request_scenario = nullptr;
	}

}

int main(int argc, char** argv){

	Benchmarking::datasetPath = "./";

	for(int i = 1; i < argc - 1; i++){
		if(string(argv[i]) == "-b"){
			Benchmarking::datasetPath = argv[i + 1];
		}
	}

	std::locale::global(getSaneLocale());

	initCuda();
	VKRenderer::init();
	CuRast::setup();

	// temporary function
	auto sequential = [&](vector<string> files){
		std::for_each(files.begin(), files.end(), [&](string& file){
			if(iEndsWith(file, ".las") || iEndsWith(file, ".laz")){
				initLoadPointBatches(file);
				while(true){
					loadPointsInBatches();
					bool done = true;
					for(uint32_t i=0; i<BATCHES_QUEUE_SIZE; i++){
						if(batchesQueue[i] && batchesQueue[i]->state != BatchState::Loaded){
							done = false;
						}
					}
					if(done){break;}
				}
				while(true){
					addPointBatches();
					bool done = true;
					for(uint32_t i=0; i<BATCHES_QUEUE_SIZE; i++){
						if(batchesQueue[i] && batchesQueue[i]->state != BatchState::Inserted){
							done = false;
						}
					}
					if(done){break;}
				}
				while(true){
					loadBatchesOnGPU(CuRast::instance);
					bool done = true;
					for(uint32_t i=0; i<BATCHES_QUEUE_SIZE; i++){
						if(batchesQueue[i] && batchesQueue[i]->state != BatchState::ToRemove){
							done = false;
						}
					}
					if(done){break;}
				}
				clearUnusedBatches();
				loadOctreeOnGPU(CuRast::instance, &context);
			}
		});
	};

	VKRenderer::onFileDrop([&](vector<string> files){
		if(CPU_PARALLELISED){
			std::for_each(std::execution::par, files.begin(), files.end(), 
				[&](string& file){
					if(iEndsWith(file, ".las") || iEndsWith(file, ".laz")){
						std::thread thread_init_batch([&](std::string file){
							initLoadPointBatches(file);
						}, file);
						thread_init_batch.detach();
					}
				}
			);
		} else {
			sequential(files);
		}
	});

	// Create temporary folder
	std::filesystem::create_directories(TEMPORARY_DIRECTORY);

	initScene();

	if(CPU_PARALLELISED){
		// Loading points routine
		std::thread thread_loading_points(loadPointcloudRoutine);
		thread_loading_points.detach();

		// Octree update routine
		std::thread thread_octree_update([&](){
			updateOctreeRoutine();
		});
		thread_octree_update.detach();

		// Clear unused batches routine
		std::thread thread_clear_batches(clearUnusedBatchesRoutine);
		thread_clear_batches.detach();

		std::thread thread_load_points_on_gpu([&](CuRast* editor){
			while(true){
				loadBatchesOnGPU(editor, &context);
			}
		}, CuRast::instance);
		thread_load_points_on_gpu.detach();
	}

	bool was_unified_set = CuRastSettings::useUnifiedMemory;

	VKRenderer::loop(
		[&]() {
			update();
			CuRast::instance->update();
			
			{
				if(!was_unified_set && CuRastSettings::useUnifiedMemory){
					println("Using unified memory...");
					cudaDeviceSynchronize();
				}
				was_unified_set = CuRastSettings::useUnifiedMemory;
			}

			DeviceState* state = CuRast::instance->deviceState;
			double stage1_millies = double(state->nanotime_stage_1 - state->nanotime_start) / 1'000'000.0;
			double stage2_millies = double(state->nanotime_stage_2 - state->nanotime_stage_1) / 1'000'000.0;
			double stage3_millies = double(state->nanotime_stage_3 - state->nanotime_stage_2) / 1'000'000.0;
			Runtime::debugValues["stage 1"] = format("{:.3f}", stage1_millies);
			Runtime::debugValues["stage 2"] = format("{:.3f}", stage2_millies);
			Runtime::debugValues["stage 3"] = format("{:.3f}", stage3_millies);

			// TODO: to remove
			{
				static AABB* test_stored_aabb = nullptr;

				// Testing stuff
				if(CuRastSettings::storeOctree){
					// mainOctree->display();
					println("Start storing octree");
					storeOctree(mainOctree.get());
					test_stored_aabb = mainOctree->aabb;
					println("Done storing octree");
					CuRastSettings::storeOctree = false;
				}
				if(CuRastSettings::loadOctree){
					println("Start loading octree");
					OctreeNode* octree = loadOctree(*test_stored_aabb);
					println("Done loading octree");
					CuRastSettings::loadOctree = false;
					
					if(*mainOctree == *octree){
						println("loaded == original, serialisation / deserialisation worked");
					} else {
						println("ERROR: loaded != original, serialisation / deserialisation failed");
					}

					mainOctree = std::shared_ptr<OctreeNode>(octree);

					cuCtxSetCurrent(context);
					loadOctreeOnGPU(CuRast::instance, &context, true);
				}
			}

			// // TODO: to remove
			// {
			// 	static OctreeNode* random_node = nullptr; 
			// 	static uint32_t random_depth = 0;
			// 	static std::vector<NodePosition> random_path = {};

			// 	// Testing stuff
			// 	if(CuRastSettings::storeOctree){
			// 		println("Start getting random node");
			// 		random_node = mainOctree.get();
			// 		random_device rd;
			// 		mt19937 gen(rd());
			// 		uniform_int_distribution<> distrib(2, mainOctree->getDepth());
			// 		random_depth = distrib(gen);
			// 		random_path = {};

			// 		for(uint32_t i=1; i<random_depth; i++){
			// 			uniform_int_distribution<> child_distrib(0, 8);
			// 			uint32_t random_child = child_distrib(gen);
			// 			bool found = false;
			// 			for(uint32_t j=0; j<8; j++){
			// 				uint32_t cur_child = (random_child + j) % 8;
			// 				if(random_node->children[cur_child]){
			// 					random_path.push_back((NodePosition)cur_child);
			// 					random_node = random_node->children[cur_child];
			// 					found = true;
			// 					break;
			// 				}
			// 			}
			// 			if(!found){break;}
			// 		}
			// 		printf("Random path: ");
			// 		for(uint32_t i=0; i<random_path.size(); i++){
			// 			printf("%d, ", random_path[i]);
			// 		}
			// 		println();

			// 		// mainOctree->display();
			// 		println("Start storing octree");
			// 		storeOctree(random_node, true);
			// 		println("Done storing octree");
			// 		CuRastSettings::storeOctree = false;
			// 	}
			// 	if(CuRastSettings::loadOctree && random_node){
			// 		println("Start loading octree");
			// 		OctreeNode* octree = loadOctree(*random_node->aabb, true);
			// 		CuRastSettings::loadOctree = false;

			// 		mainOctree->display();

			// 		println("Loaded single node:");
			// 		octree->display(random_path.back(), random_path.size(), true);
			// 		println();

			// 		random_node = mainOctree.get();
			// 		for(uint32_t i=0; i<random_path.size()-1; i++){
			// 			random_node = random_node->children[random_path[i]];
			// 		}
			// 		println("Loaded subtree:");
			// 		for(uint32_t i=0; i<8; i++){
			// 			// octree->children[i] = random_node->children[random_path.back()]->children[i];
			// 			if(random_node->children[random_path.back()]->children[i]){
			// 				const AABB& aabb = *random_node->children[random_path.back()]->children[i]->aabb;
			// 				if(LRUCache::hasBeenStored(aabb)){
			// 					octree->children[i] = loadOctree(aabb);
			// 				} else {
			// 					octree->children[i] = random_node->children[random_path.back()]->children[i];
			// 				}
			// 				octree->children[i]->display(i, random_path.size()+1);
			// 			}
			// 		}
			// 		println("Done loading octree");

			// 		if(*random_node->children[random_path.back()] == *octree){
			// 			println("loaded == original, serialisation / deserialisation worked");
			// 		} else {
			// 			println("ERROR: loaded != original, serialisation / deserialisation failed");
			// 		}
			// 		random_node->children[random_path.back()] = octree;


			// 		cuCtxSetCurrent(context);
			// 		loadOctreeOnGPU(CuRast::instance, &context, true);
			// 	}
			// }


			bool should_update = false;
			// {
			// 	std::lock_guard<std::mutex> lock(isUpdatingMtx);
			// 	if(lodUpdated){
			// 		should_update = true;
			// 		lodUpdated = false;
			// 	}
			// }

			if(should_update || elapsedFrames >= SEND_DATA_EVERY_X_FRAMES){
				elapsedFrames = 0;
				updateVisibilityCache(VKRenderer::view.view, VKRenderer::view.proj);
				loadOctreeOnGPU(CuRast::instance, &context);
			}

			freeOctreesOnGPU(CuRast::instance);
			elapsedFrames++;

			// { // TODO: remove, just for debugging

			// 	// https://forums.developer.nvidia.com/t/best-way-to-report-memory-consumption-in-cuda/21042
			// 	static double freeDB = 0.;
			// 	uint64_t free_byte, total_byte = 0;
			// 	double free_db, total_db, used_db = 0.;

			// 	CURuntime::assertCudaSuccess(cuMemGetInfo(&free_byte, &total_byte));
			// 	free_db = (double)free_byte; total_db = (double)total_byte; used_db = total_db - free_db;
			// 	free_db /= (1024 * 1024); total_db /= (1024 * 1024); used_db /= (1024 * 1024);
			// 	// Only display if changes bigger than X Mb
			// 	if(abs(freeDB - floor(free_db)) >= 10){
			// 		println("GPU usage\n    Total: {:L} Mb\n    InUse: {:L} Mb\n    Available: {:L} Mb",
			// 			total_db, used_db, free_db
			// 		);
			// 		freeDB = floor(free_db);
			// 	}
			// }
		},
		[&]() {
			CuRast::instance->render();
		},
		[&]() {CuRast::instance->postFrame();}
	);

	{
		std::lock_guard<std::mutex> lock(mainLoopIsTerminatingMtx);
		MAIN_LOOP_IS_TERMINATING = true;
		// Destroy temporary folder
		std::filesystem::remove_all(TEMPORARY_DIRECTORY);
	}

	displayTimings();
	displayBuffers();

	if(mainOctreeCpy){
		delete(mainOctreeCpy);
		mainOctreeCpy = nullptr;
	}

	VKRenderer::destroy();
}
