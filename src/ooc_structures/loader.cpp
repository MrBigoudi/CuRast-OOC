#include "loader.h"

#include "laszip/laszip_api.h"
#include "globals.h"
#include "gpuVersion.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <execution>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <thread>


void initLoadPointBatches(string file,
    std::deque<std::shared_ptr<PointBatch>>& batches_queue,
    std::deque<std::mutex>& batches_queue_mutexes
){
    std::shared_ptr<Timing> timing = Timing::addTiming(format("init load file: {}", file), true);

	// Basic checks
	if(!fs::exists(file)){
		println("ERROR: file '{}' does not exist", file);
		return;
	}
	if(!iEndsWith(file, "las") && !iEndsWith(file, "laz")){
		println("ERROR: file '{}' doesn't have a supported file type", file);
		return;
	}

	// Load header
	laszip_POINTER laszip_reader;
	if(laszip_create(&laszip_reader)){
		println("ERROR: creating laszip reader for '{}'", file);
		return;
	}
	laszip_BOOL is_compressed = 0;
	{
        std::lock_guard<std::mutex> lock(LoaderGpuVersion::laszipReaderMtx);
        if(laszip_open_reader(laszip_reader, file.c_str(), &is_compressed)){
            laszip_destroy(laszip_reader);
            return;
        }
    }
	laszip_header* header;
	if(laszip_get_header_pointer(laszip_reader, &header)){
		println("ERROR: getting laszip header pointer for '{}'", file);
		laszip_close_reader(laszip_reader);
		laszip_destroy(laszip_reader);
		return;
	}
	std::shared_ptr<laszip_header> shared_header = std::make_shared<laszip_header>(*header);
	std::shared_ptr<string> shared_file = std::make_shared<string>(file);

	// Create batches
	uint64_t num_points = header->number_of_point_records ? header->number_of_point_records : header->extended_number_of_point_records;

	for(uint64_t first_point = 0; first_point < num_points; first_point += OocSimLodSettings::MAX_POINTS_PER_BATCHES){

		std::shared_ptr<PointBatch> new_batch = std::make_shared<PointBatch>();
		new_batch->file = shared_file;
		new_batch->header = shared_header;
		new_batch->first = first_point;
		new_batch->count = std::min(num_points - first_point, uint64_t(OocSimLodSettings::MAX_POINTS_PER_BATCHES));
		new_batch->state = BatchState::ToLoad;

		// Find the index where to put the new batch
		// If no space is free on the queue, wait until space is found
		while(true){
			bool found = false;
			for(uint32_t i=0; i<OocSimLodSettings::BATCHES_LIST_SIZE; i++){
				std::lock_guard<std::mutex> lock(batches_queue_mutexes[i]);
				if(batches_queue[i]){continue;}
				batches_queue[i] = new_batch;
				found = true;
				break;
			}
			if(found){break;}

			if(!OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
				uint32_t old_queue_size = OocSimLodSettings::BATCHES_LIST_SIZE;
				OocSimLodSettings::BATCHES_LIST_SIZE *= 2;
				batches_queue.resize(OocSimLodSettings::BATCHES_LIST_SIZE);
				batches_queue_mutexes.resize(OocSimLodSettings::BATCHES_LIST_SIZE);
			} else {
				// Wait a bit to give time for the queue to be emptied
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		}
	}

    timing->stop_clock();

}






void loadBatchesOnGPU(CuRast* editor, CUcontext* ctx,
    std::deque<std::shared_ptr<PointBatch>>& batches_queue,
    std::deque<std::mutex>& batches_queue_mutexes
){
	std::vector<uint32_t> batches_indices(OocSimLodSettings::MAX_BATCHES_PER_GPU_LOAD, 0);
	uint32_t last_index = 0;

	for(uint32_t i=0; i<OocSimLodSettings::BATCHES_LIST_SIZE; i++){
		std::lock_guard<std::mutex> lock(batches_queue_mutexes[i]);
		if(batches_queue[i] && batches_queue[i]->state == BatchState::Inserted){
			batches_indices[last_index] = i;
			last_index++;
			if(last_index >= OocSimLodSettings::MAX_BATCHES_PER_GPU_LOAD){break;}
		}
	}
	if(last_index == 0){return;}
	static uint32_t lastGPULoadAttempt = 0;
	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL && batches_indices.size() < OocSimLodSettings::MIN_BATCHES_PER_GPU_LOAD){
		if(lastGPULoadAttempt < OocSimLodSettings::MAX_ATTEMPTS_BEFORE_IGNORING_MIN_VARIABLES){
			lastGPULoadAttempt++;
			return;
		} 
	}
	lastGPULoadAttempt = 0;

    std::shared_ptr<Timing> timing = Timing::addTiming("send points to GPU memory", true);

	std::mutex mtx_counter;

	auto lambda = [&](uint32_t index){
		std::lock_guard<std::mutex> lock(batches_queue_mutexes[index]);
		std::shared_ptr<PointBatch> batch = batches_queue[index];

        // Upload positions and colors to GPU
        CUdeviceptr cptr_positions, cptr_colors;

		if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
			cuCtxSetCurrent(*ctx);
		}

		// // Check if enough memory is available
		// // TODO: Debug display to remove
		// uint64_t free_byte = 0;
		// uint64_t total_byte = 0;
		// double free_db, total_db, used_db = 0.;
		// CUresult cuda_status = CUDA_SUCCESS;
		// const char* name = nullptr;
		// const char* desc = nullptr;
		// {
		// 	cuda_status = cuMemGetInfo(&free_byte, &total_byte);
		// 	if(cuda_status != CUDA_SUCCESS){
		// 		cuGetErrorName(cuda_status, &name);
		// 		cuGetErrorString(cuda_status, &desc);
		// 		println(stderr, "Error: cuMemGetInfo failed before, {} ({}): {}\n ",
		// 			int(cuda_status),
		// 			name ? name : "unknown",
		// 			desc ? desc : "unknown"
		// 		);
		// 		throw(EXIT_FAILURE);
		// 	}
		// 	free_db = (double)free_byte;
		// 	total_db = (double)total_byte;
		// 	used_db = total_db - free_db;
		// }

		// size_t positions_sizes = batch->count * sizeof(vec3);
		// size_t colors_sizes = batch->count * sizeof(uint32_t);
		// float threshold = 10.f;
		// double requested_db = threshold * (positions_sizes + colors_sizes);
		// println("free: {} Mb, requested: {} Mb",
		// 	free_db/1024/1024, requested_db/1024/1024
		// );
		// if(free_db < requested_db){
		// 	println("Can't load more points to the GPU; insufficient memory available\n    memory usage: used = {} Mb, free = {} Mb, total = {} Mb",
		// 		used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0
		// 	);
		// 	return;
		// }

        // cuMemAlloc(&cptr_positions, positions_sizes);
        // cuMemAlloc(&cptr_colors, colors_sizes);
        // cuMemcpyHtoD(cptr_positions, batch->getPositions().data(), positions_sizes);
        // cuMemcpyHtoD(cptr_colors, batch->getColors().data(),colors_sizes);

        // auto node = make_shared<SNCPoints>("pointcloud");
        // node->cptr_positions = cptr_positions;
        // node->cptr_colors    = cptr_colors;
        // node->numPoints      = batch->count;

		// {
		// 	// Unified memory rendering
		// 	auto tmp_positions = std::make_shared<std::vector<vec3>>(batch->getPositions());
		// 	auto tmp_colors = std::make_shared<std::vector<uint32_t>>(batch->getColors());
		// 	unified_positions.push_back(tmp_positions);
		// 	unified_colors.push_back(tmp_colors);
		// 	node->ptr_positions = tmp_positions.get()->data();
		// 	node->ptr_colors    = tmp_colors.get()->data();
		// }

		batch->state = BatchState::ToRemove;

		{
			std::lock_guard<std::mutex> lock_counter(mtx_counter);
			GlobalVariables::nbPoints += batch->count;
		}

		// std::lock_guard<std::mutex> lock_scene(updateSceneMutex);
        // editor->scene.world->children.push_back(node);
    };

	auto first = batches_indices.begin();
	auto last = first + last_index;
	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		std::for_each(std::execution::par, first, last, lambda);
	} else {
		std::for_each(first, last, lambda);
	}

    timing->stop_clock();
}


void loadPointcloudRoutine(
    std::deque<std::shared_ptr<PointBatch>>& batches_queue,
    std::deque<std::mutex>& batches_queue_mutexes
){
    while(true){
        loadPointsInBatches(batches_queue, batches_queue_mutexes);
		if(GlobalVariables::mainLoopIsTerminating){return;}
    }
};



void clearUnusedBatchesRoutine(
    std::deque<std::shared_ptr<PointBatch>>& batches_queue,
    std::deque<std::mutex>& batches_queue_mutexes
){
	while(true){
		clearUnusedBatches(batches_queue, batches_queue_mutexes);
		if(GlobalVariables::mainLoopIsTerminating){return;}
	}
}































/////////////////////////////////////////////////////////////////
////////////////////////// GPU VERSION //////////////////////////
/////////////////////////////////////////////////////////////////

vec3 LoaderGpuVersion::batchHeaderCenter(const laszip_header& h){
    return vec3(
        static_cast<float>((h.min_x + h.max_x) * 0.5),
        static_cast<float>((h.min_y + h.max_y) * 0.5),
        static_cast<float>((h.min_z + h.max_z) * 0.5)
    );
}

// Precise position for a batch that has already been loaded: centroid of its points.
vec3 LoaderGpuVersion::batchPointsCentroid(const std::shared_ptr<PointBatch>& batch){
    if(batch->points && !batch->points->empty()){
        vec3 sum(0.0f);
        for(const auto& p : *batch->points){ sum += p.position; }
        return sum / static_cast<float>(batch->points->size());
    }
    return batchHeaderCenter(*batch->header);
}

void LoaderGpuVersion::updateCameraSnapshot(){
    const mat4 view = VKRenderer::view.view;
    vec3 pos = vec3(glm::inverse(view) * vec4(0.0f, 0.0f, 0.0f, 1.0f));
    cameraPosX.store(pos.x, std::memory_order_relaxed);
    cameraPosY.store(pos.y, std::memory_order_relaxed);
    cameraPosZ.store(pos.z, std::memory_order_relaxed);
}

vec3 LoaderGpuVersion::getCameraSnapshot(){
    return vec3(
        cameraPosX.load(std::memory_order_relaxed),
        cameraPosY.load(std::memory_order_relaxed),
        cameraPosZ.load(std::memory_order_relaxed)
    );
}

void loadPointsInBatches(
    std::deque<std::shared_ptr<PointBatch>>& batches_queue,
    std::deque<std::mutex>& batches_queue_mutexes
){
	// std::vector<uint32_t> batches_indices(OocSimLodSettings::MAX_BATCHES_PER_LOAD, 0);
	// uint32_t last_index = 0;
	
	// for(uint32_t i=0; i<OocSimLodSettings::BATCHES_LIST_SIZE; i++){
	// 	std::lock_guard<std::mutex> lock(batches_queue_mutexes[i]);
	// 	if(batches_queue[i] && batches_queue[i]->state == BatchState::ToLoad){
	// 		batches_indices[last_index] = i;
	// 		last_index++;
	// 		if(last_index >= OocSimLodSettings::MAX_BATCHES_PER_LOAD){break;}
	// 	}
	// }
	// if(last_index == 0){return;}
	// static uint32_t lastLoadAttempt = 0;
	// if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL && batches_indices.size() < OocSimLodSettings::MIN_BATCHES_PER_LOAD){
	// 	if(lastLoadAttempt < OocSimLodSettings::MAX_ATTEMPTS_BEFORE_IGNORING_MIN_VARIABLES){
	// 		lastLoadAttempt++;
	// 		return;
	// 	} 
	// }
	// lastLoadAttempt = 0;

	struct Candidate { uint32_t index; float dist2; };
    std::vector<Candidate> candidates;
    candidates.reserve(OocSimLodSettings::BATCHES_LIST_SIZE);

    vec3 camera_pos = LoaderGpuVersion::getCameraSnapshot();

    for(uint32_t i=0; i<OocSimLodSettings::BATCHES_LIST_SIZE; i++){
        std::lock_guard<std::mutex> lock(batches_queue_mutexes[i]);
        std::shared_ptr<PointBatch>& batch = batches_queue[i];
        if(batch && batch->state == BatchState::ToLoad){
            vec3 d = LoaderGpuVersion::batchHeaderCenter(*batch->header) - camera_pos;
            candidates.push_back({i, glm::dot(d, d)});
        }
    }
    if(candidates.empty()){return;}

    // Sort by distance to the camera, closest first
    std::sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b){ return a.dist2 < b.dist2; });

    uint32_t last_index = std::min<uint32_t>(
        (uint32_t)candidates.size(), OocSimLodSettings::MAX_BATCHES_PER_LOAD
    );

    std::vector<uint32_t> batches_indices(last_index);
    for(uint32_t k=0; k<last_index; k++){
        batches_indices[k] = candidates[k].index;
    }

    static uint32_t lastLoadAttempt = 0;
    if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL && candidates.size() < OocSimLodSettings::MIN_BATCHES_PER_LOAD){
        if(lastLoadAttempt < OocSimLodSettings::MAX_ATTEMPTS_BEFORE_IGNORING_MIN_VARIABLES){
            lastLoadAttempt++;
            return;
        }
    }
    lastLoadAttempt = 0;

    std::shared_ptr<Timing> timing = Timing::addTiming("load points in batches", true);

    auto lambda = [&](uint32_t index){
        std::shared_ptr<PointBatch> batch;
        {
            std::lock_guard<std::mutex> lock(batches_queue_mutexes[index]);
            batch = batches_queue[index];
            if(!batch || batch->state != BatchState::ToLoad){ return; }
            batch->state = BatchState::Loading;
        }

        // Get or create a mutex for this specific file
        std::mutex* file_mutex = nullptr;
        {
            std::lock_guard<std::mutex> lock(LoaderGpuVersion::perFileMutexesMtx);
            auto it = LoaderGpuVersion::perFileMutexes.find(*batch->file);
            if(it == LoaderGpuVersion::perFileMutexes.end()){
                printf("ERROR: no file mutex for '%s' — was createNewBatches called?\n", batch->file->c_str());
                throw(EXIT_FAILURE);
            }
            file_mutex = &it->second;
        }

        laszip_POINTER laszip_reader;
        if(laszip_create(&laszip_reader)){
            std::lock_guard<std::mutex> lock(batches_queue_mutexes[index]);
            batch->state = BatchState::ToLoad;
            return;
        }

        laszip_BOOL is_compressed = 0;
        laszip_point* laz_point;

        {
            std::lock_guard<std::mutex> file_lock(*file_mutex);
            if(laszip_open_reader(laszip_reader, (*batch->file).c_str(), &is_compressed)){
                laszip_destroy(laszip_reader);
                return;
            }
            if(laszip_get_point_pointer(laszip_reader, &laz_point)){
                laszip_close_reader(laszip_reader);
                laszip_destroy(laszip_reader);
                return;
            }
            if(laszip_seek_point(laszip_reader, batch->first)){
                laszip_close_reader(laszip_reader);
                laszip_destroy(laszip_reader);
                return;
            }

            double scale_x = batch->header->x_scale_factor;
            double scale_y = batch->header->y_scale_factor;
            double scale_z = batch->header->z_scale_factor;
            double offset_x = batch->header->x_offset;
            double offset_y = batch->header->y_offset;
            double offset_z = batch->header->z_offset;

            uint8_t fmt = batch->header->point_data_format;
            bool has_rgb = (fmt == 2 || fmt == 3 || fmt == 5 || fmt == 7 || fmt == 8 || fmt == 10);
            batch->points = std::make_shared<vector<Point>>(vector<Point>());

            for (uint64_t i = 0; i < batch->count; i++) {
                if(laszip_read_point(laszip_reader)){
                    println("ERROR: reading point {} for '{}'", i+batch->first, *batch->file);
                    batch->count = i+1;
                    break;
                }

                Point new_point = {};
                float x = (float)(laz_point->X * scale_x + offset_x);
                float y = (float)(laz_point->Y * scale_y + offset_y);
                float z = (float)(laz_point->Z * scale_z + offset_z);
                new_point.position = {x,y,z};

                if(has_rgb){
                    for(size_t j=0; j<3; j++){
                        new_point.color[j] = laz_point->rgb[j] > 255 ? (uint8_t)(laz_point->rgb[j] >> 8) : (uint8_t)laz_point->rgb[j];
                    }
                } else {
                    uint8_t intensity = (uint8_t)(laz_point->intensity >> 8);
                    for(size_t j=0; j<3; j++){
                        new_point.color[j] = intensity;
                    }
                }

                batch->points->push_back(new_point);
				batch->centroid += new_point.position;
            }
			if(batch->count > 0){batch->centroid /= batch->count;}

            laszip_close_reader(laszip_reader);
            laszip_destroy(laszip_reader);
        }

        {
            std::lock_guard<std::mutex> lock(LoaderGpuVersion::loadedBatchesMtx);
            LoaderGpuVersion::loadedBatches.push_back({index, batch});
        }

        {
            std::lock_guard<std::mutex> lock(LoaderGpuVersion::filesRecordMtx);
            LoaderGpuVersion::filesRecord[*batch->file].nb_batches_loaded.fetch_add(1);
        }
    };

    auto first = batches_indices.begin();
    auto last = first + last_index;
    if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
        std::for_each(std::execution::par, first, last, lambda);
    } else {
        std::for_each(first, last, lambda);
    }

    timing->stop_clock();
}

void LoaderGpuVersion::init(){
    CURuntime::assertCudaSuccess(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CURuntime::assertCudaSuccess(cuEventCreate(&eventLoadComplete, CU_EVENT_DISABLE_TIMING));

	batchesQueue = std::deque<std::shared_ptr<PointBatch>>(OocSimLodSettings::BATCHES_LIST_SIZE, nullptr);
    batchesQueueMutexes = std::deque<std::mutex>(OocSimLodSettings::BATCHES_LIST_SIZE);

	batchesOnGpu = std::vector<uint32_t>(OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE, -1);
	batchesOnGpuStatus = GpuVersion::allocHost<uint32_t>(OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE);
	for(uint32_t i = 0; i < OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE; i++){
		((uint32_t*)(batchesOnGpuStatus))[i] = BatchHandled;
	}
}

void LoaderGpuVersion::destroy(){
    CURuntime::assertCudaSuccess(cuEventDestroy(eventLoadComplete));
    CURuntime::assertCudaSuccess(cuMemFreeHost(batchesOnGpuStatus));
    CURuntime::assertCudaSuccess(cuStreamDestroy(stream));
}

void LoaderGpuVersion::fetchFromDevice(){
	cudaStreamSynchronize(stream);

	CURuntime::assertCudaSuccess(cuMemcpyDtoH(
		batchesOnGpuStatus, 
		(CUdeviceptr)GpuVersion::hostStaging.batchesAddedMask,
		OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE * sizeof(uint32_t)
	));

	for(uint32_t i=0; i<OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE; i++){
		if(((uint32_t*)(batchesOnGpuStatus))[i] == BatchHandled){ 
			uint32_t real_index = batchesOnGpu[i];
			batchesOnGpu[i] = -1;
			if(real_index != -1){
				std::lock_guard<std::mutex> lock(batchesQueueMutexes[real_index]);
				std::shared_ptr<PointBatch>& batch = batchesQueue[real_index];
				batch->state = BatchState::ToRemove;

				{
					std::lock_guard<std::mutex> lock(filesRecordMtx);
					filesRecord[*batch->file].nb_batches_inserted.fetch_add(1);
				}
			}
		}
	}
}

bool LoaderGpuVersion::sendToDevice(){
	bool has_send_new_points = false;

    std::vector<CUdeviceptr> srcs_host;
    std::vector<CUdeviceptr> dsts_device;
    std::vector<uint64_t> sizes;
	std::vector<CUdeviceptr> mask_srcs_host;
    std::vector<CUdeviceptr> mask_dsts_device;
    std::vector<uint64_t> mask_sizes;

    std::vector<std::pair<uint32_t, std::shared_ptr<PointBatch>>> ready;
    std::vector<std::pair<uint32_t, std::shared_ptr<PointBatch>>> overview;
    {
        std::lock_guard<std::mutex> lock(loadedBatchesMtx);
        std::swap(ready, loadedBatches);
        std::swap(overview, loadedOverviewBatches);
    }

	// Sort by closest to camera
	vec3 camera_pos = getCameraSnapshot();
	std::sort(ready.begin(), ready.end(),
	[&](const std::pair<uint32_t, std::shared_ptr<PointBatch>>& a,
		const std::pair<uint32_t, std::shared_ptr<PointBatch>>& b){
		glm::vec3 da = a.second->centroid - camera_pos;
		glm::vec3 db = b.second->centroid - camera_pos;
		return glm::dot(da, da) < glm::dot(db, db);
	});

	// The overview batches go before everything else (see `createNewBatchesFromFiles')
	const uint32_t nb_overview = static_cast<uint32_t>(overview.size());
	ready.insert(ready.begin(), overview.begin(), overview.end());

    uint32_t ready_index = 0;
    for(uint32_t i = 0; i < OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE; i++){
        if(batchesOnGpu[i] != -1){continue;}
        if(ready_index >= ready.size()){break;}

        auto& [slot, batch] = ready[ready_index++];
        batchesOnGpu[i] = slot;
        batch->state = BatchState::Sent;
		{
			std::lock_guard<std::mutex> lock(filesRecordMtx);
			filesRecord[*batch->file].nb_batches_on_device.fetch_add(1);
		}

        dsts_device.push_back(((CUdeviceptr*)(GpuVersion::batchesToAddPointsPointers))[i]);
        srcs_host.push_back((CUdeviceptr)(batch->points->data()));
        sizes.push_back(batch->count * sizeof(CPoint));

        dsts_device.push_back((CUdeviceptr)(GpuVersion::hostStaging.batchesToAddCounts) + i * sizeof(uint32_t));
        srcs_host.push_back((CUdeviceptr)(&batch->count));
        sizes.push_back(sizeof(uint32_t));

        mask_dsts_device.push_back((CUdeviceptr)(GpuVersion::hostStaging.batchesAddedMask) + i * sizeof(uint32_t));
        mask_srcs_host.push_back((CUdeviceptr)&GpuVersion::RESET); // == BatchToHandle
        mask_sizes.push_back(sizeof(uint32_t));

        has_send_new_points = true;
    }

    // Put back any ready batches that didn't fit this frame (the overview ones in their own list,
    // so that they stay first)
    if(ready_index < ready.size()){
        const uint32_t overview_end = std::max(ready_index, nb_overview);
        std::lock_guard<std::mutex> lock(loadedBatchesMtx);
        loadedOverviewBatches.insert(
            loadedOverviewBatches.begin(),
            ready.begin() + ready_index, ready.begin() + overview_end
        );
        loadedBatches.insert(
            loadedBatches.begin(),
            ready.begin() + overview_end, ready.end()
        );
    }

    if(has_send_new_points){
        CURuntime::assertCudaSuccess(cuMemcpyBatchAsync(
            dsts_device.data(), srcs_host.data(), sizes.data(), 
			sizes.size(),
            loadingAttributes.data(),
            loadingAttributesIndices.data(),
            loadingAttributes.size(),
            stream
        ));
		CURuntime::assertCudaSuccess(cuMemcpyBatchAsync(
            mask_dsts_device.data(), mask_srcs_host.data(), mask_sizes.data(), 
			mask_sizes.size(),
            loadingAttributes.data(),
            loadingAttributesIndices.data(),
            loadingAttributes.size(),
            stream
        ));
        // CURuntime::assertCudaSuccess(cuEventRecord(eventLoadComplete, stream));
        // cudaStreamWaitEvent(0, eventLoadComplete);
    }

    return has_send_new_points;
}

void clearUnusedBatches(
    std::deque<std::shared_ptr<PointBatch>>& batches_queue,
    std::deque<std::mutex>& batches_queue_mutexes
){
	for(uint32_t i=0; i<OocSimLodSettings::BATCHES_LIST_SIZE; i++){
		std::lock_guard<std::mutex> lock(batches_queue_mutexes[i]);
		if(batches_queue[i] && batches_queue[i]->state == BatchState::ToRemove){
			batches_queue[i] = nullptr;
		}
	}
}

void LoaderGpuVersion::loadingRoutine(){
	enqueueBatches();
	// Clear completed batches
	clearUnusedBatches(batchesQueue, batchesQueueMutexes);
	// Try loading points from disk
	loadPointsInBatches(batchesQueue, batchesQueueMutexes);
}

bool LoaderGpuVersion::run(CuRast* editor, CUcontext* context){
	updateCameraSnapshot();

	// Check if batches are done on GPU side
	fetchFromDevice();

	if(!OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
		loadingRoutine();
	}

	// Get the batches to send to device side
	return sendToDevice();
}


namespace {

/// A file registered for loading (header read, file mutex and record created)
struct LoaderFileEntry {
	std::shared_ptr<std::string> file = nullptr;
	std::shared_ptr<laszip_header> header = nullptr;
	uint64_t num_points = 0;
};

/// Read the header of `file' and register it (first half of the former `createNewBatches')
std::optional<LoaderFileEntry> loaderRegisterFile(const std::string& file){
	// Basic checks
	if(!fs::exists(file)){
		println("ERROR: file '{}' does not exist", file);
		return std::nullopt;
	}
	if(!iEndsWith(file, "las") && !iEndsWith(file, "laz")){
		println("ERROR: file '{}' doesn't have a supported file type", file);
		return std::nullopt;
	}

	// Load header
	laszip_POINTER laszip_reader;
	if(laszip_create(&laszip_reader)){
		println("ERROR: creating laszip reader for '{}'", file);
		return std::nullopt;
	}
	laszip_BOOL is_compressed = 0;
	{
        std::lock_guard<std::mutex> lock(LoaderGpuVersion::laszipReaderMtx);
        if(laszip_open_reader(laszip_reader, file.c_str(), &is_compressed)){
            laszip_destroy(laszip_reader);
            return std::nullopt;
        }
    }
	laszip_header* header;
	if(laszip_get_header_pointer(laszip_reader, &header)){
		println("ERROR: getting laszip header pointer for '{}'", file);
		laszip_close_reader(laszip_reader);
		laszip_destroy(laszip_reader);
		return std::nullopt;
	}
	LoaderFileEntry entry = {};
	entry.header = std::make_shared<laszip_header>(*header);
	entry.file = std::make_shared<string>(file);
	laszip_close_reader(laszip_reader);
	laszip_destroy(laszip_reader);
	header = nullptr;

	entry.num_points = entry.header->number_of_point_records 
		? entry.header->number_of_point_records 
		: entry.header->extended_number_of_point_records
	;

	{
		std::lock_guard<std::mutex> lock(LoaderGpuVersion::perFileMutexesMtx);
		LoaderGpuVersion::perFileMutexes.try_emplace(*entry.file);
		{
			std::lock_guard<std::mutex> lock_record(LoaderGpuVersion::filesRecordMtx);
			LoaderGpuVersion::filesRecord[file].header = entry.header;
			uint32_t nb_batches = (entry.num_points + OocSimLodSettings::MAX_POINTS_PER_BATCHES - 1) / OocSimLodSettings::MAX_POINTS_PER_BATCHES;
			LoaderGpuVersion::filesRecord[file].nb_batches += nb_batches;
		}
	}
	return entry;
}

/// Create the regular batches of a registered file (second half of the former `createNewBatches')
void loaderCreateFileBatches(const LoaderFileEntry& entry){
	std::lock_guard<std::mutex> lock(LoaderGpuVersion::batchesToEnqueueMtx);
	for(uint64_t first_point = 0; first_point < entry.num_points; first_point += OocSimLodSettings::MAX_POINTS_PER_BATCHES){

		std::shared_ptr<PointBatch> new_batch = std::make_shared<PointBatch>();
		new_batch->file = entry.file;
		new_batch->header = entry.header;
		new_batch->first = first_point;
		new_batch->count = std::min(entry.num_points - first_point, uint64_t(OocSimLodSettings::MAX_POINTS_PER_BATCHES));
		new_batch->state = BatchState::ToLoad;

		LoaderGpuVersion::batchesToEnqueue.push_back(new_batch);
	}
}

/// Chunk size of a LAZ file, read from its LASzip VLR ("laszip encoded", record 22204: the chunk
/// size is the U32 at byte 12 of the payload)
/// The VLR is read from the file itself: the LASzip DLL may remove it from the header it exposes
/// Returns 0 if not found (uncompressed file, unreadable file), U32_MAX for variable-size chunks
uint32_t loaderReadLazChunkSize(const std::string& file){
	std::ifstream stream(file, std::ios::binary);
	if(!stream.is_open()){return 0;}

	auto read_bytes = [&](uint64_t offset, void* dst, uint64_t size) -> bool {
		stream.seekg(static_cast<std::streamoff>(offset));
		stream.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(size));
		return static_cast<uint64_t>(stream.gcount()) == size;
	};

	// Public header block: signature, header size (offset 94), number of VLRs (offset 100)
	char signature[4] = {};
	uint16_t header_size = 0;
	uint32_t nb_vlrs = 0;
	if(!read_bytes(0, signature, 4) || std::memcmp(signature, "LASF", 4) != 0){return 0;}
	if(!read_bytes(94, &header_size, sizeof(header_size))){return 0;}
	if(!read_bytes(100, &nb_vlrs, sizeof(nb_vlrs))){return 0;}

	// VLR header: reserved U16, user_id char[16], record_id U16, record_length_after_header U16,
	// description char[32] (54 bytes)
	uint64_t offset = header_size;
	for(uint32_t i = 0; i < nb_vlrs; i++){
		uint8_t vlr_header[54] = {};
		if(!read_bytes(offset, vlr_header, sizeof(vlr_header))){return 0;}
		char user_id[17] = {};
		std::memcpy(user_id, vlr_header + 2, 16);
		uint16_t record_id = 0;
		uint16_t record_length = 0;
		std::memcpy(&record_id, vlr_header + 18, sizeof(record_id));
		std::memcpy(&record_length, vlr_header + 20, sizeof(record_length));

		if(std::strncmp(user_id, "laszip encoded", 16) == 0 && record_id == 22204 && record_length >= 16){
			uint32_t chunk_size = 0;
			if(!read_bytes(offset + 54 + 12, &chunk_size, sizeof(chunk_size))){return 0;}
			return chunk_size;
		}
		offset += 54 + uint64_t(record_length);
	}
	return 0;
}

/// Same conversion as `loadPointsInBatches'
Point loaderToPoint(const laszip_point* laz_point, const laszip_header& header){
	Point new_point = {};
	float x = (float)(laz_point->X * header.x_scale_factor + header.x_offset);
	float y = (float)(laz_point->Y * header.y_scale_factor + header.y_offset);
	float z = (float)(laz_point->Z * header.z_scale_factor + header.z_offset);
	new_point.position = {x,y,z};

	uint8_t fmt = header.point_data_format;
	bool has_rgb = (fmt == 2 || fmt == 3 || fmt == 5 || fmt == 7 || fmt == 8 || fmt == 10);
	if(has_rgb){
		for(size_t j=0; j<3; j++){
			new_point.color[j] = laz_point->rgb[j] > 255 ? (uint8_t)(laz_point->rgb[j] >> 8) : (uint8_t)laz_point->rgb[j];
		}
	} else {
		uint8_t intensity = (uint8_t)(laz_point->intensity >> 8);
		for(size_t j=0; j<3; j++){
			new_point.color[j] = intensity;
		}
	}
	return new_point;
}

/// The overview sample of a file: the first point of every LAZ chunk (stored uncompressed, so
/// seeking to it and reading it decompresses nothing else), or every `OVERVIEW_FALLBACK_STRIDE'-th
/// point for uncompressed LAS files and LAZ files with variable-size chunks
std::vector<Point> loaderReadChunkPoints(const LoaderFileEntry& entry){
	std::vector<Point> points = {};
	if(entry.num_points == 0){return points;}

	std::mutex* file_mutex = nullptr;
	{
		std::lock_guard<std::mutex> lock(LoaderGpuVersion::perFileMutexesMtx);
		file_mutex = &LoaderGpuVersion::perFileMutexes[*entry.file];
	}
	std::lock_guard<std::mutex> file_lock(*file_mutex);

	laszip_POINTER laszip_reader;
	if(laszip_create(&laszip_reader)){return points;}
	laszip_BOOL is_compressed = 0;
	{
		std::lock_guard<std::mutex> lock(LoaderGpuVersion::laszipReaderMtx);
		if(laszip_open_reader(laszip_reader, entry.file->c_str(), &is_compressed)){
			laszip_destroy(laszip_reader);
			return points;
		}
	}
	laszip_point* laz_point = nullptr;
	if(laszip_get_point_pointer(laszip_reader, &laz_point)){
		laszip_close_reader(laszip_reader);
		laszip_destroy(laszip_reader);
		return points;
	}

	uint64_t stride = LoaderGpuVersion::OVERVIEW_FALLBACK_STRIDE;
	if(is_compressed){
		const uint32_t chunk_size = loaderReadLazChunkSize(*entry.file);
		if(chunk_size != 0 && chunk_size != std::numeric_limits<uint32_t>::max()){
			stride = chunk_size;
		}
	}
	stride = std::max<uint64_t>(stride, 1);

	points.reserve((entry.num_points + stride - 1) / stride);
	for(uint64_t index = 0; index < entry.num_points; index += stride){
		if(laszip_seek_point(laszip_reader, static_cast<laszip_I64>(index))){break;}
		if(laszip_read_point(laszip_reader)){break;}
		points.push_back(loaderToPoint(laz_point, *entry.header));
	}

	laszip_close_reader(laszip_reader);
	laszip_destroy(laszip_reader);
	return points;
}

/// Union of the header bounding boxes of the registered files (headers with invalid bounds are
/// skipped); false if there is none
bool loaderHeadersBounds(const std::vector<std::optional<LoaderFileEntry>>& entries,
	glm::vec3& mins, glm::vec3& maxs
){
	double min_x =  std::numeric_limits<double>::infinity(), min_y = min_x, min_z = min_x;
	double max_x = -std::numeric_limits<double>::infinity(), max_y = max_x, max_z = max_x;
	bool has_bounds = false;
	for(const std::optional<LoaderFileEntry>& entry : entries){
		if(!entry){continue;}
		const laszip_header& h = *entry->header;
		const bool is_valid = std::isfinite(h.min_x) && std::isfinite(h.min_y) && std::isfinite(h.min_z)
			&& std::isfinite(h.max_x) && std::isfinite(h.max_y) && std::isfinite(h.max_z)
			&& h.min_x <= h.max_x && h.min_y <= h.max_y && h.min_z <= h.max_z;
		if(!is_valid){
			println("WARNING: '{}' has invalid header bounds, not used for the initial octree root", *entry->file);
			continue;
		}
		min_x = std::min(min_x, h.min_x); min_y = std::min(min_y, h.min_y); min_z = std::min(min_z, h.min_z);
		max_x = std::max(max_x, h.max_x); max_y = std::max(max_y, h.max_y); max_z = std::max(max_z, h.max_z);
		has_bounds = true;
	}
	if(!has_bounds){return false;}
	// Same conversion as the points (double to float)
	mins = glm::vec3(static_cast<float>(min_x), static_cast<float>(min_y), static_cast<float>(min_z));
	maxs = glm::vec3(static_cast<float>(max_x), static_cast<float>(max_y), static_cast<float>(max_z));
	return true;
}

/// Mix the overview samples of all the files into batches that are sent before any other batch
void loaderCreateOverviewBatches(const std::vector<std::optional<LoaderFileEntry>>& entries,
	const std::vector<std::vector<Point>>& samples
){
	uint64_t nb_samples = 0;
	for(const std::vector<Point>& file_samples : samples){ nb_samples += file_samples.size(); }
	if(nb_samples == 0){return;}

	// All the samples, shuffled so that every overview batch spans every file
	std::vector<Point> all_samples = {};
	all_samples.reserve(nb_samples);
	for(const std::vector<Point>& file_samples : samples){
		all_samples.insert(all_samples.end(), file_samples.begin(), file_samples.end());
	}
	std::shuffle(all_samples.begin(), all_samples.end(), std::minstd_rand(0x5EEDu));

	// Header covering every file: lets the file explorer frame the whole data set
	uint32_t nb_files = 0;
	std::shared_ptr<laszip_header> overview_header = nullptr;
	for(const std::optional<LoaderFileEntry>& entry : entries){
		if(!entry){continue;}
		nb_files++;
		const laszip_header& h = *entry->header;
		if(!overview_header){
			overview_header = std::make_shared<laszip_header>(h);
			// Not owned by this copy
			overview_header->number_of_variable_length_records = 0;
			overview_header->vlrs = nullptr;
			overview_header->user_data_in_header_size = 0;
			overview_header->user_data_in_header = nullptr;
			overview_header->user_data_after_header_size = 0;
			overview_header->user_data_after_header = nullptr;
			continue;
		}
		overview_header->min_x = std::min(overview_header->min_x, h.min_x);
		overview_header->min_y = std::min(overview_header->min_y, h.min_y);
		overview_header->min_z = std::min(overview_header->min_z, h.min_z);
		overview_header->max_x = std::max(overview_header->max_x, h.max_x);
		overview_header->max_y = std::max(overview_header->max_y, h.max_y);
		overview_header->max_z = std::max(overview_header->max_z, h.max_z);
	}

	// Even split: every batch holds at most MAX_POINTS_PER_BATCHES points
	const uint64_t batch_size = OocSimLodSettings::MAX_POINTS_PER_BATCHES;
	const uint64_t nb_batches = (nb_samples + batch_size - 1) / batch_size;

	// A record of its own in the file explorer (the batches belong to no single file)
	static std::atomic<uint32_t> nb_overviews{0};
	const std::string name = format("[Overview #{}]/{} chunk points of {} files",
		nb_overviews.fetch_add(1) + 1, nb_samples, nb_files
	);
	std::shared_ptr<std::string> shared_name = std::make_shared<std::string>(name);
	{
		std::lock_guard<std::mutex> lock(LoaderGpuVersion::filesRecordMtx);
		LoaderGpuVersion::filesRecord[name].header = overview_header;
		LoaderGpuVersion::filesRecord[name].nb_batches += static_cast<uint32_t>(nb_batches);
		LoaderGpuVersion::filesRecord[name].nb_batches_loaded.fetch_add(static_cast<uint32_t>(nb_batches));
	}

	std::vector<std::pair<uint32_t, std::shared_ptr<PointBatch>>> overview_batches = {};
	for(uint64_t b = 0; b < nb_batches; b++){
		const uint64_t begin = (b * nb_samples) / nb_batches;
		const uint64_t end = ((b + 1) * nb_samples) / nb_batches;

		std::shared_ptr<PointBatch> new_batch = std::make_shared<PointBatch>();
		new_batch->file = shared_name;
		new_batch->header = overview_header;
		new_batch->first = 0;
		new_batch->count = static_cast<uint32_t>(end - begin);
		// Already in memory: never picked by the disk loader (only `ToLoad' batches are), it goes
		// through the same states as a loaded batch from here (Sent, ToRemove)
		new_batch->state = BatchState::Loading;
		new_batch->points = std::make_shared<vector<Point>>(all_samples.begin() + begin, all_samples.begin() + end);
		new_batch->centroid = vec3(0.0f);
		for(const Point& point : *new_batch->points){ new_batch->centroid += point.position; }
		if(new_batch->count > 0){ new_batch->centroid /= static_cast<float>(new_batch->count); }

		// A slot in the batches queue (the device side refers to batches by slot)
		while(true){
			bool found = false;
			for(uint32_t i = 0; i < OocSimLodSettings::BATCHES_LIST_SIZE; i++){
				std::lock_guard<std::mutex> lock(LoaderGpuVersion::batchesQueueMutexes[i]);
				if(LoaderGpuVersion::batchesQueue[i]){continue;}
				LoaderGpuVersion::batchesQueue[i] = new_batch;
				overview_batches.push_back({i, new_batch});
				found = true;
				break;
			}
			if(found){break;}
			// Wait for the queue to be emptied
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}

	// All at once, so that they are sent together
	std::lock_guard<std::mutex> lock(LoaderGpuVersion::loadedBatchesMtx);
	LoaderGpuVersion::loadedOverviewBatches.insert(LoaderGpuVersion::loadedOverviewBatches.end(),
		overview_batches.begin(), overview_batches.end()
	);
}

} // namespace


void LoaderGpuVersion::createNewBatchesFromFiles(std::vector<std::string> files){
	std::vector<uint32_t> indices(files.size());
	std::iota(indices.begin(), indices.end(), 0u);
	auto for_each_file = [&](auto func){
		if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL){
			std::for_each(std::execution::par, indices.begin(), indices.end(), func);
		} else {
			std::for_each(indices.begin(), indices.end(), func);
		}
	};

	// 1. Register every file (headers)
	std::vector<std::optional<LoaderFileEntry>> entries(files.size());
	for_each_file([&](uint32_t i){ entries[i] = loaderRegisterFile(files[i]); });

	// The first dropped files give the initial bounding volume of the octree root (the union of their
	// header bounding boxes), before any of their batches is queued: the root is built from it
	// instead of from the first points (only the first proposal counts, see `proposeInitialAABB')
	{
		glm::vec3 mins = glm::vec3(0.0f);
		glm::vec3 maxs = glm::vec3(0.0f);
		if(loaderHeadersBounds(entries, mins, maxs)){
			GpuVersion::proposeInitialAABB(mins, maxs);
		}
	}

	// 2. Overview: the chunk points of every file, sent before anything else
	std::vector<std::vector<Point>> samples(files.size());
	for_each_file([&](uint32_t i){
		if(entries[i]){ samples[i] = loaderReadChunkPoints(*entries[i]); }
	});
	loaderCreateOverviewBatches(entries, samples);

	// 3. The regular batches of every file
	for(const std::optional<LoaderFileEntry>& entry : entries){
		if(entry){ loaderCreateFileBatches(*entry); }
	}
}


void LoaderGpuVersion::createNewBatches(std::string file){
	createNewBatchesFromFiles({file});
}


void LoaderGpuVersion::enqueueBatches(){
	std::lock_guard<std::mutex> lock(batchesToEnqueueMtx);

	while(!batchesToEnqueue.empty()){
		std::shared_ptr<PointBatch> new_batch = batchesToEnqueue.front();

		// Find the index where to put the new batch
		bool found = false;
		for(uint32_t i=0; i<OocSimLodSettings::BATCHES_LIST_SIZE; i++){
			std::lock_guard<std::mutex> lock(batchesQueueMutexes[i]);
			if(batchesQueue[i]){continue;}
			batchesQueue[i] = new_batch;
			batchesToEnqueue.pop_front();
			found = true;
			break;
		}

		if(!found){return;}
	}
}







// From claude
void LoaderGpuVersion::filesRecordUi(){
	ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("File Explorer")) {
        ImGui::End();
        return;
    }

    // Build directory tree from filesRecord
    // Map: directory path -> list of (filename, full path)
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> dirTree;

    {
        std::lock_guard<std::mutex> lock(filesRecordMtx);
        for (const auto& [fullPath, info] : filesRecord) {
            fs::path p(fullPath);
            std::string dir  = p.parent_path().string();
            std::string file = p.filename().string();
            dirTree[dir].emplace_back(file, fullPath);
        }
    }

    // Render tree
    for (auto& [dir, files] : dirTree) {
        // Use the directory path as the tree node label
        bool dirOpen = ImGui::TreeNodeEx(dir.c_str(),
                                         ImGuiTreeNodeFlags_DefaultOpen |
                                         ImGuiTreeNodeFlags_SpanAvailWidth);

        if (dirOpen) {
            for (auto& [filename, fullPath] : files) {
                // Snapshot atomics once (avoid multiple reads diverging)
                laszip_header* header = nullptr;
                uint32_t  nb_batches          = 0;
                uint32_t  nb_batches_loaded    = 0;
                uint32_t  nb_batches_on_device = 0;
                uint32_t  nb_batches_inserted  = 0;

                {
                    std::lock_guard<std::mutex> lock(filesRecordMtx);
                    auto it = filesRecord.find(fullPath);
                    if (it != filesRecord.end()) {
						header  			 = it->second.header.get();
                        nb_batches           = it->second.nb_batches;
                        nb_batches_loaded    = it->second.nb_batches_loaded.load();
                        nb_batches_on_device = it->second.nb_batches_on_device.load();
                        nb_batches_inserted  = it->second.nb_batches_inserted.load();
                    }
                }

                // --- Clickable file label ---
                ImGui::PushID(fullPath.c_str());

                bool clicked = ImGui::Selectable(filename.c_str(), false,
                                                  ImGuiSelectableFlags_None,
                                                  ImVec2(0, 0));
                if (clicked) {
                    printf("Zooming in on: %s\n", fullPath.c_str());
					glm::vec3 mins = {header->min_x, header->min_y, header->min_z};
					glm::vec3 maxs = {header->max_x, header->max_y, header->max_z};
					Runtime::controls->focus(mins, maxs, 1.f);
                }

                // --- Progress bars ---
				ImVec4 bar_color = ImVec4(0.2f, 0.8f, 0.3f, 1.0f);
				float fraction_loaded   = (nb_batches > 0)
                    ? static_cast<float>(nb_batches_loaded) / static_cast<float>(nb_batches)
                    : 0.0f;
                float fraction_device   = (nb_batches > 0)
                    ? static_cast<float>(nb_batches_on_device) / static_cast<float>(nb_batches)
                    : 0.0f;
                float fraction_inserted = (nb_batches > 0)
                    ? static_cast<float>(nb_batches_inserted) / static_cast<float>(nb_batches)
                    : 0.0f;

				// On-device bar
                char overlayLoaded[64];
                snprintf(overlayLoaded, sizeof(overlayLoaded),
                         "Loaded: %u / %u", nb_batches_loaded, nb_batches);
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, bar_color);
                ImGui::ProgressBar(fraction_loaded, ImVec2(-1.0f, 0.0f), overlayLoaded);
                ImGui::PopStyleColor();


                // On-device bar
                char overlayDevice[64];
                snprintf(overlayDevice, sizeof(overlayDevice),
                         "On device: %u / %u", nb_batches_on_device, nb_batches);
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, bar_color);
                ImGui::ProgressBar(fraction_device, ImVec2(-1.0f, 0.0f), overlayDevice);
                ImGui::PopStyleColor();

                // Inserted bar
                char overlayInserted[64];
                snprintf(overlayInserted, sizeof(overlayInserted),
                         "Inserted:  %u / %u", nb_batches_inserted, nb_batches);
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, bar_color);
                ImGui::ProgressBar(fraction_inserted, ImVec2(-1.0f, 0.0f), overlayInserted);
                ImGui::PopStyleColor();


                ImGui::Separator();
                ImGui::PopID();
            }

            ImGui::TreePop();
        }
    }

    ImGui::End();
}