#include "loader.h"

#include "laszip/laszip_api.h"
#include "globals.h"
#include "gpuVersion.h"


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
	if(laszip_open_reader(laszip_reader, file.c_str(), &is_compressed)){
		println("ERROR: opening laszip reader for '{}'", file);
		laszip_destroy(laszip_reader);
		return;
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

		uint32_t free_index = 0;
		// Find the index where to put the new batch
		// If no space is free on the queue, wait until space is found
		while(true){
			bool found = false;
			for(uint32_t i=0; i<OocSimLodSettings::BATCHES_LIST_SIZE; i++){
				std::lock_guard<std::mutex> lock(batches_queue_mutexes[i]);
				if(batches_queue[i]){continue;}
				free_index = i;
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

		std::shared_ptr<PointBatch> new_batch = std::make_shared<PointBatch>();
		new_batch->file = shared_file;
		new_batch->header = shared_header;
		new_batch->first = first_point;
		new_batch->count = std::min(num_points - first_point, uint64_t(OocSimLodSettings::MAX_POINTS_PER_BATCHES));
		new_batch->state = BatchState::ToLoad;
	
		std::lock_guard<std::mutex> lock(batches_queue_mutexes[free_index]);
		batches_queue[free_index] = new_batch;
	}

    timing->stop_clock();

}



void loadPointsInBatches(
    std::deque<std::shared_ptr<PointBatch>>& batches_queue,
    std::deque<std::mutex>& batches_queue_mutexes
){
	std::vector<uint32_t> batches_indices(OocSimLodSettings::MAX_BATCHES_PER_LOAD, 0);
	uint32_t last_index = 0;

	
	for(uint32_t i=0; i<OocSimLodSettings::BATCHES_LIST_SIZE; i++){
		std::lock_guard<std::mutex> lock(batches_queue_mutexes[i]);
		if(batches_queue[i] && batches_queue[i]->state == BatchState::ToLoad){
			batches_indices[last_index] = i;
			last_index++;
			if(last_index >= OocSimLodSettings::MAX_BATCHES_PER_LOAD){break;}
		}
	}
	if(last_index == 0){return;}
	static uint32_t lastLoadAttempt = 0;
	if(OocSimLodSettings::IS_RUNNING_IN_PARALLEL && batches_indices.size() < OocSimLodSettings::MIN_BATCHES_PER_LOAD){
		if(lastLoadAttempt < OocSimLodSettings::MAX_ATTEMPTS_BEFORE_IGNORING_MIN_VARIABLES){
			lastLoadAttempt++;
			return;
		} 
	}
	lastLoadAttempt = 0;

    std::shared_ptr<Timing> timing = Timing::addTiming("load points in batches", true);

	auto lambda = [&](uint32_t index){
		std::lock_guard<std::mutex> lock(batches_queue_mutexes[index]);
		std::shared_ptr<PointBatch> batch = batches_queue[index];

		laszip_POINTER laszip_reader;
		if(laszip_create(&laszip_reader)){
			return;
		}
		laszip_BOOL is_compressed = 0;
		if(laszip_open_reader(laszip_reader, (*batch->file).c_str(), &is_compressed)){
			laszip_destroy(laszip_reader);
			return;
		}
		laszip_point* laz_point;
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
				break;
			}

			Point new_point = {};
			float x = (float)(laz_point->X * scale_x + offset_x);
			float y = (float)(laz_point->Y * scale_y + offset_y);
			float z = (float)(laz_point->Z * scale_z + offset_z);
			new_point.position = {x,y,z};

			if(has_rgb){
				// LAS RGB is 16-bit; many writers use the high byte, some use the low byte
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
		}

		laszip_close_reader(laszip_reader);

		batch->state = BatchState::Loaded;
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

void LoaderGpuVersion::init(){
	batchesQueue = std::deque<std::shared_ptr<PointBatch>>(OocSimLodSettings::BATCHES_LIST_SIZE, nullptr);
    batchesQueueMutexes = std::deque<std::mutex>(OocSimLodSettings::BATCHES_LIST_SIZE);

	batchesOnGpu = std::vector<uint32_t>(OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE, -1);
	batchesOnGpuStatus = std::vector<uint32_t>(OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE, true);
}


void LoaderGpuVersion::createNewBatches(string file){
	initLoadPointBatches(file, batchesQueue, batchesQueueMutexes);
}

void LoaderGpuVersion::fetchFromDevice(CUstream* stream){
	CURuntime::assertCudaSuccess(cuMemcpyDtoHAsync(
		batchesOnGpuStatus.data(), 
		(CUdeviceptr)GpuVersion::hostStaging.batchesAddedMask,
		OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE * sizeof(uint32_t),
		OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? *stream : 0
	));

	cudaStreamSynchronize(*stream);

	for(uint32_t i=0; i<OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE; i++){
		if(batchesOnGpuStatus[i]){
			uint32_t real_index = batchesOnGpu[i];
			batchesOnGpu[i] = -1;
			if(real_index != -1){
				std::lock_guard<std::mutex> lock(batchesQueueMutexes[real_index]);
				batchesQueue[real_index]->state = BatchState::ToRemove;
			}
		}
	}
}

void LoaderGpuVersion::sendToDevice(CUstream* stream){
	uint32_t last_index = 0;
	for(uint32_t i=0; i<OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE; i++){
		// Check if the batch is still being used on device side
		if(batchesOnGpu[i] != -1 && !batchesOnGpuStatus[i]){continue;}

		for(uint32_t j=last_index; j<OocSimLodSettings::BATCHES_LIST_SIZE; j++){
			std::lock_guard<std::mutex> lock(batchesQueueMutexes[j]);
			if(batchesQueue[j] && batchesQueue[j]->state == BatchState::Loaded){
				// Mark the batch as being sent to the device
				last_index = j+1;
				batchesOnGpu[i] = j;
				batchesOnGpuStatus[i] = false;
				batchesQueue[j]->state = BatchState::Sent;

				// Send the batches to device side
				CUdeviceptr dst_points = ((CUdeviceptr*)(GpuVersion::batchesToAddPointsPointers))[i];
				const void* src_points = batchesQueue[j]->points->data();
				size_t     size_points = batchesQueue[j]->count * sizeof(CPoint);

				CUdeviceptr dst_count = (CUdeviceptr)(GpuVersion::hostStaging.batchesToAddCounts) + (CUdeviceptr)(i*sizeof(uint32_t));
				const void* src_count = &batchesQueue[j]->count;
				size_t     size_count = sizeof(uint32_t);

				CUdeviceptr dst_flag = (CUdeviceptr)(GpuVersion::hostStaging.batchesAddedMask) + (CUdeviceptr)(i*sizeof(uint32_t));
				uint32_t    src_flag = false;
				size_t     size_flag = sizeof(uint32_t);

				CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(dst_points, src_points, size_points,
					OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? *stream : 0
				));
				CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(dst_count, src_count, size_count,
					OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? *stream : 0
				));
				CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(dst_flag, &src_flag, size_flag,
					OocSimLodSettings::IS_RUNNING_IN_PARALLEL ? *stream : 0
				));
				
				break;
			}
		}
	}	
}

void LoaderGpuVersion::run(CUstream* stream, CuRast* editor, CUcontext* context){
	// Check if batches are done on GPU side
	fetchFromDevice(stream);

	// Clear completed batches
	clearUnusedBatches(batchesQueue, batchesQueueMutexes);

	// Try loading points from disk
	loadPointsInBatches(batchesQueue, batchesQueueMutexes);

	// Get the batches to send to device side
	sendToDevice(stream);
}