#include "loader.h"

#include "laszip/laszip_api.h"
#include "globals.h"


void initLoadPointBatches(string file){
    std::shared_ptr<Timing> timing = addTiming(format("init load file: {}", file), true);

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

	for(uint64_t first_point = 0; first_point < num_points; first_point += CuRastSettings::maxBatchSize){

		uint32_t free_index = 0;
		// Find the index where to put the new batch
		// If no space is free on the queue, wait until space is found
		while(true){
			bool found = false;
			for(uint32_t i=0; i<BATCHES_QUEUE_SIZE; i++){
				std::lock_guard<std::mutex> lock(batchesQueueMutexes[i]);
				if(batchesQueue[i]){continue;}
				free_index = i;
				found = true;
				break;
			}
			if(found){break;}

			if(!CPU_PARALLELISED){
				uint32_t old_queue_size = BATCHES_QUEUE_SIZE;
				BATCHES_QUEUE_SIZE *= 2;
				batchesQueue.resize(BATCHES_QUEUE_SIZE);
				batchesQueueMutexes.resize(BATCHES_QUEUE_SIZE);
			} else {
				// Wait a bit to give time for the queue to be emptied
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		}

		std::shared_ptr<PointBatch> new_batch = std::make_shared<PointBatch>();
		new_batch->file = shared_file;
		new_batch->header = shared_header;
		new_batch->first = first_point;
		new_batch->count = std::min(num_points - first_point, uint64_t(CuRastSettings::maxBatchSize));
		new_batch->state = BatchState::ToLoad;
	
		std::lock_guard<std::mutex> lock(batchesQueueMutexes[free_index]);
		batchesQueue[free_index] = new_batch;
	}

    timing->stop_clock();

}



void loadPointsInBatches(){
	std::vector<uint32_t> batches_indices(MAX_BATCHES_PER_LOAD, 0);
	uint32_t last_index = 0;

	
	for(uint32_t i=0; i<BATCHES_QUEUE_SIZE; i++){
		std::lock_guard<std::mutex> lock(batchesQueueMutexes[i]);
		if(batchesQueue[i] && batchesQueue[i]->state == BatchState::ToLoad){
			batches_indices[last_index] = i;
			last_index++;
			if(last_index >= MAX_BATCHES_PER_LOAD){break;}
		}
	}
	if(last_index == 0){return;}

    std::shared_ptr<Timing> timing = addTiming("load points in batches", true);

	auto lambda = [&](uint32_t index){
		std::lock_guard<std::mutex> lock(batchesQueueMutexes[index]);
		std::shared_ptr<PointBatch> batch = batchesQueue[index];

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
	if(CPU_PARALLELISED){
		std::for_each(std::execution::par, first, last, lambda);
	} else {
		std::for_each(first, last, lambda);
	}

    timing->stop_clock();
}


void loadBatchesOnGPU(CuRast* editor, CUcontext* ctx){
	std::vector<uint32_t> batches_indices(MAX_BATCHES_PER_GPU_LOAD, 0);
	uint32_t last_index = 0;

	for(uint32_t i=0; i<BATCHES_QUEUE_SIZE; i++){
		std::lock_guard<std::mutex> lock(batchesQueueMutexes[i]);
		if(batchesQueue[i] && batchesQueue[i]->state == BatchState::Inserted){
			batches_indices[last_index] = i;
			last_index++;
			if(last_index >= MAX_BATCHES_PER_GPU_LOAD){break;}
		}
	}
	if(last_index == 0){return;}

    std::shared_ptr<Timing> timing = addTiming("send points to GPU memory", true);

	std::mutex mtx_counter;

	auto lambda = [&](uint32_t index){
		std::lock_guard<std::mutex> lock(batchesQueueMutexes[index]);
		std::shared_ptr<PointBatch> batch = batchesQueue[index];

        // Upload positions and colors to GPU
        CUdeviceptr cptr_positions, cptr_colors;

		if(CPU_PARALLELISED){
			cuCtxSetCurrent(*ctx);
		}

        cuMemAlloc(&cptr_positions, batch->count * sizeof(vec3));
        cuMemAlloc(&cptr_colors,    batch->count * sizeof(uint32_t));
        cuMemcpyHtoD(cptr_positions, batch->getPositions().data(), batch->count * sizeof(vec3));
        cuMemcpyHtoD(cptr_colors, batch->getColors().data(),batch->count * sizeof(uint32_t));

        auto node = make_shared<SNCPoints>("pointcloud");
        node->cptr_positions = cptr_positions;
        node->cptr_colors    = cptr_colors;
        node->numPoints      = batch->count;

		{
			// Unified memory rendering
			auto tmp_positions = std::make_shared<std::vector<vec3>>(batch->getPositions());
			auto tmp_colors = std::make_shared<std::vector<uint32_t>>(batch->getColors());
			unified_positions.push_back(tmp_positions);
			unified_colors.push_back(tmp_colors);
			node->ptr_positions = tmp_positions.get()->data();
			node->ptr_colors    = tmp_colors.get()->data();
		}

		batch->state = BatchState::ToRemove;

		{
			std::lock_guard<std::mutex> lock_counter(mtx_counter);
			NB_POINTS += batch->count;
		}

		std::lock_guard<std::mutex> lock_scene(updateSceneMutex);
        editor->scene.world->children.push_back(node);
    };

	auto first = batches_indices.begin();
	auto last = first + last_index;
	if(CPU_PARALLELISED){
		std::for_each(std::execution::par, first, last, lambda);
	} else {
		std::for_each(first, last, lambda);
	}

    timing->stop_clock();
}


void loadPointcloudRoutine(){
    while(true){
        loadPointsInBatches();
    }
};

void clearUnusedBatches(){
	for(uint32_t i=0; i<BATCHES_QUEUE_SIZE; i++){
		std::lock_guard<std::mutex> lock(batchesQueueMutexes[i]);
		if(batchesQueue[i] && batchesQueue[i]->state == BatchState::ToRemove){
			batchesQueue[i] = nullptr;
		}
	}
}

void clearUnusedBatchesRoutine(){
	while(true){
		clearUnusedBatches();
	}
}