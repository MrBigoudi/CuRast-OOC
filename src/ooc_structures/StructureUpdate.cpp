#include "StructureUpdate.h"

#include "laszip/laszip_api.h"
#include "globals.h"


/// The batches that still need to be read
std::deque<PointBatch> batchesToLoad = {};
/// The batches that are already loaded
std::deque<PointBatch> batchesLoaded = {};

void initLoadPointBatches(string file){
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
	for(uint64_t first_point = 0; first_point < num_points; first_point += MAX_BATCH_SIZE){
		PointBatch new_batch = {};
		new_batch.file = shared_file;
		new_batch.header = shared_header;
		new_batch.first = first_point;
		new_batch.count = std::min(num_points - first_point, MAX_BATCH_SIZE);
		batchesToLoad.push_back(new_batch);
	}
}


void loadPointsInBatches(){
	// TODO: in parallel
	while(!batchesToLoad.empty()){
		PointBatch& batch = batchesToLoad.front();

		laszip_POINTER laszip_reader;
		if(laszip_create(&laszip_reader)){
			println("ERROR: creating laszip reader for '{}'", *batch.file);
			return;
		}
		laszip_BOOL is_compressed = 0;
		if(laszip_open_reader(laszip_reader, (*batch.file).c_str(), &is_compressed)){
			println("ERROR: opening laszip reader for '{}'", *batch.file);
			laszip_destroy(laszip_reader);
			return;
		}
		laszip_point* laz_point;
		if(laszip_get_point_pointer(laszip_reader, &laz_point)){
			println("ERROR: getting laszip point pointer for '{}'", *batch.file);
			laszip_close_reader(laszip_reader);
			laszip_destroy(laszip_reader);
			return;
		}
		if(laszip_seek_point(laszip_reader, batch.first)){
			println("ERROR: seeking laszip point for for '{}'", *batch.file);
			laszip_close_reader(laszip_reader);
			laszip_destroy(laszip_reader);
			return;	
		}

		double scale_x = batch.header->x_scale_factor;
		double scale_y = batch.header->y_scale_factor;
		double scale_z = batch.header->z_scale_factor;
		double offset_x = batch.header->x_offset;
		double offset_y = batch.header->y_offset;
		double offset_z = batch.header->z_offset;

		uint8_t fmt = batch.header->point_data_format;
		bool has_rgb = (fmt == 2 || fmt == 3 || fmt == 5 || fmt == 7 || fmt == 8 || fmt == 10);
		batch.points = std::make_shared<vector<Point>>();

		for (uint64_t i = 0; i < batch.count; i++) {
			if(laszip_read_point(laszip_reader)){
				println("ERROR: reading point {} for '{}'", i+batch.first, *batch.file);
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

			batch.points->push_back(new_point);
		}

		println("done loading a batch of {} points", batch.count);

		laszip_close_reader(laszip_reader);
		batchesLoaded.push_back(batch);
		batchesToLoad.pop_front();
	}
}

void loadBatchesOnGPU(CuRast* editor){
	while(!batchesLoaded.empty()){
		PointBatch& batch = batchesLoaded.front();

		// Upload positions and colors to GPU
		CUdeviceptr cptr_positions, cptr_colors;
		cuMemAlloc(&cptr_positions, batch.count * sizeof(vec3));
		cuMemAlloc(&cptr_colors,    batch.count * sizeof(uint32_t));
		cuMemcpyHtoD(cptr_positions, batch.getPositions().data(), batch.count * sizeof(vec3));
		cuMemcpyHtoD(cptr_colors, batch.getColors().data(),batch.count * sizeof(uint32_t));

		auto node = make_shared<SNCPoints>("pointcloud");
		node->cptr_positions = cptr_positions;
		node->cptr_colors    = cptr_colors;
		node->numPoints      = batch.count;

		editor->scene.world->children.push_back(node);

		batchesLoaded.pop_front();
	}
}

void loadPointcloud(string file, CuRast* editor){
	initLoadPointBatches(file);
	loadPointsInBatches();
	loadBatchesOnGPU(editor);
};