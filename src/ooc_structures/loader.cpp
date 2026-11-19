#include "loader.h"


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


void LoaderGpuVersion::updateCameraSnapshot(){
    const mat4 view = VKRenderer::view.view;
    vec3 pos = vec3(glm::inverse(view) * vec4(0.0f, 0.0f, 0.0f, 1.0f));
    cameraPosX.store(pos.x, std::memory_order_relaxed);
    cameraPosY.store(pos.y, std::memory_order_relaxed);
    cameraPosZ.store(pos.z, std::memory_order_relaxed);

    // Full camera for the screen-space priorities (read by the loading thread)
    CameraState state = {};
    state.view_proj = mat4(VKRenderer::view.proj) * view;
    state.position = pos;
    state.width = std::max(1.0f, static_cast<float>(VKRenderer::width));
    state.height = std::max(1.0f, static_cast<float>(VKRenderer::height));
    std::lock_guard<std::mutex> lock(cameraStateMtx);
    cameraState = state;
}

LoaderGpuVersion::CameraState LoaderGpuVersion::getCameraState(){
    std::lock_guard<std::mutex> lock(cameraStateMtx);
    return cameraState;
}

LoaderGpuVersion::LoadPriority LoaderGpuVersion::loadPriority(const BatchBounds& bounds, const CameraState& camera){
    LoadPriority priority = {};
    if(!bounds.isValid()){ return priority; }   // tier 2: unknown

    // Distance from the camera to the box (0 inside)
    const glm::vec3 closest = glm::clamp(camera.position, bounds.mins, bounds.maxs);
    const float distance = glm::length(camera.position - closest);
    if(distance == 0.0f){
        priority.tier = 0;
        priority.key = -INFINITY;   // the camera is inside: first
        return priority;
    }

    // Projected screen rectangle of the box (same idea as the visibility priority on the device)
    float min_x =  INFINITY, min_y =  INFINITY;
    float max_x = -INFINITY, max_y = -INFINITY;
    bool any_behind = false;
    bool all_behind = true;
    for(uint32_t corner = 0; corner < 8; corner++){
        const glm::vec4 p = glm::vec4(
            (corner & 1) ? bounds.maxs.x : bounds.mins.x,
            (corner & 2) ? bounds.maxs.y : bounds.mins.y,
            (corner & 4) ? bounds.maxs.z : bounds.mins.z,
            1.0f
        );
        const glm::vec4 clip = camera.view_proj * p;
        if(!(clip.w > 1e-6f)){ any_behind = true; continue; }
        all_behind = false;
        const float ndc_x = clip.x / clip.w;
        const float ndc_y = clip.y / clip.w;
        min_x = std::min(min_x, ndc_x); max_x = std::max(max_x, ndc_x);
        min_y = std::min(min_y, ndc_y); max_y = std::max(max_y, ndc_y);
    }

    priority.tier = 1;
    priority.key = distance;
    if(all_behind){ return priority; }
    if(any_behind){
        // Crosses the camera plane: right next to the camera
        priority.tier = 0;
        priority.key = -1.0f / (distance * distance);
        return priority;
    }

    // Visible part of the rectangle
    const float x0 = std::max(min_x, -1.0f), x1 = std::min(max_x, 1.0f);
    const float y0 = std::max(min_y, -1.0f), y1 = std::min(max_y, 1.0f);
    if(!(x0 < x1 && y0 < y1)){ return priority; }   // outside of the view

    // Screen-space size of the data at the box's nearest point (~ 1 / distance^2), not the projected
    // area of the whole box: before loading, a box is only the hull of the few chunk points of the
    // batch, so its area is a noisy estimate, while its nearest point is well estimated; and a batch
    // spread over a large area must not win over a closer one just for being large
    const float size = 1.0f / (distance * distance);

    // Centre weighting on the clipped rectangle's centre (as `getVisibilityPriority' on the device)
    const glm::vec2 centre = glm::vec2(
        0.5f * (x0 + x1) * 0.5f * camera.width,
        0.5f * (y0 + y1) * 0.5f * camera.height
    );
    const float r_norm = std::clamp(glm::length(centre) / (0.5f * glm::length(glm::vec2(camera.width, camera.height))), 0.0f, 1.0f);
    const float score = size * (1.0f - LOAD_CENTER_WEIGHT * r_norm);

    priority.tier = 0;
    priority.key = std::isfinite(score) ? -score : 0.0f;
    return priority;
}

vec3 LoaderGpuVersion::getCameraSnapshot(){
    return vec3(
        cameraPosX.load(std::memory_order_relaxed),
        cameraPosY.load(std::memory_order_relaxed),
        cameraPosZ.load(std::memory_order_relaxed)
    );
}



void LoaderGpuVersion::init(){
    CURuntime::assertCudaSuccess(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CURuntime::assertCudaSuccess(cuEventCreate(&eventLoadComplete, CU_EVENT_DISABLE_TIMING));
    initPreview();

	batchesQueue = std::deque<std::shared_ptr<PointBatch>>(OocSimLodSettings::BATCHES_LIST_SIZE, nullptr);
    batchesQueueMutexes = std::deque<std::mutex>(OocSimLodSettings::BATCHES_LIST_SIZE);
    slotsBounds = std::vector<BatchBounds>(OocSimLodSettings::BATCHES_LIST_SIZE);

	batchesOnGpu = std::vector<uint32_t>(OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE, -1);
	batchesOnGpuStatus = GpuVersion::allocHost<uint32_t>(OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE);
	for(uint32_t i = 0; i < OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE; i++){
		((uint32_t*)(batchesOnGpuStatus))[i] = BatchHandled;
	}
}

void LoaderGpuVersion::destroy(){
    // The loading workers first (they use the queues)
    stopLoading();
    destroyPreview();

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

	// Same priority as the loading (in view by screen-space size, then out of view by distance), with the
	// exact bounds of the loaded batches (written before they were published in `loadedBatches')
	const CameraState camera = getCameraState();
	std::vector<std::pair<LoadPriority, uint32_t>> ranked(ready.size());
	for(uint32_t k = 0; k < ready.size(); k++){
		ranked[k] = {loadPriority(slotsBounds[ready[k].first], camera), k};
	}
	std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b){
		if(a.first < b.first){ return true; }
		if(b.first < a.first){ return false; }
		return a.second < b.second;
	});
	std::vector<LoadPriority> ready_priorities(ready.size());
	{
		std::vector<std::pair<uint32_t, std::shared_ptr<PointBatch>>> sorted(ready.size());
		for(uint32_t k = 0; k < ranked.size(); k++){
			sorted[k] = std::move(ready[ranked[k].second]);
			ready_priorities[k] = ranked[k].first;
		}
		ready = std::move(sorted);
	}

	// Free device batch slots (freed by `fetchFromDevice' once the device has inserted their batch)
	std::vector<uint32_t> free_device_slots = {};
	for(uint32_t i = 0; i < OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE; i++){
		if(batchesOnGpu[i] == -1){ free_device_slots.push_back(i); }
	}

	// While a batch in view is still pending or being loaded, the out-of-view batches are held back:
	// once sent, a batch belongs to the device update until it completes, so sending them would spend
	// a whole update on data the camera left. They stay loaded (the workers can still demote them
	// for better batches: a batch in view is always clearly better) and are sent once nothing in
	// view is left to load
	bool hold_back_out_of_view = false;
	if(!free_device_slots.empty()){
		const bool has_out_of_view = std::any_of(ready_priorities.begin(), ready_priorities.end(),
			[](const LoadPriority& priority){ return priority.tier != 0; }
		);
		if(has_out_of_view){ hold_back_out_of_view = hasInViewWork(camera); }
	}

	uint32_t nb_used_device_slots = 0;
	auto send = [&](uint32_t slot, const std::shared_ptr<PointBatch>& batch){
		const uint32_t i = free_device_slots[nb_used_device_slots++];
        batchesOnGpu[i] = slot;
        {
            // Under the slot's mutex, like every other access to the state of a queued batch
            std::lock_guard<std::mutex> lock(batchesQueueMutexes[slot]);
            batch->state = BatchState::Sent;
        }
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
	};

	// The overview batches go before everything else (see `createNewBatchesFromFiles'), then the
	// loaded batches by priority. What is not sent goes back (the overview ones in their own list,
	// so that they stay first)
	std::vector<std::pair<uint32_t, std::shared_ptr<PointBatch>>> kept_overview = {};
	std::vector<std::pair<uint32_t, std::shared_ptr<PointBatch>>> kept_ready = {};
	for(auto& entry : overview){
		if(nb_used_device_slots < free_device_slots.size()){ send(entry.first, entry.second); }
		else { kept_overview.push_back(std::move(entry)); }
	}
	for(uint32_t k = 0; k < ready.size(); k++){
		// A previewed batch is never held back: its preview is only released once it is sent
		const bool is_held_back = hold_back_out_of_view && ready_priorities[k].tier != 0
			&& !ready[k].second->is_previewed;
		if(!is_held_back && nb_used_device_slots < free_device_slots.size()){ send(ready[k].first, ready[k].second); }
		else { kept_ready.push_back(std::move(ready[k])); }
	}

    if(!kept_overview.empty() || !kept_ready.empty()){
        std::lock_guard<std::mutex> lock(loadedBatchesMtx);
        loadedOverviewBatches.insert(loadedOverviewBatches.begin(), kept_overview.begin(), kept_overview.end());
        loadedBatches.insert(loadedBatches.begin(), kept_ready.begin(), kept_ready.end());
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

void LoaderGpuVersion::clearUnusedBatches(){
	for(uint32_t i=0; i<OocSimLodSettings::BATCHES_LIST_SIZE; i++){
		std::lock_guard<std::mutex> lock(batchesQueueMutexes[i]);
		if(batchesQueue[i] && batchesQueue[i]->state == BatchState::ToRemove){
			batchesQueue[i] = nullptr;
		}
	}
}

void LoaderGpuVersion::initPreview(){
    const uint32_t nb_regions = std::min(PREVIEW_NB_REGIONS, LOADER_PREVIEW_MAX_REGIONS);
    previewPointsPerRegion = PREVIEW_POINTS_PER_REGION;
    if(!PREVIEW_ENABLED || nb_regions == 0 || previewPointsPerRegion == 0){ return; }

    const uint64_t nb_points = uint64_t(nb_regions) * previewPointsPerRegion;
    CUdeviceptr device_points = 0;
    CURuntime::assertCudaSuccess(cuMemAlloc(&device_points, nb_points * sizeof(CPoint)));
    previewDevicePoints = reinterpret_cast<CPoint*>(device_points);
    void* pinned_points = nullptr;
    CURuntime::assertCudaSuccess(cuMemAllocHost(&pinned_points, nb_points * sizeof(CPoint)));
    previewPinnedPoints = reinterpret_cast<CPoint*>(pinned_points);

    CURuntime::assertCudaSuccess(cuStreamCreate(&previewStream, CU_STREAM_NON_BLOCKING));
    CURuntime::assertCudaSuccess(cuEventCreate(&previewDrawnEvent, CU_EVENT_DISABLE_TIMING));
    previewRegions = std::vector<LoaderPreviewRegion>(nb_regions);
    for(LoaderPreviewRegion& region : previewRegions){
        CURuntime::assertCudaSuccess(cuEventCreate(&region.upload_event, CU_EVENT_DISABLE_TIMING));
    }
}

void LoaderGpuVersion::destroyPreview(){
    if(!previewDevicePoints){ return; }
    CURuntime::assertCudaSuccess(cuStreamSynchronize(previewStream));
    for(LoaderPreviewRegion& region : previewRegions){
        CURuntime::assertCudaSuccess(cuEventDestroy(region.upload_event));
    }
    previewRegions.clear();
    CURuntime::assertCudaSuccess(cuEventDestroy(previewDrawnEvent));
    CURuntime::assertCudaSuccess(cuStreamDestroy(previewStream));
    CURuntime::assertCudaSuccess(cuMemFreeHost(previewPinnedPoints));
    CURuntime::assertCudaSuccess(cuMemFree(reinterpret_cast<CUdeviceptr>(previewDevicePoints)));
    previewPinnedPoints = nullptr;
    previewDevicePoints = nullptr;
}

void LoaderGpuVersion::updatePreview(){
    if(!previewDevicePoints){ return; }

    // 1. Completed uploads become visible (if their batch is still previewed)
    for(LoaderPreviewRegion& region : previewRegions){
        if(!region.is_uploading){continue;}
        const CUresult res = cuEventQuery(region.upload_event);
        if(res == CUDA_ERROR_NOT_READY){continue;}
        CURuntime::assertCudaSuccess(res);
        region.is_uploading = false;
        region.is_visible = (region.batch != nullptr);
    }

    // What has been drawn once is never lost: a region keeps its batch, whatever the camera does,
    // until that batch is sent to the device. From the next frame on, `kernel_draw_pending_batches'
    // draws it from its device slot, then the octree once it is inserted.
    // For that, a previewed batch is never demoted back to the pending batches (its points would be
    // dropped while the preview still shows them, see `loaderDemoteWorstLoaded') nor held back at send
    // time (`sendToDevice'): it always goes on to the device, and its region is always freed.
    struct Upload {
        uint32_t region = 0;
        std::shared_ptr<PointBatch> batch = nullptr;
        std::shared_ptr<vector<CPoint>> points = nullptr;   // kept alive while it is copied
    };
    std::vector<Upload> uploads = {};
    {
        // Under `loadedBatchesMtx': the batches leave the loaded lists only under it (sent by the main
        // thread, demoted by the workers), `is_previewed' is set under it too, so a batch is either
        // demoted before it is previewed, or previewed and never demoted.
        // `batch->points' of a published batch is only written under it; so is `slotsBounds' read here
        std::lock_guard<std::mutex> lock(loadedBatchesMtx);

        std::unordered_set<const PointBatch*> loaded = {};
        for(const auto& [slot, batch] : loadedOverviewBatches){ loaded.insert(batch.get()); }
        for(const auto& [slot, batch] : loadedBatches){ loaded.insert(batch.get()); }

        // 2. Regions of the batches that left the loaded lists: sent (a previewed batch is never
        //    demoted). `run' calls this before `sendToDevice', after `fetchFromDevice' synchronised the
        //    copies of the loading stream: their device slot is complete and drawn from this frame on
        for(LoaderPreviewRegion& region : previewRegions){
            if(!region.batch || loaded.contains(region.batch.get())){continue;}
            region.batch->is_previewed = false;
            region.batch.reset();
            region.is_visible = false;
            region.nb_points = 0;
        }

        // 3. New previews: the best loaded batches in view without a region, as many as free regions
        //    (a region released while its copy is in flight is free once the copy is complete)
        uint32_t nb_free = 0;
        for(const LoaderPreviewRegion& region : previewRegions){
            if(!region.batch && !region.is_uploading){ nb_free++; }
        }
        if(PREVIEW_ENABLED && nb_free > 0){
            struct Candidate {
                LoadPriority priority = {};
                std::shared_ptr<PointBatch> batch = nullptr;
            };
            std::vector<Candidate> candidates = {};
            const CameraState camera = getCameraState();
            auto add = [&](const std::vector<std::pair<uint32_t, std::shared_ptr<PointBatch>>>& list){
                for(const auto& [slot, batch] : list){
                    if(!batch || batch->is_previewed || !batch->points || batch->points->empty()){continue;}
                    const LoadPriority priority = loadPriority(slotsBounds[slot], camera);
                    if(priority.tier != 0){continue;}
                    candidates.push_back({priority, batch});
                }
            };
            add(loadedOverviewBatches);
            add(loadedBatches);
            std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b){
                return a.priority < b.priority;
            });

            uint32_t next_region = 0;
            for(const Candidate& candidate : candidates){
                while(next_region < previewRegions.size()
                    && (previewRegions[next_region].batch || previewRegions[next_region].is_uploading)
                ){ next_region++; }
                if(next_region == previewRegions.size()){break;}

                // Previewed from now on: not demoted anymore, not held back at send time
                candidate.batch->is_previewed = true;
                LoaderPreviewRegion& region = previewRegions[next_region];
                region.batch = candidate.batch;
                region.nb_points = 0;
                region.is_visible = false;
                uploads.push_back({next_region, candidate.batch, candidate.batch->points});
                next_region++;
            }
        }
    }

    // 4. Copies: an even subsample of the batch into its region; the region is drawn once the copy
    //    is complete (step 1 of a next frame). If the batch is sent meanwhile, its region is released
    //    before it was ever drawn: nothing disappears
    for(const Upload& upload : uploads){
        LoaderPreviewRegion& region = previewRegions[upload.region];
        const std::vector<CPoint>& source = *upload.points;
        const uint64_t nb_source = source.size();
        const uint32_t nb_points = static_cast<uint32_t>(std::min<uint64_t>(nb_source, previewPointsPerRegion));

        CPoint* pinned = previewPinnedPoints + uint64_t(upload.region) * previewPointsPerRegion;
        for(uint32_t j = 0; j < nb_points; j++){
            pinned[j] = source[(uint64_t(j) * nb_source) / nb_points];
        }

        // The draws launched before may still read this region (drawn for another batch by the frames
        // in flight): the copy waits for them
        CURuntime::assertCudaSuccess(cuStreamWaitEvent(previewStream, previewDrawnEvent, 0));
        const CUdeviceptr destination = reinterpret_cast<CUdeviceptr>(previewDevicePoints + uint64_t(upload.region) * previewPointsPerRegion);
        CURuntime::assertCudaSuccess(cuMemcpyHtoDAsync(destination, pinned, uint64_t(nb_points) * sizeof(CPoint), previewStream));
        CURuntime::assertCudaSuccess(cuEventRecord(region.upload_event, previewStream));
        region.nb_points = nb_points;
        region.is_uploading = true;
    }
}

bool LoaderGpuVersion::previewDrawCounts(LoaderPreviewCounts& counts){
    counts = {};
    bool has_visible = false;
    for(uint32_t r = 0; r < previewRegions.size() && r < LOADER_PREVIEW_MAX_REGIONS; r++){
        const LoaderPreviewRegion& region = previewRegions[r];
        if(!region.is_visible){continue;}
        counts.counts[r] = region.nb_points;
        has_visible = has_visible || (region.nb_points > 0);
    }
    return has_visible;
}

void LoaderGpuVersion::previewRecordDrawn(){
    if(!previewDevicePoints){ return; }
    // Same stream as the draws (the default stream)
    CURuntime::assertCudaSuccess(cuEventRecord(previewDrawnEvent, 0));
}

namespace {
/// True on a thread while it runs `loadingRoutine' (`stopLoading' must not wait for itself)
thread_local bool loaderIsInsideLoadingRoutine = false;
}

void LoaderGpuVersion::loadingRoutine(){
	// Once the loading is stopped (program exit), the loader's static data may be being destroyed:
	// do nothing (main.cpp's loading thread may still be looping when the program exits on an error)
	if(stopLoadingWorkers){
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		return;
	}
	nbLoadingRoutinesRunning++;
	loaderIsInsideLoadingRoutine = true;
	if(!stopLoadingWorkers){
		// Clear completed batches
		clearUnusedBatches();
		// Load the highest priority batches from disk
		loadPendingBatches();
	}
	loaderIsInsideLoadingRoutine = false;
	nbLoadingRoutinesRunning--;
}

bool LoaderGpuVersion::run(CuRast* editor, CUcontext* context){
	updateCameraSnapshot();

	// Check if batches are done on GPU side
	fetchFromDevice();

	// Preview of the loaded batches in view (before sending: see `updatePreview')
	updatePreview();

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
/// Their bounds are estimated from the overview sample: `samples[k]' is point `k * stride' of the
/// file, so the samples in [first, first + count) outline the batch (else the header bounds)
void loaderCreateFileBatches(const LoaderFileEntry& entry, const std::vector<CPoint>& samples, uint64_t stride){
	LoaderGpuVersion::BatchBounds header_bounds = {};
	header_bounds.extend(glm::vec3(float(entry.header->min_x), float(entry.header->min_y), float(entry.header->min_z)));
	header_bounds.extend(glm::vec3(float(entry.header->max_x), float(entry.header->max_y), float(entry.header->max_z)));

	std::lock_guard<std::mutex> lock(LoaderGpuVersion::pendingBatchesMtx);
	for(uint64_t first_point = 0; first_point < entry.num_points; first_point += OocSimLodSettings::MAX_POINTS_PER_BATCHES){

		std::shared_ptr<PointBatch> new_batch = std::make_shared<PointBatch>();
		new_batch->file = entry.file;
		new_batch->header = entry.header;
		new_batch->first = first_point;
		new_batch->count = std::min(entry.num_points - first_point, uint64_t(OocSimLodSettings::MAX_POINTS_PER_BATCHES));
		new_batch->state = BatchState::ToLoad;

		LoaderGpuVersion::PendingBatch pending = {};
		pending.batch = new_batch;
		if(stride > 0){
			const uint64_t last_point = first_point + new_batch->count;   // excluded
			for(uint64_t k = (first_point + stride - 1) / stride; k < samples.size() && k * stride < last_point; k++){
				pending.bounds.extend(samples[k].position);
			}
		}
		if(!pending.bounds.isValid()){ pending.bounds = header_bounds; }
		LoaderGpuVersion::pendingBatches.push_back(std::move(pending));
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

/// LAS / LAZ point to `CPoint': the header fields are resolved once (per file or batch), not per point,
/// and the points are written in place into their destination (no temporary, no conversion)
struct LoaderPointConverter {
	double scale_x = 1.0, scale_y = 1.0, scale_z = 1.0;
	double offset_x = 0.0, offset_y = 0.0, offset_z = 0.0;
	bool has_rgb = false;

	explicit LoaderPointConverter(const laszip_header& header)
		: scale_x(header.x_scale_factor), scale_y(header.y_scale_factor), scale_z(header.z_scale_factor)
		, offset_x(header.x_offset), offset_y(header.y_offset), offset_z(header.z_offset)
	{
		const uint8_t fmt = header.point_data_format;
		has_rgb = (fmt == 2 || fmt == 3 || fmt == 5 || fmt == 7 || fmt == 8 || fmt == 10);
	}

	/// Color bytes in memory: R, G, B, then A = 0 (the alpha byte is left at 0, as before)
	void operator()(const laszip_point* laz_point, CPoint& out) const {
		out.position = glm::vec3(
			static_cast<float>(laz_point->X * scale_x + offset_x),
			static_cast<float>(laz_point->Y * scale_y + offset_y),
			static_cast<float>(laz_point->Z * scale_z + offset_z)
		);

		uint32_t r, g, b;
		if(has_rgb){
			// 16-bit channels (or 8-bit values stored as is)
			auto to8 = [](uint16_t c) -> uint32_t { return c > 255 ? uint32_t(c >> 8) : uint32_t(c); };
			r = to8(laz_point->rgb[0]);
			g = to8(laz_point->rgb[1]);
			b = to8(laz_point->rgb[2]);
		} else {
			r = g = b = uint32_t(laz_point->intensity >> 8);
		}
		out.color = r | (g << 8) | (b << 16);
	}
};

/// The overview sample of a file: the first point of every LAZ chunk (stored uncompressed, so
/// seeking to it and reading it decompresses nothing else), or every `OVERVIEW_FALLBACK_STRIDE'-th
/// point for uncompressed LAS files and LAZ files with variable-size chunks
std::vector<CPoint> loaderReadChunkPoints(const LoaderFileEntry& entry, uint64_t& stride_out){
	stride_out = 0;
	std::vector<CPoint> points = {};
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
	stride_out = stride;

	const LoaderPointConverter convert(*entry.header);
	points.reserve((entry.num_points + stride - 1) / stride);
	for(uint64_t index = 0; index < entry.num_points; index += stride){
		// The first decoding reads the chunk table and may log (LASzip logging is global and not
		// thread-safe): serialised like the opening
		std::unique_lock<std::mutex> first_lock(LoaderGpuVersion::laszipReaderMtx, std::defer_lock);
		if(index == 0){ first_lock.lock(); }
		if(laszip_seek_point(laszip_reader, static_cast<laszip_I64>(index))){break;}
		if(laszip_read_point(laszip_reader)){break;}
		convert(laz_point, points.emplace_back());
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

/// Bounds of the first overview ("Overview 0"), for the "zoom to Overview 0" entry of the file
/// explorer: the exact bounds of its chunk points (the header bounds may be missing or inaccurate)
/// Function-local static: built on first use, whatever the initialisation order of this file
struct LoaderFirstOverview {
	std::mutex mtx;
	bool is_valid = false;
	glm::vec3 mins = glm::vec3(0.0f);
	glm::vec3 maxs = glm::vec3(0.0f);
};
LoaderFirstOverview& loaderFirstOverview(){
	static LoaderFirstOverview first_overview;
	return first_overview;
}

/// Mix the overview samples of all the files into batches that are sent before any other batch
void loaderCreateOverviewBatches(const std::vector<std::optional<LoaderFileEntry>>& entries,
	const std::vector<std::vector<CPoint>>& samples
){
	uint64_t nb_samples = 0;
	for(const std::vector<CPoint>& file_samples : samples){ nb_samples += file_samples.size(); }
	if(nb_samples == 0){return;}

	// All the samples, shuffled so that every overview batch spans every file
	std::vector<CPoint> all_samples = {};
	all_samples.reserve(nb_samples);
	for(const std::vector<CPoint>& file_samples : samples){
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
	// Numbered from 0: "Overview 0" is the overview of the first dropped files
	static std::atomic<uint32_t> nb_overviews{0};
	const uint32_t overview_index = nb_overviews.fetch_add(1);
	const std::string name = format("[Overview #{}]/{} chunk points of {} files",
		overview_index, nb_samples, nb_files
	);

	// The first overview's bounds, for the "Overview 0" entry of the file explorer
	if(overview_index == 0){
		LoaderGpuVersion::BatchBounds bounds = {};
		for(const CPoint& point : all_samples){ bounds.extend(point.position); }
		if(bounds.isValid()){
			LoaderFirstOverview& first_overview = loaderFirstOverview();
			std::lock_guard<std::mutex> lock(first_overview.mtx);
			first_overview.mins = bounds.mins;
			first_overview.maxs = bounds.maxs;
			first_overview.is_valid = true;
		}
	}
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
		new_batch->points = std::make_shared<vector<CPoint>>(all_samples.begin() + begin, all_samples.begin() + end);

		// A slot in the batches queue (the device side refers to batches by slot)
		while(true){
			bool found = false;
			for(uint32_t i = 0; i < OocSimLodSettings::BATCHES_LIST_SIZE; i++){
				std::lock_guard<std::mutex> lock(LoaderGpuVersion::batchesQueueMutexes[i]);
				if(LoaderGpuVersion::batchesQueue[i]){continue;}
				LoaderGpuVersion::batchesQueue[i] = new_batch;
				LoaderGpuVersion::slotsBounds[i] = {};
				for(const CPoint& point : *new_batch->points){ LoaderGpuVersion::slotsBounds[i].extend(point.position); }
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


uint32_t LoaderGpuVersion::nbLoadingThreads(){
	const uint32_t nb_cores = std::max(1u, std::thread::hardware_concurrency());
	uint32_t nb_threads = OocSimLodSettings::MAX_LOADING_THREADS;
	if(nb_threads == 0){ nb_threads = nb_cores / 4; }   // automatic: quarter of the cores
	nb_threads = std::min(nb_threads, nb_cores);
	nb_threads = std::min(nb_threads, std::max(1u, OocSimLodSettings::MAX_BATCHES_PER_LOAD));
	return std::clamp(nb_threads, 1u, 4096u);
}

namespace {

/// Bounds the LASzip decoding of the loading workers: at most `nbLoadingThreads()' cores decode at
/// once (the overview of newly dropped files is read with every core instead, see
/// `createNewBatchesFromFiles': one seek and one point per chunk, a short burst)
std::counting_semaphore<4096>& loaderDecodingSlots(){
	static std::counting_semaphore<4096> slots(static_cast<std::ptrdiff_t>(LoaderGpuVersion::nbLoadingThreads()));
	return slots;
}

/// Holds one decoding slot for its lifetime. Taken before any per-file mutex, everywhere
struct LoaderDecodingSlot {
	LoaderDecodingSlot(){ loaderDecodingSlots().acquire(); }
	~LoaderDecodingSlot(){ loaderDecodingSlots().release(); }
	LoaderDecodingSlot(const LoaderDecodingSlot&) = delete;
	LoaderDecodingSlot& operator=(const LoaderDecodingSlot&) = delete;
};

} // namespace


void LoaderGpuVersion::createNewBatchesFromFiles(std::vector<std::string> files){
	std::vector<uint32_t> indices(files.size());
	std::iota(indices.begin(), indices.end(), 0u);
	// The overview (headers + one point per LAZ chunk of every file) is created with the full
	// parallelism: every core. Only this step: the regular batches are then loaded by the
	// `nbLoadingThreads()' loading workers, as before
	const uint32_t nb_overview_threads = std::max(1u, std::thread::hardware_concurrency());
	auto for_each_file = [&](auto func){
		const uint32_t nb_threads = std::max(1u, std::min<uint32_t>(nb_overview_threads, static_cast<uint32_t>(indices.size())));
		std::atomic<uint32_t> next{0};
		auto work = [&](){
			for(uint32_t k = next.fetch_add(1); k < indices.size(); k = next.fetch_add(1)){
				func(indices[k]);
			}
		};
		std::vector<std::thread> pool;
		pool.reserve(nb_threads);
		for(uint32_t t = 1; t < nb_threads; t++){ pool.emplace_back(work); }
		work();   // the calling thread takes part
		for(std::thread& t : pool){ t.join(); }
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
	std::vector<std::vector<CPoint>> samples(files.size());
	std::vector<uint64_t> strides(files.size(), 0);
	// Full parallelism: no decoding slot here (the slots only limit the loading workers)
	for_each_file([&](uint32_t i){
		if(!entries[i]){ return; }
		samples[i] = loaderReadChunkPoints(*entries[i], strides[i]);
	});
	loaderCreateOverviewBatches(entries, samples);

	// 3. The regular batches of every file
	for(uint32_t i = 0; i < entries.size(); i++){
		if(entries[i]){ loaderCreateFileBatches(*entries[i], samples[i], strides[i]); }
	}
}


void LoaderGpuVersion::createNewBatches(std::string file){
	createNewBatchesFromFiles({file});
}


bool LoaderGpuVersion::isClearlyBetter(const LoadPriority& a, const LoadPriority& b){
    if(a.tier != b.tier){ return a.tier < b.tier; }
    if(a.tier == 0){ return a.key < PREEMPTION_FACTOR * b.key; }          // keys are -scores: a's score > factor x b's
    if(a.tier == 1){ return a.key * PREEMPTION_FACTOR < b.key; }          // keys are distances
    return false;
}

int64_t LoaderGpuVersion::selectBestPending(const CameraState& camera, LoadPriority& best_priority){
    int64_t best = -1;
    best_priority = {};
    for(uint32_t i = 0; i < pendingBatches.size(); i++){
        if(MAX_READERS_PER_FILE > 0){
            auto busy = filesBeingRead.find(*pendingBatches[i].batch->file);
            if(busy != filesBeingRead.end() && busy->second >= MAX_READERS_PER_FILE){continue;}
        }
        const LoadPriority priority = loadPriority(pendingBatches[i].bounds, camera);
        if(best < 0 || priority < best_priority){ best = i; best_priority = priority; }
    }
    return best;
}

LoaderGpuVersion::PendingSummary LoaderGpuVersion::getPendingSummary(){
    const auto now = std::chrono::steady_clock::now();
    const auto max_age = std::chrono::duration<double>(PENDING_SUMMARY_REFRESH_SECONDS);
    {
        std::lock_guard<std::mutex> lock(pendingSummaryMtx);
        if(pendingSummaryIsValid && (now - pendingSummaryTime) < max_age){ return pendingSummary; }
    }

    // Scanned and stored under `pendingBatchesMtx' (lock order: pending -> summary), so that a pick
    // (which invalidates the summary under the same lock) can't be overwritten by an older scan
    const CameraState camera = getCameraState();
    std::lock_guard<std::mutex> lock(pendingBatchesMtx);
    PendingSummary summary = {};
    LoadPriority best = {};
    summary.has_any = selectBestPending(camera, best) >= 0;
    summary.best = best;
    std::lock_guard<std::mutex> lock_summary(pendingSummaryMtx);
    pendingSummary = summary;
    pendingSummaryTime = now;
    pendingSummaryIsValid = true;
    return summary;
}

bool LoaderGpuVersion::hasInViewWork(const CameraState& camera){
    const PendingSummary summary = getPendingSummary();
    if(summary.has_any && summary.best.tier == 0){ return true; }
    std::lock_guard<std::mutex> lock(pendingBatchesMtx);
    for(const auto& [batch, bounds] : loadsInFlightBounds){
        if(loadPriority(bounds, camera).tier == 0){ return true; }
    }
    return false;
}

namespace {

/// Demote the worst loaded (not yet sent) batch if `candidate' is clearly better than it: back to
/// the pending batches (in `demoted', with its exact bounds), its memory and queue slot freed
/// The previewed batches are never demoted (see `updatePreview')
/// `pendingBatchesMtx' and `loadedBatchesMtx' must be held (lock order: pending -> loaded -> slot)
bool loaderDemoteWorstLoaded(
    const LoaderGpuVersion::LoadPriority& candidate,
    const LoaderGpuVersion::CameraState& camera,
    std::vector<LoaderGpuVersion::PendingBatch>& demoted
){
    int64_t worst = -1;
    LoaderGpuVersion::LoadPriority worst_priority = {};
    for(uint32_t k = 0; k < LoaderGpuVersion::loadedBatches.size(); k++){
        // A previewed batch is drawn: it must go on to the device (see `updatePreview')
        if(LoaderGpuVersion::loadedBatches[k].second->is_previewed){continue;}
        const LoaderGpuVersion::LoadPriority priority = LoaderGpuVersion::loadPriority(
            LoaderGpuVersion::slotsBounds[LoaderGpuVersion::loadedBatches[k].first], camera
        );
        if(worst < 0 || worst_priority < priority){ worst = k; worst_priority = priority; }
    }
    if(worst < 0 || !LoaderGpuVersion::isClearlyBetter(candidate, worst_priority)){ return false; }

    const auto [slot, batch] = LoaderGpuVersion::loadedBatches[worst];   // copy: erased below
    LoaderGpuVersion::loadedBatches.erase(LoaderGpuVersion::loadedBatches.begin() + worst);
    LoaderGpuVersion::PendingBatch back = {};
    back.batch = batch;
    back.bounds = LoaderGpuVersion::slotsBounds[slot];
    {
        std::lock_guard<std::mutex> lock_slot(LoaderGpuVersion::batchesQueueMutexes[slot]);
        batch->points.reset();
        batch->state = BatchState::ToLoad;
        LoaderGpuVersion::batchesQueue[slot] = nullptr;
    }
    demoted.push_back(std::move(back));
    return true;
}

} // namespace

bool LoaderGpuVersion::loadOneBatch(){
    const CameraState camera = getCameraState();
    const uint32_t max_ahead = std::max<uint32_t>(1u,
        LOAD_AHEAD_FACTOR * OocSimLodSettings::MAX_BATCHES_PER_OCTREE_UPDATE + nbLoadingWorkers
    );

    // Demoted batches are back in `pendingBatches' (under its lock); their file records after
    std::vector<PendingBatch> demoted = {};
    auto record_demoted = [&](){
        for(const PendingBatch& back : demoted){
            std::lock_guard<std::mutex> lock(filesRecordMtx);
            filesRecord[*back.batch->file].nb_batches_loaded.fetch_sub(1);
        }
        demoted.clear();
    };

    // 1. Pick a batch (lock order everywhere: pending -> loaded -> slot; pending -> summary)
    PendingBatch picked = {};
    LoadPriority picked_priority = {};
    {
        std::lock_guard<std::mutex> lock(pendingBatchesMtx);
        if(pendingBatches.empty()){ return false; }

        // Best batch overall (several workers may read the same file, up to `MAX_READERS_PER_FILE')
        const int64_t chosen = selectBestPending(camera, picked_priority);
        if(chosen < 0){ return false; }

        // Throttle: at most `max_ahead' batches loaded (not yet sent) or being loaded. When full, a
        // clearly better batch replaces the worst loaded one (stale after a camera move, or held back
        // by `sendToDevice' because it is out of view), which goes back to the pending batches
        {
            std::lock_guard<std::mutex> lock_loaded(loadedBatchesMtx);
            const uint32_t nb_ahead = static_cast<uint32_t>(loadedBatches.size() + loadedOverviewBatches.size()) + nbLoadsInFlight.load();
            if(nb_ahead >= max_ahead && !loaderDemoteWorstLoaded(picked_priority, camera, demoted)){
                return false;
            }
        }

        picked = std::move(pendingBatches[chosen]);
        pendingBatches[chosen] = std::move(pendingBatches.back());
        pendingBatches.pop_back();
        pendingBatches.insert(pendingBatches.end(), demoted.begin(), demoted.end());
        filesBeingRead[*picked.batch->file]++;
        loadsInFlightBounds[picked.batch.get()] = picked.bounds;
        nbLoadsInFlight++;

        // The summary may name the batch just picked: scanned again on its next use
        std::lock_guard<std::mutex> lock_summary(pendingSummaryMtx);
        pendingSummaryIsValid = false;
    }
    record_demoted();

    // Undo the bookkeeping of step 1, and put the batch back into the pending batches if asked
    // (in the same critical section: another worker may pick it right away)
    auto finish = [&](bool back_to_pending){
        std::lock_guard<std::mutex> lock(pendingBatchesMtx);
        filesBeingRead[*picked.batch->file]--;
        loadsInFlightBounds.erase(picked.batch.get());
        nbLoadsInFlight--;
        if(back_to_pending){ pendingBatches.push_back(picked); }
    };

    // 2. A queue slot (the device side refers to batches by slot)
    auto take_slot = [&]() -> uint32_t {
        for(uint32_t i = 0; i < OocSimLodSettings::BATCHES_LIST_SIZE; i++){
            std::lock_guard<std::mutex> lock(batchesQueueMutexes[i]);
            if(batchesQueue[i]){continue;}
            picked.batch->state = BatchState::Loading;
            batchesQueue[i] = picked.batch;
            slotsBounds[i] = picked.bounds;
            return i;
        }
        return OocSimLodSettings::BATCHES_LIST_SIZE;
    };
    uint32_t slot = take_slot();
    if(slot == OocSimLodSettings::BATCHES_LIST_SIZE){
        // Every slot is taken: free the slot of the worst loaded batch if this one is clearly better
        // (the batches held back by `sendToDevice' must never block a batch in view)
        bool has_demoted = false;
        {
            std::lock_guard<std::mutex> lock(pendingBatchesMtx);
            std::lock_guard<std::mutex> lock_loaded(loadedBatchesMtx);
            has_demoted = loaderDemoteWorstLoaded(picked_priority, camera, demoted);
            pendingBatches.insert(pendingBatches.end(), demoted.begin(), demoted.end());
        }
        record_demoted();
        if(has_demoted){ slot = take_slot(); }
    }
    if(slot == OocSimLodSettings::BATCHES_LIST_SIZE){
        // No free slot: back to the pending batches for later
        finish(true);
        return false;
    }
    const std::shared_ptr<PointBatch>& batch = picked.batch;

    // 3. Load it into a local buffer: nothing is published unless the whole batch was read
    //    (after a LAZ decoding error, the points decoded just before it can't be trusted)
    auto release_slot = [&](){
        std::lock_guard<std::mutex> lock(batchesQueueMutexes[slot]);
        batchesQueue[slot] = nullptr;
    };
    auto fail = [&](const char* reason){
        release_slot();
        picked.nb_failures++;
        const bool will_retry = picked.nb_failures < MAX_LOAD_ATTEMPTS;
        if(will_retry){
            println("WARNING: {} for '{}' (points {} to {}), attempt {}/{}; the batch will be loaded again",
                reason, *batch->file, batch->first, batch->first + batch->count, picked.nb_failures, MAX_LOAD_ATTEMPTS
            );
            batch->points.reset();
            batch->state = BatchState::ToLoad;
        } else {
            println("ERROR: {} for '{}' (points {} to {}), {} attempts failed; the batch is skipped",
                reason, *batch->file, batch->first, batch->first + batch->count, picked.nb_failures
            );
        }
        finish(will_retry);
    };
    // Cancelled because a pending batch became clearly better (camera move): nothing of it was
    // published, it goes back to the pending batches as it was (not a failure)
    auto cancel = [&](){
        release_slot();
        batch->points.reset();
        batch->state = BatchState::ToLoad;
        finish(true);
    };
    auto should_cancel = [&]() -> bool {
        if(stopLoadingWorkers){ return true; }
        const LoadPriority mine = loadPriority(picked.bounds, getCameraState());
        const PendingSummary summary = getPendingSummary();
        return summary.has_any && isClearlyBetter(summary.best, mine);
    };

    std::vector<CPoint> points = {};
    const char* failure = nullptr;
    bool is_cancelled = false;
    {
        // At most `nbLoadingThreads()' decodings at once
        // (no per-file lock: every worker has its own LASzip reader, so several workers can decode
        // different batches of the same file, see `MAX_READERS_PER_FILE')
        LoaderDecodingSlot decoding_slot;
        // Closed and destroyed on every exit of this scope, exceptions included (an escaping
        // exception used to leak the reader, its file, the slot and the in-flight counters)
        struct ReaderGuard {
            laszip_POINTER reader = nullptr;
            bool is_open = false;
            ~ReaderGuard(){
                if(is_open){ laszip_close_reader(reader); }
                if(reader){ laszip_destroy(reader); }
            }
        } guard;
        laszip_point* laz_point = nullptr;
        {
            // LASzip's logging (lasmessage.cpp) is global and not thread-safe: opening a file and the
            // first decoding (which reads the chunk table) may log, so they are serialised
            std::lock_guard<std::mutex> lock(laszipReaderMtx);
            laszip_BOOL is_compressed = 0;
            if(laszip_create(&guard.reader)){
                guard.reader = nullptr;
                failure = "creating the laszip reader";
            } else if(laszip_open_reader(guard.reader, (*batch->file).c_str(), &is_compressed)){
                failure = "opening the file";
            } else {
                guard.is_open = true;
                if(laszip_get_point_pointer(guard.reader, &laz_point) || laszip_read_point(guard.reader)){
                    failure = "initialising the decoder";
                }
            }
        }

        if(!failure && laszip_seek_point(guard.reader, static_cast<laszip_I64>(batch->first))){
            failure = "seeking the first point";
        }
        if(!failure){
            // The batch size is read once: nothing may change it while it is being loaded
            const uint64_t nb_points_to_read = batch->count;
            const uint64_t check_interval = std::max<uint64_t>(1, CANCEL_CHECK_INTERVAL);
            const LoaderPointConverter convert(*batch->header);
            try {
                points.reserve(nb_points_to_read);
                for(uint64_t i = 0; i < nb_points_to_read; i++){
                    // Still worth it with the current camera? (not once half of it is decoded)
                    if(i > 0 && (i % check_interval) == 0 && 2 * i < nb_points_to_read && should_cancel()){
                        is_cancelled = true;
                        break;
                    }
                    if(laszip_read_point(guard.reader)){
                        failure = "reading a point";
                        break;
                    }
                    convert(laz_point, points.emplace_back());   // written in place (capacity reserved)
                }
            } catch(const std::bad_alloc&){
                failure = "out of host memory";
                points = {};
            }
        }
    }
    if(failure){ fail(failure); return true; }
    if(is_cancelled){ cancel(); return true; }

    // Points outside of the file's extent can only be decoding garbage (a corrupted LAZ chunk is only
    // detected some points after the decoder started producing garbage, possibly in the next batch):
    // dropped. The tolerance covers the quantisation of the coordinates and a slightly stale header.
    {
        const laszip_header& h = *batch->header;
        const double tolerance_x = 2.0 * h.x_scale_factor + 1e-3 * (h.max_x - h.min_x);
        const double tolerance_y = 2.0 * h.y_scale_factor + 1e-3 * (h.max_y - h.min_y);
        const double tolerance_z = 2.0 * h.z_scale_factor + 1e-3 * (h.max_z - h.min_z);
        const glm::vec3 lo = glm::vec3(float(h.min_x - tolerance_x), float(h.min_y - tolerance_y), float(h.min_z - tolerance_z));
        const glm::vec3 hi = glm::vec3(float(h.max_x + tolerance_x), float(h.max_y + tolerance_y), float(h.max_z + tolerance_z));
        const size_t nb_read = points.size();
        points.erase(std::remove_if(points.begin(), points.end(), [&](const CPoint& point){
            const glm::vec3& p = point.position;
            return !(p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y && p.z >= lo.z && p.z <= hi.z);
        }), points.end());
        if(points.size() != nb_read){
            static std::mutex warned_mtx;
            static std::unordered_set<std::string> warned_files;
            std::lock_guard<std::mutex> lock(warned_mtx);
            if(warned_files.insert(*batch->file).second){
                println("WARNING: '{}' (points {} to {}): {} points outside of the file's extent were dropped; the file may be corrupted (further cases for this file not reported)",
                    *batch->file, batch->first, batch->first + batch->count, nb_read - points.size()
                );
            }
        }
    }
    if(points.empty()){
        // Nothing valid: free the slot, nothing to send
        release_slot();
        finish(false);
        return true;
    }
    batch->count = static_cast<uint32_t>(points.size());

    BatchBounds loaded_bounds = {};
    for(const CPoint& point : points){
        loaded_bounds.extend(point.position);
    }
    try {
        batch->points = std::make_shared<vector<CPoint>>(std::move(points));
    } catch(const std::bad_alloc&){
        fail("out of host memory");
        return true;
    }

    // 4. Publish it, with its exact bounds (written before it is visible in `loadedBatches')
    slotsBounds[slot] = loaded_bounds;
    {
        std::lock_guard<std::mutex> lock(loadedBatchesMtx);
        loadedBatches.push_back({slot, batch});
    }
    {
        std::lock_guard<std::mutex> lock(filesRecordMtx);
        filesRecord[*batch->file].nb_batches_loaded.fetch_add(1);
    }
    finish(false);
    return true;
}

void LoaderGpuVersion::loadPendingBatches(){
    // Parallel: persistent workers, each loading the current best batch as soon as it is free
    // (no round waiting for its slowest batch, decisions taken with the camera of the moment)
    {
        std::lock_guard<std::mutex> lock(loadingWorkersMtx);
        if(loadingWorkers.empty() && !stopLoadingWorkers){
            // Not every core: see `nbLoadingThreads' (`MAX_LOADING_THREADS' in settings.toml)
            const uint32_t nb_workers = nbLoadingThreads();
            nbLoadingWorkers = nb_workers;
            // Joined before the static data they use is destroyed, however the program exits: an
            // `atexit' handler registered now runs before the destructors of the statics constructed
            // earlier (an error elsewhere used to destroy `filesBeingRead' under a running worker)
            static std::once_flag exit_handler_flag;
            std::call_once(exit_handler_flag, [](){ std::atexit([](){ LoaderGpuVersion::stopLoading(); }); });
            for(uint32_t w = 0; w < nb_workers; w++){
                loadingWorkers.emplace_back([](){
                    while(!stopLoadingWorkers){
                        bool has_loaded = false;
                        try {
                            has_loaded = loadOneBatch();
                        } catch(const std::exception& e){
                            println("ERROR: loading worker: {}", e.what());
                        } catch(...){
                            println("ERROR: loading worker: unknown exception");
                        }
                        if(!has_loaded){
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    }
                });
            }
        }
    }
    // The calling loop (`loadingRoutine') keeps clearing the inserted batches
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

void LoaderGpuVersion::stopLoading(){
    stopLoadingWorkers = true;
    {
        std::lock_guard<std::mutex> lock(loadingWorkersMtx);
        for(std::thread& worker : loadingWorkers){
            if(worker.joinable()){ worker.join(); }
        }
        loadingWorkers.clear();
    }
    // Wait for the calls of `loadingRoutine' in progress on other threads (they check
    // `stopLoadingWorkers' after announcing themselves, so none can start after this)
    const uint32_t own = loaderIsInsideLoadingRoutine ? 1u : 0u;
    while(nbLoadingRoutinesRunning.load() > own){
        std::this_thread::yield();
    }
}









void LoaderGpuVersion::filesRecordUi(){
	ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("File Explorer")) {
        ImGui::End();
        return;
    }

    // Zoom on the first overview (the chunk points of the first dropped files)
    {
        bool has_first_overview = false;
        glm::vec3 mins = glm::vec3(0.0f);
        glm::vec3 maxs = glm::vec3(0.0f);
        {
            LoaderFirstOverview& first_overview = loaderFirstOverview();
            std::lock_guard<std::mutex> lock(first_overview.mtx);
            has_first_overview = first_overview.is_valid;
            mins = first_overview.mins;
            maxs = first_overview.maxs;
        }
        if (!has_first_overview) { ImGui::BeginDisabled(); }
        if (ImGui::Selectable("Overview", false, ImGuiSelectableFlags_None, ImVec2(0, 0)) && has_first_overview) {
            printf("Zooming out on the entire cloud\n");
            Runtime::controls->focus(mins, maxs, 1.f);
        }
        if (!has_first_overview) { ImGui::EndDisabled(); }
        ImGui::Separator();
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
        // Collapsed when it first appears (no `ImGuiTreeNodeFlags_DefaultOpen'): ImGui then keeps
        // the open / closed state chosen by the user (the label is the node id)
        bool dirOpen = ImGui::TreeNodeEx(dir.c_str(),
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
                if (clicked && header) {
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