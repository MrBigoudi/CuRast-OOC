#pragma once

#include <cmath>
#include "../../ooc_structures/settings.h"
#include "./GpuVersionStructs.h"

typedef uint32_t CIdAABB;
constexpr CIdAABB CINVALID_ID = UINT32_MAX;

enum CNodePosition {
	CFrontTopLeft,
	CFrontTopRight,
	CFrontBottomLeft,
	CFrontBottomRight,
	CBackTopLeft,
	CBackTopRight,
	CBackBottomLeft,
	CBackBottomRight,
};
__device__ __forceinline__ void updateNodePosition(CNodePosition& position){
	switch(position){
        case CFrontTopLeft:
            position = CFrontTopRight;
            break;
        case CFrontTopRight:
            position = CFrontBottomLeft;
            break;
        case CFrontBottomLeft:
            position = CFrontBottomRight;
            break;
        case CFrontBottomRight:
            position = CBackTopLeft;
            break;
        case CBackTopLeft:
            position = CBackTopRight;
            break;
        case CBackTopRight:
            position = CBackBottomLeft;
            break;
        case CBackBottomLeft:
            position = CBackBottomRight;
            break;
        case CBackBottomRight:
            position = CFrontTopLeft;
            break;
    }
}

struct CAABB {
	glm::vec3 mins = {INFINITY, INFINITY, INFINITY};
	glm::vec3 maxs = {-INFINITY, -INFINITY, -INFINITY};

	__device__ __forceinline__  CAABB(){}
	__device__ __forceinline__  CAABB(const CAABB& cpy) : mins(cpy.mins), maxs(cpy.maxs){}

	__device__ __forceinline__  bool operator==(const CAABB& rhs) const{
		return rhs.mins == mins && rhs.maxs == maxs;
	}

	inline glm::vec3 getCentroid() const {
		return 0.5f * (mins + maxs);
	}

	__device__ __forceinline__  glm::vec3 getSize() const {return maxs - mins;}
	__device__ __forceinline__  bool contains(const glm::vec3& position) const {
		return position.x > mins.x && position.x < maxs.x
			&& position.y > mins.y && position.y < maxs.y
			&& position.z > mins.z && position.z < maxs.z
		;
	}
	__device__ __forceinline__  void extend(const CNodePosition& position) {
		glm::vec3 size = getSize();
		switch (position) {
			case CFrontTopLeft:
				maxs.x += size.x;
				mins.y -= size.y;
				mins.z -= size.z;
				break;
			case CFrontTopRight:
				mins.x -= size.x;
				mins.y -= size.y;
				mins.z -= size.z;
				break;
			case CFrontBottomLeft:
				maxs.x += size.x;
				maxs.y += size.y;
				mins.z -= size.z;
				break;
			case CFrontBottomRight:
				mins.x -= size.x;
				maxs.y += size.y;
				mins.z -= size.z;
				break;
			case CBackTopLeft:
				maxs.x += size.x;
				mins.y -= size.y;
				maxs.z += size.z;
				break;
			case CBackTopRight:
				mins.x -= size.x;
				mins.y -= size.y;
				maxs.z += size.z;
				break;
			case CBackBottomLeft:
				maxs.x += size.x;
				maxs.y += size.y;
				maxs.z += size.z;
				break;
			case CBackBottomRight:
				mins.x -= size.x;
				maxs.y += size.y;
				maxs.z += size.z;
				break;
		}
	}

	__device__ __forceinline__  void shrink(const CNodePosition& position) {
		glm::vec3 size = getSize()*0.5f;
		switch (position) {
			case CFrontTopLeft:
				maxs.x -= size.x;
				mins.y += size.y;
				mins.z += size.z;
				break;
			case CFrontTopRight:
				mins.x += size.x;
				mins.y += size.y;
				mins.z += size.z;
				break;
			case CFrontBottomLeft:
				maxs.x -= size.x;
				maxs.y -= size.y;
				mins.z += size.z;
				break;
			case CFrontBottomRight:
				mins.x += size.x;
				maxs.y -= size.y;
				mins.z += size.z;
				break;
			case CBackTopLeft:
				maxs.x -= size.x;
				mins.y += size.y;
				maxs.z -= size.z;
				break;
			case CBackTopRight:
				mins.x += size.x;
				mins.y += size.y;
				maxs.z -= size.z;
				break;
			case CBackBottomLeft:
				maxs.x -= size.x;
				maxs.y -= size.y;
				maxs.z -= size.z;
				break;
			case CBackBottomRight:
				mins.x += size.x;
				maxs.y -= size.y;
				maxs.z -= size.z;
				break;
		}
	}

	__device__ __forceinline__  glm::vec3 getPointNormalizedCoordinates(const glm::vec3& position) const {
		return glm::vec3(
			(position.x - mins.x) / (maxs.x - mins.x),
			(position.y - mins.y) / (maxs.y - mins.y),
			(position.z - mins.z) / (maxs.z - mins.z)
		);
	}

	__device__ __forceinline__ CNodePosition getNextChildIndex(const glm::vec3& position) const {
		glm::vec3 normalized_coordinates = getPointNormalizedCoordinates(position);
		bool is_front = normalized_coordinates.z >= 0.5f;
		bool is_top = normalized_coordinates.y >= 0.5f;
		bool is_right = normalized_coordinates.x >= 0.5f;

		// if(normalized_coordinates.x < 0.f || normalized_coordinates.x > 1.f){
		// 	printf("WTF x\n");
		// 	#ifdef __CUDACC__
		// 		customAssert();
		// 	#endif
		// }
		// if(normalized_coordinates.y < 0.f || normalized_coordinates.y > 1.f){
		// 	printf("WTF x\n");
		// 	#ifdef __CUDACC__
		// 		customAssert();
		// 	#endif
		// }
		// if(normalized_coordinates.z < 0.f || normalized_coordinates.z > 1.f){
		// 	printf("WTF x\n");
		// 	#ifdef __CUDACC__
		// 		customAssert();
		// 	#endif
		// }

		if(is_right){
			if(is_top){
				if(is_front){
					return CNodePosition::CFrontTopRight;
				} else {
					return CNodePosition::CBackTopRight;
				}
			} else {
				if(is_front){
					return CNodePosition::CFrontBottomRight;
				} else {
					return CNodePosition::CBackBottomRight;
				}
			}
		} else {
			if(is_top){
				if(is_front){
					return CNodePosition::CFrontTopLeft;
				} else {
					return CNodePosition::CBackTopLeft;
				}
			} else {
				if(is_front){
					return CNodePosition::CFrontBottomLeft;
				} else {
					return CNodePosition::CBackBottomLeft;
				}
			}
		}
	}
};

struct CPoint {
	glm::vec3 position = {};
	uint32_t color = 0;

	__device__ __forceinline__ CPoint():position(glm::vec3(0,0,0)), color(0) {}
	__device__ __forceinline__ CPoint(const CPoint& cpy):position(cpy.position), color(cpy.color) {}

	__device__ __forceinline__  bool operator==(const CPoint& rhs) const{
		return rhs.position == position && rhs.color == color;
	}

	inline void setColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xFFu){
		color = (uint32_t)r
			| ((uint32_t)g << 8)
			| ((uint32_t)b << 16)
			| ((uint32_t)a << 24);
	}

	enum Channel {
		R, G, B, A
	};
	__device__ __forceinline__ const uint8_t getChannel(Channel channel) const {
		switch(channel){
			case R:
				return uint8_t((color << 24) >> 24);
			case G:
				return uint8_t(((color >> 8) << 24) >> 24);
			case B:
				return uint8_t(((color >> 16) << 24) >> 24);
			case A:
				return uint8_t(color >> 24);
		}
	}
	__device__ __forceinline__ const uint8_t getAlpha() const {
		return getChannel(Channel::A);
	}
	__device__ __forceinline__ void resetAlpha() {
		color &= (0x00FFFFFF); // Reset old alpha value
	}
	__device__ __forceinline__ void setAlpha(uint8_t a) {
		resetAlpha();
		color |= (uint32_t(a) << 24);
	}
};

struct COccupancyGrid {
	uint32_t values[OocSimLodSettings::GRID_SIZE / 32] = {0};

	/// Store a pair (word_index, bit_index)
	struct GridIndex {
		uint32_t word;
		uint32_t bit;
		glm::uvec3 grid;
		__device__ __forceinline__ GridIndex(uint32_t word, uint32_t bit, glm::uvec3 grid):word(word), bit(bit), grid(grid){}
	};

	__device__ __forceinline__ static GridIndex getCellIndices(const CAABB& aabb, const CPoint& point){
		glm::vec3 normalized_coordinates = aabb.getPointNormalizedCoordinates(point.position);
		uint32_t grid_x = glm::clamp(
			uint32_t(floor(OocSimLodSettings::GRID_SIZE_PER_DIMENSION * normalized_coordinates.x)), 
			0u, 
			OocSimLodSettings::GRID_SIZE_PER_DIMENSION - 1u
		);
		uint32_t grid_y = glm::clamp(
			uint32_t(floor(OocSimLodSettings::GRID_SIZE_PER_DIMENSION * normalized_coordinates.y)), 
			0u, 
			OocSimLodSettings::GRID_SIZE_PER_DIMENSION - 1u
		);
		uint32_t grid_z = glm::clamp(
			uint32_t(floor(OocSimLodSettings::GRID_SIZE_PER_DIMENSION * normalized_coordinates.z)), 
			0u, 
			OocSimLodSettings::GRID_SIZE_PER_DIMENSION - 1u
		);
		uint32_t index = grid_x + OocSimLodSettings::GRID_SIZE_PER_DIMENSION * (grid_y + OocSimLodSettings::GRID_SIZE_PER_DIMENSION * grid_z);
		uint32_t word_index = index >> 5u;
		uint32_t bit_index = index & 31u;
		return GridIndex(word_index, bit_index, glm::uvec3(grid_x, grid_y, grid_z));
	}

#ifdef __CUDACC__
	/// Return true if the cell was already flagged
	__device__ __forceinline__ bool markCellAsFilled(const GridIndex& index){
		uint32_t old_value = __nv_atomic_fetch_or(&values[index.word], (1u << index.bit), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
		return (old_value & (1u << index.bit));
	}
#endif // __CUDACC__

	__device__ __forceinline__ static glm::vec3 getCellCentroid(const CAABB& aabb, const GridIndex& index){
		glm::vec3 world_grid_size = aabb.getSize() / float(OocSimLodSettings::GRID_SIZE_PER_DIMENSION);
		return aabb.mins + world_grid_size * glm::vec3(index.grid.x, index.grid.y, index.grid.z) + 0.5f * world_grid_size;
	}
};

struct CChunk{
	CPoint points[OocSimLodSettings::NB_POINTS_PER_CHUNK] = {CPoint()};
	uint32_t size = 0;
	CChunk* next = nullptr;

	__device__ __forceinline__ CChunk(): size(0), next(nullptr) {
		// for(uint32_t i=0; i<OocSimLodSettings::NB_POINTS_PER_CHUNK; i++){
		// 	points[i] = CPoint();
		// }
	}

	~CChunk(){
		size = 0;
		next = nullptr;
	}
};

enum CNodeFlagType {
	// Counter for a node
	CFlagCounter0,
	CFlagCounter1,
	CFlagCounter2,
	CFlagCounter3,
	CFlagCounter4,
	CFlagCounter5,
	CFlagCounter6,
	CFlagCounter7,

	// Node properties
	CFlagIsUpdated,
	CFlagToLoad,
	CFlagToStore,
	CFlagIsInUpdatesCache,
	CFlagWillBeInUpdatesCache,

	// Rendering properties
	CFlagHasEnoughPoints,
	CFlagHasEnoughVoxels,
	CFlagIsVisible,
	CFlagIsLarge,
	CFlagIsCut,
	CFlagIsInVisibilityCache,
	CFlagIsFromVoxelsInVisibilityCache,

	// Pads to be replaced on need
	CFlagPad20,
	CFlagPad21,
	CFlagPad22,
	CFlagPad23,
	CFlagPad24,
	CFlagPad25,
	CFlagPad26,
	CFlagPad27,
	CFlagPad28,
	CFlagPad29,
	CFlagPad30,
	CFlagPad31,
};

struct COctreeNode {
	COctreeNode* children[8] = {nullptr};
	CChunk* points = nullptr;
	CChunk* voxels = nullptr;
	COccupancyGrid* occupancy = nullptr;
	CIdAABB aabb_index = CINVALID_ID;

	uint32_t points_counter = 0;
	uint32_t voxels_counter = 0;

	uint32_t points_stored = 0;
	uint32_t voxels_stored = 0;

	uint32_t points_last_stored = 0;
	uint32_t voxels_last_stored = 0;

	uint32_t children_ids = 0;
	uint8_t level = 0;

	__device__ __forceinline__ COctreeNode(): points(nullptr), voxels(nullptr), occupancy(nullptr),
		aabb_index(CINVALID_ID), points_counter(0), voxels_counter(0),
		points_stored(0), voxels_stored(0), children_ids(0), level(0)
	{
		for(uint32_t i=0; i<8; i++){
			children[i] = nullptr;
		}
	}

	~COctreeNode(){
		for(uint32_t i=0; i<8; i++){
			children[i] = nullptr;
		}
		points = nullptr;
		voxels = nullptr;
		occupancy = nullptr;
		aabb_index = CINVALID_ID;
		points_counter = 0;
		voxels_counter = 0;
		points_stored = 0;
		voxels_stored = 0;
		children_ids = 0b00000000;
		level = 0;
	}
};

struct CFullOctree {
	glm::mat4 world;
	COctreeNode** nodes;
	CAABB* aabbs;
	uint32_t num_nodes;
	uint32_t max_lod_level;

	// TODO: put inside uniforms structure
	int32_t debug_lod_to_render = -1;
	uint32_t voxels_nb_points_per_axis = 1;
	float min_pixel_span = 0.;
	bool use_voxels_debug_color = false;
	bool use_aabb_debug_color = false;
	uint32_t nb_blocks_per_node = 0;
};




/// TODO: refactor
/// Tests unified memory


struct CAABBUnified;
struct CPointUnified;
struct COccupancyGridUnified;
struct CChunkUnified;
struct COctreeNodeUnified;

struct CFullOctreeUnified {
	glm::mat4 world;
	COctreeNodeUnified** nodes;
	uint32_t num_nodes;
	uint32_t max_lod_level;
	
	int32_t debug_lod_to_render;
	uint32_t voxels_nb_points_per_axis;
	float min_pixel_span;
	bool use_voxels_debug_color;
};
struct CAABBUnified {
	glm::vec3 mins = {INFINITY, INFINITY, INFINITY};
	glm::vec3 maxs = {-INFINITY, -INFINITY, -INFINITY};
};
struct CPointUnified {
	glm::vec3 position = glm::vec3();
	uint8_t color[4] = {0,0,0,0};
};
struct COccupancyGridUnified {
	uint32_t values[OocSimLodSettings::GRID_SIZE / 32u] = {0};
};
struct CChunkUnified {
	CPointUnified points[OocSimLodSettings::NB_POINTS_PER_CHUNK] = {CPointUnified()};
	uint32_t size = 0;
	CChunkUnified* next = nullptr;
};
struct COctreeNodeUnified {
	COctreeNodeUnified* children[8] = {nullptr};
	uint16_t counter = 0;
	uint8_t children_ids = 0b00000000;
	CChunkUnified* points = nullptr;
	CChunkUnified* voxels = nullptr;
	COccupancyGridUnified* occupancy = nullptr;
	bool updated = false;
	CAABBUnified* aabb = nullptr;
	uint8_t level = 0;
	bool is_large = false;
	bool is_visible = false;
	bool is_cut = false;
};



struct CRenderTarget{
	uint64_t* framebuffer;
	uint64_t* colorbuffer;
	int width;
	int height;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec3 camera_pos;
};



struct CAllocationPlan {
	CChunk* points_attach;   // chunk to append new points chunks to (nullptr if none needed)
    uint32_t points_first_new;
    uint32_t points_nb_new;
    uint32_t points_last_size;

    CChunk* voxels_attach;   // chunk to append new voxels chunks to (nullptr if none needed)
    uint32_t voxels_first_new;
    uint32_t voxels_nb_new;
    uint32_t voxels_last_size;
};




struct CSemaphore {
#ifdef __CUDACC__
	enum State {
		IN_USE = 0,
		FREE = 1,
	};
	uint32_t flag = State::FREE;
	
	__device__ __forceinline__ CSemaphore(const State& state = State::FREE){
		flag = state;
	}

	// Try to acquire the semaphore but do not block
	// Return true is the semaphore was successfully acquired
	__device__ __forceinline__ bool tryAcquire(){
		uint32_t old_flag = __nv_atomic_fetch_and(&flag, 0, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
		// If old semaphore was free
		return (old_flag == State::FREE);
	}

	// Block until the semaphore can be acquired
	__device__ __forceinline__ void acquire(){
		while(!tryAcquire()){}
	}

	// Release the semaphore
	// It's the user responsability to only free a semaphore it already has acquired
	__device__ __forceinline__ void release(){
		__nv_atomic_or(&flag, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}
#endif // __CUDACC__
};

struct CGlobalVariables {
    ///////////////////////////////////////////////////////////////////////
    /////////////////////////// UNBOUNDED DATA ////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    // TODO: Be carefull to double their sizes on need
    /// The relationship map for all AABBs created during runtime
    struct Relationship {
        CIdAABB children[8] = { 
            CINVALID_ID, CINVALID_ID, CINVALID_ID, CINVALID_ID,
            CINVALID_ID, CINVALID_ID, CINVALID_ID, CINVALID_ID
        };
		CIdAABB parent = CINVALID_ID;
		CAABB aabb = CAABB();
    };
    /// The list of all AABBs created during runtime
	uint32_t totalNbNodes = 0;
    uint32_t maxNbConcurrentNodes = 0;
    Relationship* relationshipMap = nullptr;
	uint32_t* nodesFlags = nullptr;

	/// The main octree
	bool isInitialised = false;
	bool isUpdating = false;
	bool isTemporarySwitching = false;
	COctreeNode* mainOctree = nullptr;

	/// The buffer of nodes for updates
	uint32_t curNbNodes = 0;
	COctreeNode** packedNodes = nullptr;

	/// The buffer of nodes for rendering
	uint32_t octreeDepth = 0;

	uint32_t maxCountSplitIterations = 0;

	uint32_t temporaryBufferSize = 0;
	CIdAABB* temporaryIdBuffer = nullptr;
	CIdAABB* temporaryIdBuffer2 = nullptr;
    COctreeNode** temporaryNodeBuffer = nullptr;
	uint32_t nbNodesExchangedBeforeLoadComplete = 0;

	uint32_t chunksAllocatorCounter = 0;
	uint32_t maxAllocatedChunks = 0;
	CChunk** allocatedChunks = nullptr;
	CAllocationPlan* allocationPlans = nullptr;
	
	COctreeNode** memoizedBatchPointsNodes = nullptr; 
	COctreeNode** memoizedSpilledPointsNodes = nullptr; 


	uint32_t nbOrderedNodes = 0;


    ///////////////////////////////////////////////////////////////////////
    ////////////////////////// EXCHANGEABLE DATA //////////////////////////
    ///////////////////////////////////////////////////////////////////////

	/// TODO: use cuda unified memory for them ??
	/// Data exchanged from the host
	uint32_t nbNodesExchanged = 0;
	uint32_t maxNbNodesExchanged = 0;
	uint32_t maxNbPointsChunksPerExchangedNode = 0;
	uint32_t maxNbVoxelsChunksPerExchangedNode = 0;
	CIdAABB* exchangedAABBIndices = nullptr;
	CIdAABB* exchangedAABBParentsIndices = nullptr;
	CAABB* exchangedAABBs = nullptr;
	uint32_t* exchangedChildrenIds = nullptr;
	uint32_t* exchangedPointsCounters = nullptr;
	uint32_t* exchangedVoxelsCounters = nullptr;
	CPoint** exchangedPoints = nullptr;
	CPoint** exchangedVoxels = nullptr;
	uint64_t** exchangedGrids = nullptr;

	bool isDoneLoading = true;
	bool isDoneStoring = true;
	bool isDoneIterating = true;
	bool isFirstCountSplitIteration = true;

	uint32_t nbGridsToInit = 0;
	COctreeNode** gridsToInit = nullptr;
	uint32_t* gridsToInitExchangedIndex = nullptr;

    /// A mask to know which batches have been handled
    uint32_t maxNbBatches = 0;
	uint32_t maxBatchSize = 0;
    uint32_t* batchesAddedMask = nullptr;
	/// The batches to add to the scene
	CPoint** batchesToAddPoints = nullptr;
	uint32_t* batchesToAddCounts = nullptr;
	uint32_t batchesToAddBottomUpCount = 0;



	/// The rendering budget
	uint32_t nbRenderedPoints = 0;
	uint32_t maxNbRenderedPoints = 0;
	CPoint* renderedPoints = nullptr;
	uint32_t nbRenderedVoxels = 0;
	uint32_t maxNbRenderedVoxels = 0;
	CPoint* renderedVoxels = nullptr;
	// glm::vec3* renderedVoxelsSizes = nullptr;
	// CNodePosition* renderedVoxelsNextChildIndex = nullptr;
	CIdAABB* renderedVoxelsNodes = nullptr;



    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////// LRU CACHES //////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    uint32_t updatesCacheSize = 0;
    CIdAABB* updatesCache = nullptr;
	
	uint32_t visibilityCacheSize = 0;
	uint32_t visibilityCacheCurrentSize = 0;
	CIdAABB* visibilityCache = nullptr;
	


    ///////////////////////////////////////////////////////////////////////
    ////////////////////////// TEMPORARY BUFFERS //////////////////////////
    ///////////////////////////////////////////////////////////////////////
    /// The list of spilled points
    bool hasSpillingNodes = false;
    uint32_t nbSpilledPoints = 0;
    uint32_t maxNbSpilledPoints = 0;
    CPoint* spilledPoints = nullptr;
    uint32_t nbSpillingNodes = 0;
	uint32_t nbSpillingChunks = 0;
    COctreeNode** spillingNodes = nullptr;
	uint32_t* spilledChunksCounter = nullptr;
	CChunk** spillingChunks = nullptr;

    /// The backlog buffer for new voxels
    uint32_t nbBacklogVoxels = 0;
    uint32_t maxNbBacklogVoxels = 0;
    CPoint* backlogVoxels = nullptr;
    COctreeNode** backlogVoxelsNodes = nullptr;

	uint32_t maxPointsPerLeaf = 0;




    ///////////////////////////////////////////////////////////////////////
    ///////////////////////// UI DISPLAYED VALUES /////////////////////////
    ///////////////////////////////////////////////////////////////////////

	uint32_t nbTotalUpdates = 0;

	uint32_t currentNbChunks = 0;
	uint32_t currentNbGrids = 0;
    uint32_t currentNbPoints = 0;
    uint32_t currentNbVoxels = 0;

    uint32_t nbTotalPoints = 0;
    uint32_t nbTotalVoxels = 0;
    uint32_t nbTotalNewNodes = 0;
    uint32_t nbTotalNewGrids = 0;
    uint32_t nbTotalNewChunks = 0;
	uint32_t nbTotalDeletedNodes = 0;
    uint32_t nbTotalDeletedGrids = 0;
    uint32_t nbTotalDeletedChunks = 0;
	uint32_t nbTotalLoadedNodes = 0;
    uint32_t nbTotalSplitNodes = 0;
    uint32_t nbTotalStoredNodes = 0;

	uint32_t nbNewPointsThisUpdate = 0;
	uint32_t nbNewVoxelsThisUpdate = 0;
	uint32_t nbNewNodesThisUpdate = 0;
	uint32_t nbLoadedNodesThisUpdate = 0;
	uint32_t nbStoredNodesThisUpdate = 0;
	uint32_t nbSplitNodesThisUpdate = 0;
	uint32_t nbDeletedNodesThisUpdate = 0;
	uint32_t nbDeletedChunksThisUpdate = 0;
	uint32_t nbDeletedGridsThisUpdate = 0;
	uint32_t nbNewChunksThisUpdate = 0;
	uint32_t nbNewGridsThisUpdate = 0;



#ifdef __CUDACC__
	__device__ __forceinline__ uint32_t getFlagsSync(const CIdAABB& aabb_index) const {
		return __nv_atomic_load_n(&nodesFlags[aabb_index], __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}
	__device__ __forceinline__ uint32_t getFlags(const CIdAABB& aabb_index) const {
		return nodesFlags[aabb_index];
	}
	__device__ __forceinline__ bool getFlag(const CIdAABB& aabb_index, const CNodeFlagType& flag) const {
		return nodesFlags[aabb_index] & (0x01 << flag);
	}
	__device__ __forceinline__ void setFlags(const CIdAABB& aabb_index, uint32_t flags){
		nodesFlags[aabb_index] = flags;
	}
	__device__ __forceinline__ void setFlagsSync(const CIdAABB& aabb_index, uint32_t flags){
		__nv_atomic_and(&nodesFlags[aabb_index], flags, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}
	__device__ __forceinline__ void setFlag(const CIdAABB& aabb_index, const CNodeFlagType& flag){
		nodesFlags[aabb_index] |= (0x01 << flag);
	}
	__device__ __forceinline__ void setFlagSync(const CIdAABB& aabb_index, const CNodeFlagType& flag){
		__nv_atomic_or(&nodesFlags[aabb_index], (0x01 << flag), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}
	__device__ __forceinline__ uint32_t fetchSetFlagSync(const CIdAABB& aabb_index, const CNodeFlagType& flag){
		return __nv_atomic_fetch_or(&nodesFlags[aabb_index], (0x01 << flag), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}
	__device__ __forceinline__ void resetFlagsSync(const CIdAABB& aabb_index){
		__nv_atomic_and(&nodesFlags[aabb_index], 0, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}
	__device__ __forceinline__ void resetFlags(const CIdAABB& aabb_index){
		nodesFlags[aabb_index] = 0;
	}
	__device__ __forceinline__ void unsetFlagSync(const CIdAABB& aabb_index, const CNodeFlagType& flag){
		__nv_atomic_and(&nodesFlags[aabb_index], ~(1u << flag), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}
	__device__ __forceinline__ void unsetFlag(const CIdAABB& aabb_index, const CNodeFlagType& flag){
		nodesFlags[aabb_index] &= ~(1u << flag);
	}
	__device__ __forceinline__ uint32_t fetchUnsetFlagSync(const CIdAABB& aabb_index, const CNodeFlagType& flag){
		return __nv_atomic_fetch_and(&nodesFlags[aabb_index], ~(1u << flag), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}


	__device__ __forceinline__ uint32_t getCounterFlagMask() const {
		uint32_t mask = (0x01 << CFlagCounter0)
			| (0x01 << CFlagCounter1)
			| (0x01 << CFlagCounter2)
			| (0x01 << CFlagCounter3)
			| (0x01 << CFlagCounter4)
			| (0x01 << CFlagCounter5)
			| (0x01 << CFlagCounter6)
			| (0x01 << CFlagCounter7)
		;
		return mask;
	}
	__device__ __forceinline__ uint32_t resetCounterFlagSync(const CIdAABB& aabb_index) const {
		uint32_t mask = getCounterFlagMask();
		__nv_atomic_and(&nodesFlags[aabb_index], ~mask, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}

	__device__ __forceinline__ uint32_t getCounterFlagSync(const CIdAABB& aabb_index) const {
		uint32_t mask = getCounterFlagMask();
		uint32_t flags = getFlagsSync(aabb_index);
		return (flags & mask) >> CFlagCounter0;
	}
	
	__device__ __forceinline__ void setCounterFlagSync(const CIdAABB& aabb_index, uint8_t new_value) const {
		uint32_t new_counter = (new_value) << CFlagCounter0;
		// reset old counter
		resetCounterFlagSync(aabb_index);
		// set new counter
		__nv_atomic_or(&nodesFlags[aabb_index], new_counter, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}

	// Return the old counter
	__device__ __forceinline__ uint32_t increaseCounterFlagSync(const CIdAABB& aabb_index) const {
		uint32_t old_counter = getCounterFlagSync(aabb_index);
		if(old_counter == UINT8_MAX){
			printf("ERROR: Reached max of counter flag\n");
			return old_counter;
		}
		uint32_t new_counter = (old_counter + 1) << CFlagCounter0;
		// reset old counter
		resetCounterFlagSync(aabb_index);
		// set new counter
		__nv_atomic_or(&nodesFlags[aabb_index], new_counter, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
		return old_counter;
	}

	// Return the old counter
	__device__ __forceinline__ uint32_t decreaseCounterFlagSync(const CIdAABB& aabb_index) const {
		uint32_t old_counter = getCounterFlagSync(aabb_index);
		if(old_counter == 0){
			printf("ERROR: Can't decrease a 0 counter flag\n");
			return old_counter;
		}
		uint32_t new_counter = (old_counter - 1) << CFlagCounter0;
		// reset old counter
		resetCounterFlagSync(aabb_index);
		// set new counter
		__nv_atomic_or(&nodesFlags[aabb_index], new_counter, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
		return old_counter;
	}

	__device__ __forceinline__ uint32_t resetCounterFlag(const CIdAABB& aabb_index) const {
		uint32_t mask = getCounterFlagMask();
		nodesFlags[aabb_index] &= ~mask;
	}

	__device__ __forceinline__ uint32_t getCounterFlag(const CIdAABB& aabb_index) const {
		uint32_t mask = getCounterFlagMask();
		uint32_t flags = getFlags(aabb_index);
		return (flags & mask) >> CFlagCounter0;
	}
	
	__device__ __forceinline__ void setCounterFlag(const CIdAABB& aabb_index, uint8_t new_value) const {
		uint32_t new_counter = (new_value) << CFlagCounter0;
		resetCounterFlag(aabb_index);
		nodesFlags[aabb_index] |= new_counter;
	}

	// Return the old counter
	__device__ __forceinline__ uint32_t increaseCounterFlag(const CIdAABB& aabb_index) const {
		uint32_t old_counter = getCounterFlag(aabb_index);
		if(old_counter == UINT8_MAX){
			printf("ERROR: Reached max of counter flag\n");
			return old_counter;
		}
		uint32_t new_counter = (old_counter + 1) << CFlagCounter0;
		resetCounterFlag(aabb_index);
		nodesFlags[aabb_index] |= new_counter;
		return old_counter;
	}

	// Return the old counter
	__device__ __forceinline__ uint32_t decreaseCounterFlag(const CIdAABB& aabb_index) const {
		uint32_t old_counter = getCounterFlag(aabb_index);
		if(old_counter == 0){
			printf("ERROR: Can't decrease a 0 counter flag\n");
			return old_counter;
		}
		uint32_t new_counter = (old_counter - 1) << CFlagCounter0;
		resetCounterFlag(aabb_index);
		nodesFlags[aabb_index] |= new_counter;
		return old_counter;
	}


	__device__ __forceinline__ bool isInUpdatesCache(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagIsInUpdatesCache);
	}
	__device__ __forceinline__ bool willBeInUpdatesCache(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagWillBeInUpdatesCache);
	}
	__device__ __forceinline__ bool isUpdated(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagIsUpdated);
	}
	__device__ __forceinline__ bool isLarge(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagIsLarge);
	}
	__device__ __forceinline__ bool isVisible(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagIsVisible);
	}
	__device__ __forceinline__ bool isCut(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagIsCut);
	}
	__device__ __forceinline__ bool isInVisibilityCache(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagIsInVisibilityCache);
	}
	__device__ __forceinline__ bool isFromVoxelsInVisibilityCache(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagIsFromVoxelsInVisibilityCache);
	}
	__device__ __forceinline__ bool hasEnoughPoints(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagHasEnoughPoints);
	}
	__device__ __forceinline__ bool hasEnoughVoxels(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagHasEnoughVoxels);
	}

#endif // __CUDACC__
};




enum PipelineLevel {
	LevelInit,
	LevelBottomUp,
	LevelSimlodLoad,
	LevelSimlodSplitCount,
	LevelSimlodVoxelSampling,
	LevelSimlodInsertion,
	LevelSimlod,
	LevelCacheUpdate
};


struct CRenderingSettings {
	int32_t debug_lod_to_render = 0;
	bool use_voxels_debug_color = false;
	uint32_t min_pixel_span = 0;
	uint32_t voxels_nb_points_per_axis = 0;
};