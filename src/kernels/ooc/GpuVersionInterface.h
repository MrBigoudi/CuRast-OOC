#pragma once

#include <cmath>
#include "../../ooc_structures/settings.h"

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
		return position.x >= mins.x && position.x <= maxs.x
			&& position.y >= mins.y && position.y <= maxs.y
			&& position.z >= mins.z && position.z <= maxs.z
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
		return getChildIndex(is_right, is_top, is_front);
	}

	/// Octant from the half of the box along each axis (shared by `getNextChildIndex' and
	/// `CVoxel::getOctant', so that a point and the voxel of its cell always agree)
	__device__ __forceinline__ static CNodePosition getChildIndex(bool is_right, bool is_top, bool is_front) {
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
		return getCellCentroid(aabb, index.grid);
	}
	__device__ __forceinline__ static glm::vec3 getCellCentroid(const CAABB& aabb, const glm::uvec3& cell){
		glm::vec3 world_grid_size = aabb.getSize() / float(OocSimLodSettings::GRID_SIZE_PER_DIMENSION);
		return aabb.mins + world_grid_size * glm::vec3(cell.x, cell.y, cell.z) + 0.5f * world_grid_size;
	}

	/// Word / bit of a cell from its coordinates (same layout as `getCellIndices')
	__device__ __forceinline__ static GridIndex getCellIndices(const glm::uvec3& cell){
		uint32_t index = cell.x + OocSimLodSettings::GRID_SIZE_PER_DIMENSION * (cell.y + OocSimLodSettings::GRID_SIZE_PER_DIMENSION * cell.z);
		return GridIndex(index >> 5u, index & 31u, cell);
	}
};

/// A voxel: an occupied cell of the occupancy grid of an inner node, with the color of the first point
/// that fell into it. Its position is the centre of its cell, so it only stores the cell coordinates
/// within the grid of its node (8 bits per axis, GRID_SIZE_PER_DIMENSION <= 256): the position is
/// rebuilt from the bounding box of the node (`getPosition'), which is always known where a voxel is
/// used (a node's chunks, an exchange slot, a retained entry, a segment of the rendering buffers).
/// Lossless: the position of a voxel always was the centre of its cell
/// 8 bytes instead of 16 (`CPoint'): voxel chunks, exchange / backlog / rendering buffers, host
/// memory and node files are halved, and the separate occupancy indices (8 bytes per voxel) are gone:
/// the occupancy grid of a loaded node is rebuilt from the cells of its voxels
struct CVoxel {
	/// x | y << 8 | z << 16; bits 24-31 are reserved (0)
	uint32_t cell = 0;
	/// Same encoding as `CPoint::color'
	uint32_t color = 0;

	__device__ __forceinline__ CVoxel(): cell(0), color(0) {}
	__device__ __forceinline__ CVoxel(const glm::uvec3& grid_cell, uint32_t color):
		cell(pack(grid_cell)), color(color) {}

	__device__ __forceinline__ static uint32_t pack(const glm::uvec3& grid_cell){
		return (grid_cell.x & 0xFFu) | ((grid_cell.y & 0xFFu) << 8) | ((grid_cell.z & 0xFFu) << 16);
	}
	__device__ __forceinline__ glm::uvec3 getCell() const {
		return glm::uvec3(cell & 0xFFu, (cell >> 8) & 0xFFu, (cell >> 16) & 0xFFu);
	}

	/// Centre of the cell in the node with this bounding box (as `COccupancyGrid::getCellCentroid')
	__device__ __forceinline__ glm::vec3 getPosition(const CAABB& node_aabb) const {
		return COccupancyGrid::getCellCentroid(node_aabb, getCell());
	}

	/// Word / bit of the cell in the occupancy grid of its node
	__device__ __forceinline__ COccupancyGrid::GridIndex getGridIndex() const {
		return COccupancyGrid::getCellIndices(getCell());
	}

	/// Octant of the node holding the cell (same as `CAABB::getNextChildIndex' of the cell centre: the
	/// centre of cell c is in the upper half of an axis iff c >= GRID_SIZE_PER_DIMENSION / 2)
	__device__ __forceinline__ CNodePosition getOctant() const {
		constexpr uint32_t HALF = OocSimLodSettings::GRID_SIZE_PER_DIMENSION / 2;
		const glm::uvec3 c = getCell();
		return CAABB::getChildIndex(c.x >= HALF, c.y >= HALF, c.z >= HALF);
	}
};
static_assert(sizeof(CVoxel) == 8, "CVoxel is meant to be 8 bytes");
static_assert(OocSimLodSettings::GRID_SIZE_PER_DIMENSION <= 256, "CVoxel stores its cell on 8 bits per axis");

/// A chunk of points of a node (linked list)
struct CPointChunk {
	CPoint points[OocSimLodSettings::NB_POINTS_PER_CHUNK];
	uint32_t size;
	CPointChunk* next;
};

/// A chunk of voxels of a node (linked list): twice as many elements as a point chunk, same size
struct CVoxelChunk {
	CVoxel voxels[OocSimLodSettings::NB_VOXELS_PER_CHUNK];
	uint32_t size;
	CVoxelChunk* next;
};

static_assert(sizeof(CPointChunk) == sizeof(CVoxelChunk), "Point and voxel chunks share the chunk pool");
static_assert(alignof(CPointChunk) == alignof(CVoxelChunk), "Point and voxel chunks share the chunk pool");

/// An element of the chunk pool: the memory of either a point chunk or a voxel chunk. A chunk is
/// allocated as one kind (`CMemoryAllocator::newPointChunk' / `newVoxelChunk'), only used as that
/// kind, and given back to the pool (`delChunk'); the pool is shared, so the memory goes to whichever
/// kind needs it (no separate sizing of points and voxels)
struct alignas(alignof(CPointChunk)) CChunk {
	unsigned char storage[sizeof(CPointChunk)];
};
static_assert(sizeof(CChunk) == sizeof(CPointChunk), "A pool element holds exactly one chunk");

__device__ __forceinline__ CPointChunk* asPointChunk(CChunk* chunk){ return reinterpret_cast<CPointChunk*>(chunk); }
__device__ __forceinline__ CVoxelChunk* asVoxelChunk(CChunk* chunk){ return reinterpret_cast<CVoxelChunk*>(chunk); }
__device__ __forceinline__ CChunk* asPoolChunk(CPointChunk* chunk){ return reinterpret_cast<CChunk*>(chunk); }
__device__ __forceinline__ CChunk* asPoolChunk(CVoxelChunk* chunk){ return reinterpret_cast<CChunk*>(chunk); }

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
	CFlagToStore,

	// Rendering properties
	CFlagHasEnoughPoints,
	CFlagHasEnoughVoxels,
	CFlagIsLarge,
	CFlagIsCut,

	// Pads to be replaced on need
	CFlagPad14,
	CFlagPad15,
	CFlagPad16,
	CFlagPad17,
	CFlagPad18,
	CFlagPad19,
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

enum CGlobalNodeFlagType {
	CFlagToLoad,
	CFlagWillBeInUpdatesCache,
	CFlagIsInUpdatesCache,
	CFlagWillBeStored,
	CFlagHasBeenStored,
	CFlagIsVisibleHost,
	CFlagIsLargeHost,
	CFlagIsCutHost,
	CFlagWasVisibleHost,

	// Cut of the host rendering buffers currently drawn (copied from `CFlagIsLargeHost' /
	// `CFlagIsCutHost' when they are published): the device draws its resident and evicted nodes
	// with the same cut, so that both renderers always agree (see `kernel_publish_host_cut')
	CFlagIsLargePublished,
	CFlagIsCutPublished,

	// Pads to be replaced on need
	// Sticky: some voxels of the node were dropped when it was stored (the host keeps at most
	// `maxPointsPerLeaf' voxels per node, the device sends up to `maxNbVoxelsChunksPerExchangedNode'
	// chunks): its voxels are an incomplete representation, its children (which hold all the data
	// behind its voxels) are drawn instead (see `isLargeForRendering' in render.cu)
	CFlagHasIncompleteVoxels,
	// Visibility of the published host rendering buffers (copied from `CFlagIsVisibleHost' with the
	// cut, see `kernel_publish_host_cut'): the device draws the out-of-view fallback with it
	CFlagIsVisiblePublished,
	CGlobalFlagPad13,
	CGlobalFlagPad14,
	CGlobalFlagPad15,
	CGlobalFlagPad16,
	CGlobalFlagPad17,
	CGlobalFlagPad18,
	CGlobalFlagPad19,
	CGlobalFlagPad20,
	CGlobalFlagPad21,
	CGlobalFlagPad22,
	CGlobalFlagPad23,
	CGlobalFlagPad24,
	CGlobalFlagPad25,
	CGlobalFlagPad26,
	CGlobalFlagPad27,
	CGlobalFlagPad28,
	CGlobalFlagPad29,
	CGlobalFlagPad30,
	CGlobalFlagPad31,
};

struct COctreeNode {
	COctreeNode* children[8] = {nullptr};
	CPointChunk* points = nullptr;
	CVoxelChunk* voxels = nullptr;
	COccupancyGrid* occupancy = nullptr;
	CIdAABB aabb_index = CINVALID_ID;

	uint32_t points_counter = 0;
	uint32_t voxels_counter = 0;

	uint32_t points_stored = 0;
	uint32_t voxels_stored = 0;

	uint32_t points_last_stored = 0;
	uint32_t voxels_last_stored = 0;

	uint32_t children_ids = 0;
	uint32_t cur_id = 0;

	uint32_t flags = 0;
	uint8_t level = 0;

	__device__ __forceinline__ COctreeNode(): points(nullptr), voxels(nullptr), occupancy(nullptr),
		aabb_index(CINVALID_ID), points_counter(0), voxels_counter(0),
		points_stored(0), voxels_stored(0), children_ids(0), 
		cur_id(0), flags(0), level(0)
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

	__device__ __forceinline__ void flagAsUpdated() {
		flags |= (0x01 << CFlagIsUpdated);
	}
	__device__ __forceinline__ bool isUpdated() const {
		return flags & (0x01 << CFlagIsUpdated);
	}
	__device__ __forceinline__ void flagAsHavingEnoughPoints() {
		flags |= (0x01 << CFlagHasEnoughPoints);
	}
	__device__ __forceinline__ bool hasEnoughPoints() const {
		return flags & (0x01 << CFlagHasEnoughPoints);
	}
	__device__ __forceinline__ void flagAsHavingEnoughVoxels() {
		flags |= (0x01 << CFlagHasEnoughVoxels);
	}
	__device__ __forceinline__ bool hasEnoughVoxels() const {
		return flags & (0x01 << CFlagHasEnoughVoxels);
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

	__device__ __forceinline__ void resetCounterFlag() {
		uint32_t mask = getCounterFlagMask();
		flags &= ~mask;
	}

	__device__ __forceinline__ uint32_t getCounterFlag() const {
		uint32_t mask = getCounterFlagMask();
		return (flags & mask) >> CFlagCounter0;
	}
	
	__device__ __forceinline__ void setCounterFlag(uint8_t new_value) {
		uint32_t new_counter = (new_value) << CFlagCounter0;
		resetCounterFlag();
		flags |= new_counter;
	}

	// Return the old counter
	__device__ __forceinline__ uint32_t increaseCounterFlag() {
		uint32_t old_counter = getCounterFlag();
		if(old_counter == UINT8_MAX){
			printf("ERROR: Reached max of counter flag\n");
			return old_counter;
		}
		uint32_t new_counter = (old_counter + 1) << CFlagCounter0;
		resetCounterFlag();
		flags |= new_counter;
		return old_counter;
	}
};





struct CAllocationPlan {
	CPointChunk* points_attach;   // chunk to append new points chunks to (nullptr if none needed)
    uint32_t points_first_new;
    uint32_t points_nb_new;
    uint32_t points_last_size;
	uint32_t points_existing_count;

    CVoxelChunk* voxels_attach;   // chunk to append new voxels chunks to (nullptr if none needed)
    uint32_t voxels_first_new;
    uint32_t voxels_nb_new;
    uint32_t voxels_last_size;
	uint32_t voxels_existing_count;
};





/// A node evicted from the device and kept drawable from a device copy of its in-memory data until
/// the published host rendering buffers show it (see `CGlobalVariables::retainedEvictedNodes')
struct CEvictedRetainEntry {
	CIdAABB id = CINVALID_ID;
	uint32_t generation = 0;      // eviction pass that evicted the node
	uint32_t points_offset = 0;   // in `retainedEvictedPoints'
	uint32_t nb_points = 0;
	uint32_t voxels_offset = 0;   // in `retainedEvictedVoxels'
	uint32_t nb_voxels = 0;
};

/// A run of the rendered voxels belonging to one node (see `CGlobalVariables::renderedVoxelSegments')
struct CVoxelSegment {
	uint32_t first = 0;          // index of its first voxel in the rendered voxels
	CIdAABB id = CINVALID_ID;    // its node
};
/// Voxels per entry of `CGlobalVariables::renderedVoxelTiles*'
constexpr uint32_t RENDERED_VOXELS_TILE_SIZE = 1024;

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
		uint32_t points_stored = 0;
		uint32_t voxels_stored = 0;
    };
    /// The list of all AABBs created during runtime
	uint32_t totalNbNodes = 0;
    uint32_t maxNbConcurrentNodes = 0;
    Relationship* relationshipMap = nullptr;
	uint32_t* nodesFlags = nullptr;

	/// The main octree
	bool isInitialised = false;
	bool isUpdating = false;
	COctreeNode* mainOctree = nullptr;

	/// The resident nodes (packed): indexed by `COctreeNode::cur_id', at most `maxNbResidentNodes' (the
	/// nodes come from the node pool: `NB_ALLOCABLE_NODES')
	uint32_t maxNbResidentNodes = 0;
	COctreeNode** packedNodes = nullptr;

	/// The buffer of nodes for rendering
	uint32_t octreeDepth = 0;

	/// Scratch buffers: `temporaryIdBuffer' is indexed by node id (`maxNbConcurrentNodes'),
	/// `temporaryIdBuffer2' by resident node (`maxNbResidentNodes'), `temporaryNodeBuffer' by
	/// exchanged node (`maxNbNodesExchanged')
	CIdAABB* temporaryIdBuffer = nullptr;
	CIdAABB* temporaryIdBuffer2 = nullptr;
    COctreeNode** temporaryNodeBuffer = nullptr;
	uint32_t nbNodesExchangedBeforeLoadComplete = 0;

	uint32_t chunksAllocatorCounter = 0;
	/// Point and voxel chunks of the allocation plans (pool elements: see `asPointChunk' / `asVoxelChunk')
	CChunk** allocatedChunks = nullptr;                // `NB_ALLOCABLE_CHUNKS'
	CAllocationPlan* allocationPlans = nullptr;        // per resident node
	
	/// Leaf reached by every batch point / spilled point in the count, as a resident node index
	/// (`packedNodes'; stable within an update, the nodes are only repacked by the cache update)
	uint32_t* memoizedBatchPointsNodes = nullptr; 
	uint32_t* memoizedSpilledPointsNodes = nullptr; 


	uint32_t nbOrderedNodes = 0;


    ///////////////////////////////////////////////////////////////////////
    ////////////////////////// EXCHANGEABLE DATA //////////////////////////
    ///////////////////////////////////////////////////////////////////////

	/// TODO: use cuda unified memory for them ??
	/// Data exchanged from the host
	uint32_t nbNodesExchanged = 0;
	uint32_t maxNbNodesExchanged = 0;
	uint32_t maxNbVoxelsChunksPerExchangedNode = 0;
	CIdAABB* exchangedAABBIndices = nullptr;
	CIdAABB* exchangedAABBParentsIndices = nullptr;
	CAABB* exchangedAABBs = nullptr;
	uint32_t* exchangedChildrenIds = nullptr;
	uint32_t* exchangedPointsCounters = nullptr;
	uint32_t* exchangedVoxelsCounters = nullptr;
	CPoint** exchangedPoints = nullptr;
	/// Voxels of the evicted nodes (device -> host)
	CVoxel** exchangedVoxels = nullptr;
	/// Voxels of the loaded nodes (host -> device): only their cells are used, to rebuild the
	/// occupancy grids (`kernel_fill_new_grids'); the voxels themselves stay on the host. Separate from
	/// `exchangedVoxels', which keeps the last evicted pass drawable (see `kernel_draw_evicted_nodes')
	CVoxel** exchangedLoadedVoxels = nullptr;

	bool isDoneLoading = true;
	bool isDoneStoring = true;
	bool isDoneIterating = true;

	/// Occupancy grids to clear / refill (at most `maxNbGridsToInit': the grid pool, `NB_ALLOCABLE_GRIDS')
	uint32_t maxNbGridsToInit = 0;
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



	///////////////////////////////////////////////////////////////////////
    ////////////////////////////// RENDERING //////////////////////////////
    ///////////////////////////////////////////////////////////////////////
	uint32_t maxNbRenderedPoints = 0;
	uint32_t maxNbRenderedVoxels = 0;
	uint32_t nbRenderedPoints = 0;
	uint32_t nbRenderedPoints2 = 0;
	CPoint* renderedPoints = nullptr;
	CPoint* renderedPoints2 = nullptr;
	uint32_t nbRenderedVoxels = 0;
	uint32_t nbRenderedVoxels2 = 0;
	CVoxel* renderedVoxels = nullptr;
	CVoxel* renderedVoxels2 = nullptr;
	/// Node of every rendered voxel: a voxel only holds its cell, its position needs the bounding box
	/// of its node. The host packs the voxels of a node contiguously, so the map is a list of segments
	/// (first voxel, node), sorted by first voxel, one per node drawn: `renderedVoxelSegments*'.
	/// To find the segment of voxel i without searching the whole list, `renderedVoxelTiles*[t]' is
	/// the segment holding voxel t * RENDERED_VOXELS_TILE_SIZE (see `findRenderedVoxelNode')
	/// Same double buffering as the voxels (`isUsingSecondRenderingBuffer')
	uint32_t maxNbRenderedVoxelSegments = 0;     // min(maxNbConcurrentNodes, maxNbRenderedVoxels)
	uint32_t nbRenderedVoxelSegments = 0;
	uint32_t nbRenderedVoxelSegments2 = 0;
	CVoxelSegment* renderedVoxelSegments = nullptr;
	CVoxelSegment* renderedVoxelSegments2 = nullptr;
	uint32_t* renderedVoxelTiles = nullptr;      // ceil(maxNbRenderedVoxels / RENDERED_VOXELS_TILE_SIZE)
	uint32_t* renderedVoxelTiles2 = nullptr;

	uint32_t totalNbNodesForVisibility = 0;
	uint32_t nbNodesExchangedVisPoints = 0;
	uint32_t nbNodesExchangedVisVoxels = 0;
	bool isUsingSecondRenderingBuffer = false;
	uint32_t curNbVisNode = 0;

	/// Last evicted batch, still drawn from the exchange buffers (`exchangedPoints' / `exchangedVoxels',
	/// only overwritten by the next eviction pass) until the published host rendering buffers show it
	uint32_t nbEvictedNodes = 0;               // valid slots of the last eviction pass
	CIdAABB* evictedNodesIds = nullptr;        // node id of each exchange slot
	uint32_t* evictedNodesNbPoints = nullptr;  // points in `exchangedPoints[slot]'
	uint32_t* evictedNodesNbVoxels = nullptr;  // voxels in `exchangedVoxels[slot]'
	/// Generation of the last eviction pass (written by the host) and latest generation the published
	/// host rendering buffers show (written by the host when it publishes them): the evicted batch is
	/// drawn while `evictionGeneration > coveredEvictionGeneration'
	uint32_t evictionGeneration = 0;
	uint32_t coveredEvictionGeneration = 0;

	/// Older evicted batches: the exchange buffers only hold the last eviction pass, so every pass also
	/// copies the evicted nodes the device was drawing (large or cut in the published cut, or created
	/// since it) into these buffers. They are drawn while their generation is not covered, so a new
	/// eviction pass does not have to wait for the previous one to be covered.
	/// Append-only, emptied when every generation is covered (`kernel_reset_evicted_retain'). When full
	/// (`retainOverflowGeneration'), the host makes the next pass wait for the coverage instead
	uint32_t maxNbRetainedEvictedNodes = 0;
	uint32_t maxNbRetainedEvictedPoints = 0;
	uint32_t maxNbRetainedEvictedVoxels = 0;
	uint32_t nbRetainedEvictedNodes = 0;     // may exceed the capacity (overflow): min() when reading
	uint32_t nbRetainedEvictedPoints = 0;    // idem
	uint32_t nbRetainedEvictedVoxels = 0;    // idem
	uint32_t retainOverflowGeneration = 0;   // latest generation that did not fit entirely, 0 if none
	CEvictedRetainEntry* retainedEvictedNodes = nullptr;
	CPoint* retainedEvictedPoints = nullptr;
	CVoxel* retainedEvictedVoxels = nullptr;

	/// Number of nodes covered by the published cut (`CFlagIsLargePublished' / `CFlagIsCutPublished'):
	/// the nodes created after it are only on the device and keep the live cut (current camera)
	uint32_t publishedNbNodes = 0;
	
	CIdAABB* exchangedAABBIndicesVisPoints = nullptr;
	CIdAABB* exchangedAABBIndicesVisPointsTmp = nullptr;
	float* exchangedAABBIndicesVisPointsScreenSpaceSize = nullptr;
	CIdAABB* exchangedAABBIndicesVisVoxels = nullptr;
	CIdAABB* exchangedAABBIndicesVisVoxelsTmp = nullptr;
	float* exchangedAABBIndicesVisVoxelsScreenSpaceSize = nullptr;

	/// Out-of-view fallback ("Vox"): a large visible node whose children are not all visible draws,
	/// instead of its non-visible children, its own voxels lying in their octants (see part 3 in
	/// render.cu). The cut never reaches these children, so nothing else draws their region.
	/// Child mask of every node (bit i: `children[i]' exists and is not visible), 0 for nodes not in Vox
	uint8_t* visVoxChildMask = nullptr;
	/// Child mask of every entry of `exchangedAABBIndicesVisVoxels' (gathered after the sort,
	/// 0: every voxel of the node is drawn, i.e. a cut node)
	uint8_t* exchangedVisVoxelsChildMasks = nullptr;



    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////// LRU CACHES //////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    uint32_t updatesCacheSize = 0;
    CIdAABB* updatesCache = nullptr;
	


    ///////////////////////////////////////////////////////////////////////
    ////////////////////////// TEMPORARY BUFFERS //////////////////////////
    ///////////////////////////////////////////////////////////////////////
    /// The list of spilled points
    uint32_t nbSpilledPoints = 0;
    uint32_t maxNbSpilledPoints = 0;
    CPoint* spilledPoints = nullptr;
    /// Nodes spilling in a count iteration (at most `maxNbSpillingNodes': resident nodes), and their
    /// point chunks (at most `maxNbSpillingChunks': every chunk is full except the last one of each
    /// node, so maxNbSpilledPoints / NB_POINTS_PER_CHUNK + maxNbSpillingNodes)
    uint32_t nbSpillingNodes = 0;
    uint32_t maxNbSpillingNodes = 0;
	uint32_t nbSpillingChunks = 0;
	uint32_t maxNbSpillingChunks = 0;
    COctreeNode** spillingNodes = nullptr;
	uint32_t* spilledChunksCounter = nullptr;
	CPointChunk** spillingChunks = nullptr;

    /// The backlog buffer for new voxels
    uint32_t nbBacklogVoxels = 0;
    uint32_t maxNbBacklogVoxels = 0;
    CVoxel* backlogVoxels = nullptr;
    uint32_t* backlogVoxelsNodes = nullptr;   // resident node index (`packedNodes') of each voxel

	uint32_t maxPointsPerLeaf = 0;




    ///////////////////////////////////////////////////////////////////////
    ///////////////////////// UI DISPLAYED VALUES /////////////////////////
    ///////////////////////////////////////////////////////////////////////

	// Point and voxel counters: 64 bits (billions of points overflow 32 bits)
	uint64_t nbTotalPoints = 0;
	uint64_t nbTotalVoxels = 0;
	uint64_t currentNbPoints = 0;
	uint64_t currentNbVoxels = 0;
	uint64_t nbNewPointsThisUpdate = 0;
	uint64_t nbNewVoxelsThisUpdate = 0;

	uint32_t nbTotalUpdates = 0;
	uint32_t curNbNodes = 0;

	uint32_t currentNbChunks = 0;
	uint32_t currentNbGrids = 0;

    uint32_t nbTotalNewNodes = 0;
    uint32_t nbTotalNewGrids = 0;
    uint32_t nbTotalNewChunks = 0;
	uint32_t nbTotalDeletedNodes = 0;
    uint32_t nbTotalDeletedGrids = 0;
    uint32_t nbTotalDeletedChunks = 0;
	uint32_t nbTotalLoadedNodes = 0;
    uint32_t nbTotalSplitNodes = 0;
    uint32_t nbTotalStoredNodes = 0;

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
	__device__ __forceinline__ bool getFlag(const CIdAABB& aabb_index, const CGlobalNodeFlagType& flag) const {
		return nodesFlags[aabb_index] & (0x01 << flag);
	}
	__device__ __forceinline__ void setFlags(const CIdAABB& aabb_index, uint32_t flags){
		nodesFlags[aabb_index] = flags;
	}
	__device__ __forceinline__ void setFlagsSync(const CIdAABB& aabb_index, uint32_t flags){
		__nv_atomic_and(&nodesFlags[aabb_index], flags, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}
	__device__ __forceinline__ void setFlag(const CIdAABB& aabb_index, const CGlobalNodeFlagType& flag){
		nodesFlags[aabb_index] |= (0x01 << flag);
	}
	__device__ __forceinline__ void setFlagSync(const CIdAABB& aabb_index, const CGlobalNodeFlagType& flag){
		__nv_atomic_or(&nodesFlags[aabb_index], (0x01 << flag), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}
	__device__ __forceinline__ uint32_t fetchSetFlagSync(const CIdAABB& aabb_index, const CGlobalNodeFlagType& flag){
		return __nv_atomic_fetch_or(&nodesFlags[aabb_index], (0x01 << flag), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}
	__device__ __forceinline__ void resetFlagsSync(const CIdAABB& aabb_index){
		__nv_atomic_and(&nodesFlags[aabb_index], 0, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}
	__device__ __forceinline__ void resetFlags(const CIdAABB& aabb_index){
		nodesFlags[aabb_index] = 0;
	}
	__device__ __forceinline__ void unsetFlagSync(const CIdAABB& aabb_index, const CGlobalNodeFlagType& flag){
		__nv_atomic_and(&nodesFlags[aabb_index], ~(1u << flag), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}
	__device__ __forceinline__ void unsetFlag(const CIdAABB& aabb_index, const CGlobalNodeFlagType& flag){
		nodesFlags[aabb_index] &= ~(1u << flag);
	}
	__device__ __forceinline__ uint32_t fetchUnsetFlagSync(const CIdAABB& aabb_index, const CGlobalNodeFlagType& flag){
		return __nv_atomic_fetch_and(&nodesFlags[aabb_index], ~(1u << flag), __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
	}



	__device__ __forceinline__ bool isInUpdatesCache(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagIsInUpdatesCache);
	}
	__device__ __forceinline__ bool willBeInUpdatesCache(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagWillBeInUpdatesCache);
	}
	__device__ __forceinline__ bool willbeStored(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagWillBeStored);
	}
	__device__ __forceinline__ bool hasBeenStored(const CIdAABB& aabb_index) const {
		return getFlag(aabb_index, CFlagHasBeenStored);
	}

#endif // __CUDACC__
};






enum SentBatchState {
	BatchToHandle,
	BatchInUse,
	BatchHandled,
};


struct CRenderingSettings {
	int32_t debug_lod_to_render = -1;
	bool use_voxels_debug_color = false;
	uint32_t min_pixel_span = 0;
	uint32_t voxels_nb_points_per_axis = 0;
	bool use_multiscale = false;
	static constexpr uint32_t NB_PYRAMID_LEVELS = 4;
	uint32_t draw_every_x_points = 1;
};


struct CRenderTarget{
	// Per-level framebuffers (finest to coarsest)
    uint64_t* framebuffers[CRenderingSettings::NB_PYRAMID_LEVELS];
    uint64_t* colorbuffers[CRenderingSettings::NB_PYRAMID_LEVELS];
    uint32_t  width;
    uint32_t  height;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec3 camera_pos;
};