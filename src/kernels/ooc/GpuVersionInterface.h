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

	~CChunk(){
		size = 0;
		next = nullptr;
	}
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

	uint32_t children_ids = 0b00000000;

	// TODO: pack all of the following in a single uint32_t
	// uint8_t children_ids = 0b00000000;
	uint8_t children_visibility = 0b00000000;
	uint8_t level = 0;

	bool updated = false;
	bool is_large = false;
	bool is_visible = false;
	bool is_cut = false;

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
		children_visibility = 0b00000000;
		level = 0;
		updated = false;
		is_large = false;
		is_visible = false;
		is_cut = false;
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
	uint64_t* colorbuffer;
	int width;
	int height;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec3 camera_pos;
};


enum CNodeFlagType {
	CFlagToLoad,
	CFlagToStore,
	CFlagIsSpilling,

	CFlagIsOnUpdatesCache,

	// Pads to be replaced on need
	CFlagPad4,
	CFlagPad5,
	CFlagPad6,
	CFlagPad7,
	CFlagPad8,
	CFlagPad9,
	CFlagPad10,
	CFlagPad11,
	CFlagPad12,
	CFlagPad13,
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

/// The LRU caches for the nodes
/// https://www.geeksforgeeks.org/dsa/lru-cache-implementation-using-double-linked-lists/
struct CLRUCache {
	const uint32_t CACHE_SIZE;
	CDoubleLinkedList<CIdAABB> cache = {};
	CHashMap<CIdAABB, CDoubleLinkedList<CIdAABB>::Iterator*> cache_map = {};

	__device__ __forceinline__ CLRUCache(uint32_t cache_size) : CACHE_SIZE(cache_size){
		cache.init();
		cache_map.init(cache_size);
	}

	/// Add a node to the cache and return the id of a node if it has been removed from the cache
	__device__ __forceinline__ CIdAABB add(const CIdAABB& aabb_index){
		CDoubleLinkedList<CIdAABB>::Iterator** it = cache_map.find(aabb_index);

		// If the AABB was already in cache, remove its old version from the list
		if(it){
			cache.moveBegin(*it);
			return CINVALID_ID;
		}

		// If the cache is full, remove the last node
		if(cache_map.size >= CACHE_SIZE){
			CDoubleLinkedList<CIdAABB>::Iterator* end = cache.end();
			CIdAABB old_aabb = end->value;
			end->value = aabb_index;
			cache.moveBegin(end);
			// cache_map.erase(old_aabb);
			cache_map.partialErase(old_aabb);
			cache_map[aabb_index] = end;
			return old_aabb;
		}

		// Insert the new node at the front of the list
		cache.pushFront(aabb_index);
		cache_map[aabb_index] = cache.begin();
		return CINVALID_ID;
	}

	/// Check if a node is already in cache
	__device__ __forceinline__ bool contains(const CIdAABB& aabb_index) {
		return cache_map.contains(aabb_index);
	}

	/// Returns the number of occupied cell in the cache
	__device__ __forceinline__ uint32_t getSize() const {
		return cache_map.size;
	}
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
        CIdAABB& operator[](uint32_t child_index){return children[child_index];}
        const CIdAABB operator[](uint32_t child_index) const {
            if(child_index >= 8){return CINVALID_ID;}
            return children[child_index];
        }
    };
    /// The list of all AABBs created during runtime
    uint32_t nbAABBs = 0;
    uint32_t maxNbAABBs = 0;
    Relationship* relationshipMap = nullptr;
    CAABB* allAABBs = nullptr;
	/// The main octree
	COctreeNode* mainOctree = nullptr;
	uint32_t mainOctreeMaxLevel = 0;

	/// The buffer of nodes for rendering and looping over
	COctreeNode** nodes = nullptr;
	uint32_t curNbNodes = 0;
	COctreeNode** packedNodes = nullptr;



    ///////////////////////////////////////////////////////////////////////
    ////////////////////////// EXCHANGEABLE DATA //////////////////////////
    ///////////////////////////////////////////////////////////////////////
	uint32_t* nodesFlags = nullptr;

	/// TODO: use cuda unified memory for them ??
	/// Data received from the host
	uint32_t nbNodesReceived = 0;
	uint32_t maxNbNodesReceived = 0;
	uint32_t maxNbPointsChunksPerReceivedNode = 0;
	uint32_t maxNbVoxelsChunksPerReceivedNode = 256; // TODO: find a better value
	CIdAABB* receivedAABBIndices = nullptr;
	uint32_t* receivedChildrenIds = nullptr;
	uint32_t* receivedPointsCounters = nullptr;
	uint32_t* receivedVoxelsCounters = nullptr;
	CPoint** receivedPoints = nullptr;
	CPoint** receivedVoxels = nullptr;
	void* receivedPointsPointers = nullptr; // Just needed for host side
	void* receivedVoxelsPointers = nullptr; // Just needed for host side

    /// The buffer of nodes to load from disk
    uint32_t nbNodesToLoad = 0;
    uint32_t maxNbNodesToLoad = 0;
    CIdAABB* nodesToLoadBuffer = nullptr;

    /// The buffer of nodes to store to disk
    uint32_t nbNodesToStore = 0;

    /// A mask to know which batches have been handled
    uint32_t maxNbBatches = 0;
    uint32_t* batchesAddedMask = nullptr;
	/// The batches to add to the scene
	CPoint** batchesToAddPoints = nullptr;
	uint32_t* batchesToAddCounts = nullptr;
	uint32_t* batchesToAddBottomUpCounts = nullptr;
	void* batchesToAddPointsPointers = nullptr; // Just needed for host side

	/// The points that couldn't be handled yet
	uint32_t nbResidualPoints = 0;
	uint32_t maxNbResidualPoints =  1'000'000; // TODO: find a better value
	CPoint* residualPoints = nullptr;



    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////// LRU CACHES //////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    uint32_t updatesCacheSize = 0;
    uint32_t nbEntryInUpdatesCache = 0;
    CLRUCache* updatesCache = nullptr;
    uint32_t visibilityCacheSize = 0;
    uint32_t nbEntryInVisibilityCache = 0;
    CLRUCache* visibilityCache = nullptr;
    


    ///////////////////////////////////////////////////////////////////////
    ////////////////////////// TEMPORARY BUFFERS //////////////////////////
    ///////////////////////////////////////////////////////////////////////
    /// The list of spilled points
    bool hasSpillingNodes = false;
    uint32_t nbSpilledPoints = 0;
    uint32_t maxNbSpilledPoints = 0;
    CPoint* spilledPoints = nullptr;
    uint32_t nbSpillingNodes = 0;
    COctreeNode** spillingNodes = nullptr;

    /// The backlog buffer for new voxels
    uint32_t nbBacklogVoxels = 0;
    uint32_t maxNbBacklogVoxels = 0;
    CPoint* backlogVoxels = nullptr;
    COctreeNode** backlogVoxelsNodes = nullptr;

	uint32_t maxPointsPerLeaf = 0;
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
	uint32_t nb_blocks_per_node = 0;
	int32_t debug_lod_to_render = 0;
	bool use_voxels_debug_color = false;
	uint32_t min_pixel_span = 0;
	uint32_t voxels_nb_points_per_axis = 0;
};