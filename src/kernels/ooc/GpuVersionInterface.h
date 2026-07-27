#pragma once

#include <cmath>
#include "../../ooc_structures/settings.h"

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
};

struct CPoint {
	glm::vec3 position;
	uint32_t color;

	__device__ __forceinline__ void setColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xFFu){
		color = (uint32_t)r
			| ((uint32_t)g << 8)
			| ((uint32_t)b << 16)
			| ((uint32_t)a << 24);
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

	__device__ __forceinline__ bool isCellOcupied(const GridIndex& index) const {
		return (values[index.word] & (1u << index.bit)) != 0;
	}

	__device__ __forceinline__ void markCellAsFilled(const GridIndex& index){
		values[index.word] |= (1u << index.bit);
	}

	__device__ __forceinline__ static glm::vec3 getCellCentroid(const CAABB& aabb, const GridIndex& index){
		glm::vec3 world_grid_size = aabb.getSize() / float(OocSimLodSettings::GRID_SIZE_PER_DIMENSION);
		return aabb.mins + world_grid_size * glm::vec3(index.grid.x, index.grid.y, index.grid.z) + 0.5f * world_grid_size;
	}
};

struct CChunk{
	CPoint points[OocSimLodSettings::NB_POINTS_PER_CHUNK];
	uint32_t size = 0;
	CChunk* next = nullptr;
};

struct COctreeNode {
	COctreeNode* children[8] = {nullptr};
	CChunk* points = nullptr;
	CChunk* voxels = nullptr;
	COccupancyGrid* occupancy = nullptr;
	uint32_t aabb_index = UINT32_MAX;

	uint32_t counter = 0;
	uint8_t children_ids = 0b00000000;
	uint8_t children_visibility = 0b00000000;
	uint8_t level = 0;

	bool updated = false;
	bool is_large = false;
	bool is_visible = false;
	bool is_cut = false;
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
};


typedef uint32_t CIdAABB;
constexpr CIdAABB CINVALID_ID = UINT32_MAX;

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
	uint32_t nb_nodes = 0;
	COctreeNode** nodes = nullptr;



    ///////////////////////////////////////////////////////////////////////
    ////////////////////////// EXCHANGEABLE DATA //////////////////////////
    ///////////////////////////////////////////////////////////////////////
    /// The buffer of nodes to load from disk
    uint32_t nbNodesToLoad = 0;
    uint32_t maxNbNodesToLoad = 0;
    CIdAABB* nodesToLoadBuffer = nullptr;

    /// The buffer of nodes to store to disk
    uint32_t nbNodesToStore = 0;
    uint32_t maxNbNodesToStore = 0;
    COctreeNode** nodesToStoreBuffer = nullptr;

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
	uint32_t maxNbResidualPoints = 0;
	CPoint* residualPoints = nullptr;



    ///////////////////////////////////////////////////////////////////////
    ///////////////////////////// LRU CACHES //////////////////////////////
    ///////////////////////////////////////////////////////////////////////
    uint32_t updatesCacheSize = 0;
    CIdAABB* updatesCache = nullptr;
    uint32_t visibilityCacheSize = 0;
    CIdAABB* visibilityCache = nullptr;
    


    ///////////////////////////////////////////////////////////////////////
    ////////////////////////// TEMPORARY BUFFERS //////////////////////////
    ///////////////////////////////////////////////////////////////////////
    /// The list of spilled points
    uint32_t nbSpilledPoints = 0;
    uint32_t maxNbSpilledPoints = 0;
    CPoint* spilledPoints = nullptr;
    COctreeNode** spillingNodes = nullptr;

    /// The backlog buffer for new voxels
    uint32_t nbBacklogVoxels = 0;
    uint32_t maxNbBacklogVoxels = 0;
    CPoint* backlogVoxels = nullptr;
    COctreeNode** backlogVoxelsNodes = nullptr;
};