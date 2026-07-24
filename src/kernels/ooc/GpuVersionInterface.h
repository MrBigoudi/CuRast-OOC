#pragma once

#include <cmath>
#include "../../ooc_structures/settings.h"

struct CAABB {
	glm::vec3 mins = {INFINITY, INFINITY, INFINITY};
	glm::vec3 maxs = {-INFINITY, -INFINITY, -INFINITY};

	inline glm::vec3 getSize() const {return maxs - mins;}
};

struct CPoint {
	glm::vec3 position;
	uint32_t color;
};

struct COccupancyGrid {
	uint32_t values[OocSimLodSettings::GRID_SIZE / 32] = {0};
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
	uint64_t* framebuffer;
	uint64_t* colorbuffer;
	int width;
	int height;
	glm::mat4 view;
	glm::mat4 viewI;
	glm::mat4 proj;
	glm::vec3 cameraPos;
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
	uint32_t nbNodes = 0;
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