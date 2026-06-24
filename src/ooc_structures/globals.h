#pragma once

#include "CuRast.h"
#include "laszip/laszip_api.h"
#include <semaphore>

#include <unordered_set>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/hash.hpp"


///////////////////////////////////////////////////////////////////////////////
/////////////////////////// GLOBAL ENUM DECLARATION ///////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// The position of a child node
enum NodePosition {
	FrontTopLeft,
	FrontTopRight,
	FrontBottomLeft,
	FrontBottomRight,
	BackTopLeft,
	BackTopRight,
	BackBottomLeft,
	BackBottomRight,
};
void updateNodePosition(NodePosition& position);

/// The state of a batch
enum BatchState {
	Empty,
	ToLoad,
	Loaded,
	Inserted,
	ToRemove
};


///////////////////////////////////////////////////////////////////////////////
////////////////////////////// GLOBAL CONSTANTS ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// The maximum number of batches that should be loaded from disk at once
constexpr uint8_t MAX_BATCHES_PER_LOAD = 100;
// constexpr uint8_t MAX_BATCHES_PER_LOAD = 1000;
/// The maximum number of batches that should be loaded to the GPU at once
constexpr uint8_t MAX_BATCHES_PER_GPU_LOAD = 50;
// constexpr uint8_t MAX_BATCHES_PER_GPU_LOAD = 200;

constexpr uint32_t SEND_DATA_EVERY_X_FRAMES = 200;
// constexpr uint32_t SEND_DATA_EVERY_X_FRAMES = 1;
extern uint32_t elapsedFrames;

extern uint64_t NB_POINTS;
extern bool MAIN_LOOP_IS_TERMINATING;
extern std::mutex mainLoopIsTerminatingMtx;

// /// The maximum number of batches that should be used per octree update
// constexpr uint8_t MAX_BATCHES_PER_UPDATE = 1;
// /// The maximum number of points in a batch
// constexpr uint64_t MAX_BATCH_SIZE = 1'000'000;

/// The maximum number of points in a leaf node
constexpr uint32_t MAX_POINTS_PER_LEAF = 50'000;
// constexpr uint32_t MAX_POINTS_PER_LEAF = 250'000;
/// The number of points in a chunk
constexpr uint32_t POINTS_PER_CHUNK = 1'024;
/// The voxel grid size
constexpr uint32_t GRID_SIZE = 128;
/// The number of cells in each grid
constexpr uint32_t GRID_NUM_CELLS = GRID_SIZE * GRID_SIZE * GRID_SIZE;
/// The first position for a child node on merging
constexpr NodePosition FIRST_NODE_POSITION = FrontTopLeft;
/// The temporary files directory to store nodes in disk
const std::string TEMPORARY_DIRECTORY = format("{}/build/tmp", PROJECT_SOURCE_DIR);

constexpr bool CPU_PARALLELISED = true;
// constexpr bool CPU_PARALLELISED = false;

/// The maximum size for the batches vectors
extern uint32_t BATCHES_QUEUE_SIZE;


///////////////////////////////////////////////////////////////////////////////
//////////////////////// GLOBAL STRUCTURES DECLARATION ////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct AABB;
struct Point;
struct PointBatch;
struct Voxel;
struct OctreeNode;
struct Chunk;
struct OccupancyGrid;


///////////////////////////////////////////////////////////////////////////////
////////////////////////////// GLOBAL STRUCTURES //////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// An axis-aligned bounding box
struct AABB {
	vec3 mins = {INFINITY, INFINITY, INFINITY};
	vec3 maxs = {-INFINITY, -INFINITY, -INFINITY};

	bool contains(const vec3& position) const;
	bool isParentOf(const AABB& aabb) const;
	vec3 getCentroid() const;
	vec3 getSize() const;
	void extend(const NodePosition& position);
	void shrink(const NodePosition& position);
	vec3 getPointNormalizedCoordinates(const vec3& position) const;
	vec3 getPointWorldCoordinates(const vec3& normalized_position) const;
	NodePosition getNextChildIndex(const vec3& position) const;

	bool operator==(const AABB& rhs) const{
		return rhs.mins == mins && rhs.maxs == maxs;
	}
	bool operator==(const AABB& rhs){
		return rhs.mins == mins && rhs.maxs == maxs;
	}
	bool operator==(AABB& rhs) const{
		return rhs.mins == mins && rhs.maxs == maxs;
	}
	bool operator==(AABB& rhs){
		return rhs.mins == mins && rhs.maxs == maxs;
	}

	AABB(){}
	AABB(const AABB& cpy) : mins(cpy.mins), maxs(cpy.maxs){}
	AABB(AABB& cpy) : mins(cpy.mins), maxs(cpy.maxs){}

	struct Hash {
		std::size_t operator()(const AABB& aabb) const {
			vec3 tmp = {0,0,0};
			mat3 matrix = {
				aabb.mins,
				aabb.maxs,
				tmp
			};
			return std::hash<mat3>()(matrix);

			// auto hashCombine = [](std::size_t& seed, std::size_t value){
			// 	seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			// };

			// std::size_t seed = 0;
			// hashCombine(seed, std::hash<float>{}(aabb.mins.x));
			// hashCombine(seed, std::hash<float>{}(aabb.mins.y));
			// hashCombine(seed, std::hash<float>{}(aabb.mins.z));

			// hashCombine(seed, std::hash<float>{}(aabb.maxs.x));
			// hashCombine(seed, std::hash<float>{}(aabb.maxs.y));
			// hashCombine(seed, std::hash<float>{}(aabb.maxs.z));

			// return seed;
		}
	};
};

/// A loaded point
struct Point {
	vec3 position = vec3();
	uint8_t color[4] = {0,0,0,0};

	Point(const Point& cpy) : position(cpy.position),
		color{
			cpy.color[0], cpy.color[1], 
			cpy.color[2], cpy.color[3]
	}{}

	Point(){}
	bool operator==(const Point& rhs) const;
};

/// A set of points read from a las / laz file
struct PointBatch {
	uint64_t first = 0;
	uint64_t count = 0;
	shared_ptr<string> file = nullptr;
	shared_ptr<vector<Point>> points = nullptr;
	shared_ptr<laszip_header> header = nullptr;
	BatchState state = BatchState::Empty;

	// TODO: rethink that
	/// Helpers for CUDA memory transfer
	vector<vec3> getPositions() const;
	vector<uint32_t> getColors() const;
};

/// A voxel in intermediate nodes
struct Voxel {
	/// Position in the voxel grid
	uint8_t position[3] = {0,0,0};
	uint8_t padding;
	uint8_t color[4] = {0,0,0,0};
};

struct OccupancyGrid {
	// gridsize^3 occupancy grid; 1 bit per voxel
	uint32_t values[GRID_NUM_CELLS / 32u] = {0};

	OccupancyGrid(){}
	OccupancyGrid(const OccupancyGrid& cpy){
		for(uint32_t i=0; i<GRID_NUM_CELLS / 32u; i++){
			values[i] = cpy.values[i];
		}
		assert(cpy == *this);
	}
	bool operator==(const OccupancyGrid& rhs) const {
		for(uint32_t i=0; i<GRID_NUM_CELLS / 32u; i++){
			if(values[i] != rhs.values[i]){
				// println("OccupancyGrid::operator==: at i=={}: {} != {}", 
				// 	i, values[i], rhs.values[i]
				// );
				return false;
			}
		}
		return true;
	}

	uint32_t getNbFilledEntries() const {
		uint32_t cpt = 0;
		for(uint32_t i=0; i<GRID_NUM_CELLS / 32u; i++){
			for(uint32_t j=0; j<32; j++){
				if(values[i] & (1u << j)){
					cpt++;
				}
			}
		}
		return cpt;
	}
};

/// A chunk linked list in a node
struct Chunk {
	/// All chunk have the same physical size even if empty
	Point points[POINTS_PER_CHUNK] = {Point()};
	uint32_t size = 0;
	/// For the linked list
	Chunk* next = nullptr;

	bool operator==(const Chunk& rhs) const;

	Chunk(){}
	Chunk(const Chunk& cpy): size(cpy.size) {
		for(uint32_t i=0; i<POINTS_PER_CHUNK; i++){
			points[i] = cpy.points[i];
		}

		if(cpy.next){
			next = new Chunk(*cpy.next);
		}
		assert(cpy == *this);
	}

	~Chunk() {
		Chunk* tmp = next;
		next = nullptr;
		delete(tmp);
	}

};

/// A node in an octree
struct OctreeNode {
	OctreeNode* children[8] = {nullptr};
	Chunk* points = nullptr;
	Chunk* voxels = nullptr;
	OccupancyGrid* occupancy = nullptr;
	AABB* aabb = nullptr;

	uint32_t counter = 0;
	uint8_t children_ids = 0b00000000;

	bool from_split = false;
	bool from_bottom_up = false;
	bool updated = false;
	uint8_t level = 0;
	bool is_large = false;
	bool is_visible = false;
	bool is_cut = false;


	uint32_t getNbPoints() const;
	uint32_t getNbVoxels() const;
	uint32_t getDepth() const;

	void display(uint32_t id = 0, uint32_t level = 0, bool node_only = false) const;

	bool operator==(const OctreeNode& rhs) const;

	~OctreeNode(){
		if(points){
			Chunk* tmp = points;
			points = nullptr;
			delete(tmp);
		}
		if(voxels){
			Chunk* tmp = voxels;
			voxels = nullptr;
			delete(tmp);
		}
		if(occupancy){
			OccupancyGrid* tmp = occupancy;
			occupancy = nullptr;
			delete(tmp);
		}
		if(aabb){
			AABB* tmp = aabb;
			aabb = nullptr;
			delete(aabb);
		}
		for(uint32_t i=0; i<8; i++){
			if(children[i]){
				OctreeNode* tmp = children[i];
				children[i] = nullptr;
				delete(tmp);
			}
		}
	}

	OctreeNode(){}
	OctreeNode(const OctreeNode& cpy) : counter(cpy.counter), children_ids(cpy.children_ids)
		, from_split(cpy.from_split), from_bottom_up(cpy.from_bottom_up)
	{
		aabb = cpy.aabb ? new AABB(*cpy.aabb) : nullptr;
		points = cpy.points ? new Chunk(*cpy.points) : nullptr;
		voxels = cpy.voxels ? new Chunk(*cpy.voxels) : nullptr;
		occupancy = cpy.occupancy ? new OccupancyGrid(*cpy.occupancy) : nullptr;

		for(uint32_t child = 0; child < 8; child++){
			if(cpy.children[child]){
				children[child] = new OctreeNode(*cpy.children[child]);
			}
		}

		assert(cpy == *this);
	}
};



///////////////////////////////////////////////////////////////////////////////
////////////////////////// GLOBAL EXTERNAL VARIABLES //////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// Name of the main octree node in the scene
const std::string simLodOctreeName = std::string("MainOctreeSimLOD");
/// Counter for the number of octree created
extern uint64_t simLodOctreeCounter;
std::string getSimLodOctreeName(bool generate_new_name = false);

/// Variables tracking when the octree can be sent to GPU
extern std::binary_semaphore octreeReadyToBeSent;
extern std::binary_semaphore octreeReadyToBeUpdated;
extern std::binary_semaphore octreeNotBeingSent;
extern std::mutex isUpdatingMtx;

extern uint64_t loadCounter;
extern std::mutex loadCounterMtx;

extern bool lodUpdated;

/// The queue of batches
extern std::deque<std::shared_ptr<PointBatch>> batchesQueue;
extern std::deque<std::mutex> batchesQueueMutexes;
extern std::mutex updateSceneMutex;

/// The main octree
extern std::shared_ptr<OctreeNode> mainOctree;
extern OctreeNode* mainOctreeCpy;

/// The buffer of spilled points
extern std::shared_ptr<vector<Point>> spilledPoints;
/// The buffer of spilling nodes
extern std::shared_ptr<vector<OctreeNode*>> spillingNodes;

/// The backlog buffer for new voxels
extern std::shared_ptr<vector<Point>> backlogVoxels;
/// The backlog buffer for the nodes corresponding to the new voxels
extern std::shared_ptr<vector<OctreeNode*>> backlogVoxelsNodes;


/// The hash map to store timings
struct Timing {
	bool has_started = false;
	string name = "";
	uint32_t level = 0;
	std::chrono::time_point<std::chrono::high_resolution_clock> start = {};
	std::chrono::time_point<std::chrono::high_resolution_clock> stop = {};
	std::chrono::microseconds duration = {};

	Timing(string name, bool start_now = true, uint32_t level = 0)
	 : has_started(start_now), name(name), level(level){
		if(start_now){
			start = std::chrono::high_resolution_clock::now();
		}
	}

	void start_clock(){
		if(has_started){
			println("Can't start a timing twice");
			return;
		}
		start = std::chrono::high_resolution_clock::now();
	}

	void stop_clock(){
		if(!has_started){
			println("Can't stop an unstarted timing");
			return;
		}
		stop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	}
};

/// The timings list
extern vector<std::shared_ptr<Timing>> timingsList;
static std::mutex timingsMtx;
std::shared_ptr<Timing> addTiming(string name, bool start_now = true, uint32_t level = 0);

void displayTimings();
void displayBuffers();




///////////////////////////////////////////////////////////////////////////////
/////////////////////////// LRU CACHING SHENANIGANS ///////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// The size of the LRU cache
// constexpr uint32_t LRU_UPDATES_CACHE_SIZE = 16;
// constexpr uint32_t LRU_UPDATES_CACHE_SIZE = 32;
constexpr uint32_t LRU_UPDATES_CACHE_SIZE = 128;
// constexpr uint32_t LRU_UPDATES_CACHE_SIZE = 256;
// constexpr uint32_t LRU_UPDATES_CACHE_SIZE = 1024;
// constexpr uint32_t LRU_UPDATES_CACHE_SIZE = 4096;

// constexpr uint32_t LRU_VISIBILITY_CACHE_SIZE = 16;
// constexpr uint32_t LRU_VISIBILITY_CACHE_SIZE = 32;
// constexpr uint32_t LRU_VISIBILITY_CACHE_SIZE = 128;
constexpr uint32_t LRU_VISIBILITY_CACHE_SIZE = 512;
// constexpr uint32_t LRU_VISIBILITY_CACHE_SIZE = 1024;
// constexpr uint32_t LRU_VISIBILITY_CACHE_SIZE = 4096;

typedef std::pair<uint64_t, AABB> CacheEntry;

/// The LRU caches for the UPDATED nodes
struct LRUCache {
	static std::mutex stored_set_mtx;
	static std::unordered_set<AABB, AABB::Hash> stored_set;

	static std::mutex test_mtx;

	const uint32_t CACHE_SIZE;
	uint64_t counter = 0;
	std::vector<std::optional<CacheEntry>> cache = {};
	std::unordered_map<AABB, uint32_t, AABB::Hash> cache_map = {};
	std::string name = "";
	std::mutex mtx = {};

	LRUCache(const std::string& name, uint32_t cache_size)
		: name(name), CACHE_SIZE(cache_size){
		cache = std::vector<std::optional<CacheEntry>>(cache_size, nullopt);
	}

	LRUCache(const LRUCache& cpy): LRUCache(cpy.name, cpy.CACHE_SIZE){}


	/// Add a node to the cache and return the id of a node if it has been removed from the cache
	/// The id of a node is it's AABB
	std::optional<AABB> add(const AABB& aabb, bool sync = false);
	/// Gets the cache index of a node
	std::optional<uint32_t> getIndex(const AABB& aabb, bool sync = false);
	/// Check if a node is already in cache
	bool contains(const AABB& aabb, bool sync = false);
	/// Display the LRU cache
	void display(bool sync = false);
	/// Returns the number of occupied cell in the cache
	uint32_t getSize() const;
	bool sanityCheck(const OctreeNode* root_node);

	/// Check if a node has been stored
	static bool hasBeenStored(const AABB& aabb);
	/// Mark a node as stored
	static void mark(const AABB& aabb);
	/// Unmark a node as stored
	static void unmark(const AABB& aabb);
	/// Check if a node is in one of the global caches
	static bool isInACache(const AABB& aabb, bool sync = false);
	/// Check if a node is in all of the global caches
	static bool isInAllCaches(const AABB& aabb, bool sync = false);
	/// Display all stored nodes
	static void displayStored();
	static bool sanityCheckStored(const OctreeNode* root_node);
};

extern std::shared_ptr<LRUCache> updatesCache;
extern std::shared_ptr<LRUCache> visibilityCache;


extern std::mutex aabb_relationship_map_mtx;
extern std::unordered_map<AABB, std::array<std::optional<AABB>, 8>, AABB::Hash> aabb_relationship_map;
extern std::mutex aabb_parent_map_mtx;
extern std::unordered_map<AABB, AABB, AABB::Hash> aabb_parent_map;
extern std::unordered_map<AABB, std::mutex, AABB::Hash> aabb_mutex_map;


///////////////////////////////////////////////////////////////////////////////
/////////////////////// CUDA UNIFIED MEMORY SHENANIGANS ///////////////////////
///////////////////////////////////////////////////////////////////////////////

extern std::vector<std::shared_ptr<std::vector<vec3>>> unified_positions;
extern std::vector<std::shared_ptr<std::vector<uint32_t>>> unified_colors;

struct CFullOctreeUnifiedBuilder {
	std::vector<void*> nodes = {};
	uint32_t num_nodes = 0;
	uint32_t max_lod_level = 0;

	void update();
	CFullOctreeUnified build();
};
extern CFullOctreeUnifiedBuilder unifiedOctreeBuilder;