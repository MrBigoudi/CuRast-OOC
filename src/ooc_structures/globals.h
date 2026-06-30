#pragma once

#include "CuRast.h"
#include "laszip/laszip_api.h"
#include <semaphore>

#include <unordered_set>

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/hash.hpp"

#include "settings.h"

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

struct OccupancyGrid {
	// gridsize^3 occupancy grid; 1 bit per voxel
	uint32_t values[OocSimLodSettings::GRID_SIZE / 32] = {0};

	OccupancyGrid(){}
	OccupancyGrid(const OccupancyGrid& cpy){
		for(uint32_t i=0; i<OocSimLodSettings::GRID_SIZE / 32; i++){
			values[i] = cpy.values[i];
		}
		assert(cpy == *this);
	}
	bool operator==(const OccupancyGrid& rhs) const {
		for(uint32_t i=0; i<OocSimLodSettings::GRID_SIZE / 32; i++){
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
		for(uint32_t i=0; i<OocSimLodSettings::GRID_SIZE / 32; i++){
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
	Point points[OocSimLodSettings::NB_POINTS_PER_CHUNK] = {Point()};
	uint32_t size = 0;
	/// For the linked list
	Chunk* next = nullptr;

	bool operator==(const Chunk& rhs) const;

	Chunk(){}
	Chunk(const Chunk& cpy): size(cpy.size) {
		for(uint32_t i=0; i<OocSimLodSettings::NB_POINTS_PER_CHUNK; i++){
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
	AABB aabb;

	uint32_t counter = 0;
	uint8_t children_ids = 0b00000000;
	uint8_t children_visibility = 0b00000000;
	uint8_t level = 0;

	bool updated = false;
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
		for(uint32_t i=0; i<8; i++){
			if(children[i]){
				OctreeNode* tmp = children[i];
				children[i] = nullptr;
				delete(tmp);
			}
		}
	}

	OctreeNode(const AABB& aabb):aabb(aabb){}
	OctreeNode(const OctreeNode& cpy) : aabb(cpy.aabb), counter(cpy.counter), children_ids(cpy.children_ids) 
	{
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

	/// The timings list
	static vector<std::shared_ptr<Timing>> timingsList;
	static std::mutex timingsMtx;
	static std::shared_ptr<Timing> addTiming(string name, bool start_now = true, uint32_t level = 0);
};


void displayTimings();
void displayBuffers();




///////////////////////////////////////////////////////////////////////////////
/////////////////////////// LRU CACHING SHENANIGANS ///////////////////////////
///////////////////////////////////////////////////////////////////////////////

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



///////////////////////////////////////////////////////////////////////////////
////////////////////////// MEMORY BATCHING ALLOCATOR //////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// A structure to handle batched memory transfers
struct BatchedMemory {
    std::vector<CUdeviceptr> srcs = {};
    std::vector<CUdeviceptr> dsts = {};
    std::vector<size_t> sizes = {};

	// CPU side
    char* allocated_memory = nullptr;
    uint64_t next_space_pointer = 0;
	uint64_t memory_size = 0;

	// GPU side
	CUdeviceptr gpu_allocated_memory = 0;
	CUmemGenericAllocationHandle gpu_allocation_handle = 0;

    void init(CuRast* instance, CUcontext* context);
	void reset();

    ~BatchedMemory();

	template<typename T>
	std::pair<T*, CUdeviceptr> allocate(){
		size_t alignment = alignof(T);
		size_t size = sizeof(T);
		uint64_t aligned_pointer = next_space_pointer;
		aligned_pointer += (alignment - (aligned_pointer % alignment)) % alignment;
		if(aligned_pointer + size > memory_size){
			println("Can't allocate more memory...");
			exit(EXIT_FAILURE);
		}
		next_space_pointer = aligned_pointer + size;

		T* res = reinterpret_cast<T*>(allocated_memory + aligned_pointer);
		new (res) T();

		CUdeviceptr res_gpu = gpu_allocated_memory + aligned_pointer;
		return {res, res_gpu};
	}

	template<typename T>
    void addFutureCopy(T* src, CUdeviceptr dst){
		srcs.push_back((CUdeviceptr)src);
		dsts.push_back(dst);
		sizes.push_back(sizeof(T));
	}

    void copyMemory(CUcontext* context, CUstream* stream);

	void display() const;
};



///////////////////////////////////////////////////////////////////////////////
////////////////////////// GLOBAL EXTERNAL VARIABLES //////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct GlobalVariables {
	/// The queue of batches
	static std::deque<std::shared_ptr<PointBatch>> batchesQueue;
	static std::deque<std::mutex> batchesQueueMutexes;

	/// The buffer of spilled points
	static std::shared_ptr<vector<Point>> spilledPoints;
	/// The buffer of spilling nodes
	static std::shared_ptr<vector<OctreeNode*>> spillingNodes;

	/// The backlog buffer for new voxels
	static std::shared_ptr<vector<Point>> backlogVoxels;
	/// The backlog buffer for the nodes corresponding to the new voxels
	static std::shared_ptr<vector<OctreeNode*>> backlogVoxelsNodes;


	static uint32_t elapsedFrames;
	static uint64_t nbPoints;
	static bool mainLoopIsTerminating;
	static std::mutex mainLoopIsTerminatingMtx;
	/// Counter for the number of octree created
	static uint64_t simLodOctreeCounter;

	/// Variables tracking when the octree can be sent to GPU
	static std::binary_semaphore octreeReadyToBeSent;
	static std::binary_semaphore octreeReadyToBeUpdated;
	static std::binary_semaphore octreeNotBeingSent;
	static std::mutex isUpdatingMtx;

	static uint64_t loadCounter;
	static std::mutex loadCounterMtx;

	static bool lodUpdated;

	/// The queue of batches
	static std::mutex updateSceneMutex;

	/// The main octree
	static std::shared_ptr<OctreeNode> mainOctree;
	static OctreeNode* mainOctreeCpy;

	/// The LRU caches
	static std::shared_ptr<LRUCache> updatesCache;
	static std::shared_ptr<LRUCache> visibilityCache;


	static std::mutex aabbRelationshipMapMtx;
	static std::unordered_map<AABB, std::array<std::optional<AABB>, 8>, AABB::Hash> aabbRelationshipMap;
	static std::unordered_map<AABB, AABB, AABB::Hash> aabbParentMap;
	static std::unordered_map<AABB, std::mutex, AABB::Hash> aabbMutexMap;

	/// The global allocated memory (for batches)
	static BatchedMemory batchedMemory;

	static std::string getSimLodOctreeName(bool generate_new_name = false);
	static void init(CuRast* instance, CUcontext* context);
};