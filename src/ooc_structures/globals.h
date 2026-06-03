#pragma once

#include "CuRast.h"
#include "laszip/laszip_api.h"
#include <semaphore>


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

/// The maximum number of batches that should be load from disk at once
constexpr uint8_t MAX_BATCHES_PER_LOAD = 100;
/// The maximum number of batches that should be loaded to the GPU at once
constexpr uint8_t MAX_BATCHES_PER_GPU_LOAD = 50;
// /// The maximum number of batches that should be used per octree update
// constexpr uint8_t MAX_BATCHES_PER_UPDATE = 1;
// /// The maximum number of points in a batch
// constexpr uint64_t MAX_BATCH_SIZE = 1'000'000;
/// The maximum number of points in a leaf node
constexpr uint16_t MAX_POINTS_PER_LEAF = 50'000;
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
	vec3 getCentroid() const;
	vec3 getSize() const;
	void extend(const NodePosition& position);
	void shrink(const NodePosition& position);
	vec3 getPointNormalizedCoordinates(const vec3& position) const;
	vec3 getPointWorldCoordinates(const vec3& normalized_position) const;
	NodePosition getNextChildIndex(const vec3& position) const;
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
	}
	bool operator==(const OccupancyGrid& rhs) const {
		for(uint32_t i=0; i<GRID_NUM_CELLS / 32u; i++){
			if(values[i] != rhs.values[i]){return false;};
		}
		return true;
	}
};

/// A node in an octree
struct OctreeNode {
	std::shared_ptr<OctreeNode> children[8] = {nullptr};
	uint16_t counter = 0;
	uint8_t children_ids = 0b00000000;
	std::shared_ptr<Chunk> points = nullptr;
	std::shared_ptr<Chunk> voxels = nullptr;
	std::shared_ptr<OccupancyGrid> occupancy = nullptr;

	uint32_t getNbPoints() const;
	uint32_t getNbVoxels() const;

	void display(uint32_t id = 0, uint32_t level = 0, bool node_only = false) const;

	bool operator==(const OctreeNode& rhs) const;

	OctreeNode(){}
	OctreeNode(const OctreeNode& cpy) : counter(cpy.counter), children_ids(cpy.children_ids){
		points = cpy.points ? std::make_shared<Chunk>(*cpy.points) : nullptr;
		voxels = cpy.voxels ? std::make_shared<Chunk>(*cpy.voxels) : nullptr;
		occupancy = cpy.occupancy ? std::make_shared<OccupancyGrid>(*cpy.occupancy) : nullptr;

		for(uint32_t child = 0; child < 8; child++){
			if(cpy.children[child]){
				children[child] = std::make_shared<OctreeNode>(*cpy.children[child]);
			}
		}
	}
};

/// A chunk linked list in a node
struct Chunk {
	/// All chunk have the same physical size even if empty
	Point points[POINTS_PER_CHUNK] = {Point()};
	uint32_t size = 0;
	/// For the linked list
	std::shared_ptr<Chunk> next = nullptr;

	bool operator==(const Chunk& rhs) const;

	Chunk(){}
	Chunk(const Chunk& cpy): size(cpy.size) {
		for(uint32_t i=0; i<POINTS_PER_CHUNK; i++){
			points[i] = cpy.points[i];
		}

		if(cpy.next){
			next = std::make_shared<Chunk>(*cpy.next);
		}
	}
};



///////////////////////////////////////////////////////////////////////////////
////////////////////////// GLOBAL EXTERNAL VARIABLES //////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// Name of the main octree node in the scene
const std::string simLodOctreeName = std::string("MainOctreeSimLOD");
/// Counter for the number of octree created
static uint64_t simLodOctreeCounter = 0;
std::string getSimLodOctreeName(bool generate_new_name = false);

/// Variables tracking when the octree can be sent to GPU
extern std::binary_semaphore octreeReadyToBeSent;
extern std::binary_semaphore octreeReadyToBeUpdated;

/// The queue of batches
extern std::deque<std::shared_ptr<PointBatch>> batchesQueue;
extern std::deque<std::mutex> batchesQueueMutexes;
extern std::mutex updateSceneMutex;

/// The main octree
extern std::shared_ptr<OctreeNode> mainOctree;
/// The main bounding box
extern std::shared_ptr<AABB> mainAABB;

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
std::shared_ptr<Timing> addTiming(string name, bool start_now = true, uint32_t level = 0);

void displayTimings();
void displayBuffers();