#pragma once

#include "CuRast.h"
#include "laszip/laszip_api.h"


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


///////////////////////////////////////////////////////////////////////////////
////////////////////////////// GLOBAL CONSTANTS ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// The maximum number of points in a batch
constexpr uint64_t MAX_BATCH_SIZE = 1'000'000;
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
};

/// A set of points read from a las / laz file
struct PointBatch {
	uint64_t first = 0;
	uint64_t count = 0;
	shared_ptr<string> file = nullptr;
	shared_ptr<vector<Point>> points = nullptr;
	shared_ptr<laszip_header> header = nullptr;

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
};

/// A chunk linked list in a node
struct Chunk {
	/// All chunk have the same physical size even if empty
	Point points[POINTS_PER_CHUNK];
	uint32_t size = 0;
	/// For the linked list
	std::shared_ptr<Chunk> next = nullptr;
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

/// The batches that still need to be read
extern std::deque<PointBatch> batchesToLoad;
/// The batches that are already loaded
extern std::deque<PointBatch> batchesLoaded;
/// The batches that have already been inserted in the octree
extern std::deque<PointBatch> batchesInserted;

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
	std::chrono::time_point<std::chrono::system_clock> start = {};
	std::chrono::time_point<std::chrono::system_clock> stop = {};
	std::chrono::microseconds duration = {};

	Timing(string name, bool start_now = true, uint32_t level = 0)
	 : name(name), has_started(start_now), level(level){
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