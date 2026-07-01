#pragma once

#include "CuRast.h"
#include "globals.h"

#include <array>

///////////////////////////////////////////////////////////////////////////////
//////////////////////////// FORWARD DECLARATION //////////////////////////////
///////////////////////////////////////////////////////////////////////////////

struct ChunkSerializable;
struct OctreeNodeSerializable;



///////////////////////////////////////////////////////////////////////////////
///////////////////////////// REAL DECLARATION ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// A serializable chunk
struct ChunkSerializable {
	/// All chunk have the same physical size even if empty
	std::vector<std::array<Point, OocSimLodSettings::NB_POINTS_PER_CHUNK>> points = {};
	std::vector<uint32_t> sizes = {};

    ChunkSerializable(){};
    ChunkSerializable(const Chunk* root_chunk);
    void serialize(const std::string& filepath) const;
    static ChunkSerializable deserialize(const std::string& filepath);
    Chunk* toChunk() const;
};

/// A serializable node
struct OctreeNodeSerializable {
    AABB aabb = {};
	uint32_t counter = 0;
	uint8_t children_ids = 0b00000000;
	std::string points = "";
	std::string voxels = "";

    OctreeNodeSerializable(){};

    /// Serializes all points, voxels and grids
    static void init(const OctreeNode* node, bool node_only);

    static OctreeNode* toOctreeNodes(
        const AABB& root_aabb, bool node_only
    );

    private:
        // helpers
        void serialize(const std::string& filepath) const;
        static OctreeNodeSerializable deserialize(const std::string& filepath);
        OctreeNode* toLeafNode(const AABB& node_aabb) const;
};


/// The LRU cache for the CPU fallback (before storing on disk)
struct CPUFallbackCache {
	/// A cache entry
	struct Entry {
        OctreeNodeSerializable serializable_node;
        std::optional<ChunkSerializable> serializable_points = nullopt;
        std::optional<ChunkSerializable> serializable_voxels = nullopt;
        uint32_t cache_counter = 0;

		/// A constructor from an existing node
		Entry(const OctreeNode* node, uint32_t cache_counter);

		/// Builds an octree node from an entry
		OctreeNode* toLeafNode() const;
	};

    uint32_t cache_counter = 0;
    std::unordered_map<AABB, Entry, AABB::Hash> data_cache = {};

    /// Add a node to the cache
    /// Optionally return the node that was removed from the cache after the insertion

    /// Get a node from the cache
    ///
};



///////////////////////////////////////////////////////////////////////////////
///////////////////////////// HELPER FUNCTIONS ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// Get a file name from a bounding volume
/// Numbers are separated with two '_'
/// Points in float are replaced by a single '_'
/// Negative numbers are prefixed with a 'n'
/// Example: 
///     aabb = {(-0.1, 0.2, -0.3), (0.1, 0.4, 0.3)}
///     fileName = "n0_1__0_2__n0_3__0_1__0_4__0_3"
std::string getFileName(const AABB& aabb);
std::string getNodeFilePath(const AABB& aabb);
std::string getOccupancyFilePath(const AABB& aabb);
std::string getChunkFilePath(const AABB& aabb, bool is_voxel);



///////////////////////////////////////////////////////////////////////////////
////////////////////////////// MAIN FUNCTIONS /////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// Store an octree node given it's AABB and the main octree
void storeOctree(const OctreeNode* node, bool node_only = false);

/// Load an octree from a file
/// Recursively loads all root node's children
OctreeNode* loadOctree(const AABB& root_aabb, bool node_only = false);

/// Add nodes to the updates cache after octree update
void updateUpdatesCache(OctreeNode* root_octree);