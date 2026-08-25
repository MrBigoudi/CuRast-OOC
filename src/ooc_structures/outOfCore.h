#pragma once

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
    ChunkSerializable(const std::vector<CPoint>& points);
    void serialize(const std::string& filepath) const;
    static ChunkSerializable deserialize(const std::string& filepath);
    Chunk* toChunk() const;
};

/// A serializable node
struct OctreeNodeSerializable {
	uint32_t points_counter = 0;
	uint32_t voxels_counter = 0;
	uint8_t children_ids = 0b00000000;
	std::string points = "";
	std::string voxels = "";
    IdAABB aabb_index = {};

    friend CPUFallbackCache;

    OctreeNodeSerializable(){};

    /// Serializes all nodes, points, and voxels
    static void serialize(const OctreeNode* node);
    static OctreeNode* toOctreeNode(const IdAABB& node_aabb_index);
    
    static void serializeV2(const std::shared_ptr<HostStorageNode> node);
    static std::shared_ptr<HostStorageNode> deserializeV2(const CIdAABB& aabb_index, const std::string& msg = "");

    private:
        // helpers
        void serialize(const std::string& filepath) const;
        static OctreeNodeSerializable deserialize(const std::string& filepath, const std::string& msg = "");
};


/// The LRU cache for the CPU fallback (before storing on disk)
/// https://www.geeksforgeeks.org/dsa/lru-cache-implementation-using-double-linked-lists/
struct CPUFallbackCache {
	// TODO: rework as the allocator to avoid useless free on add

	/// A cache entry
	struct Entry {
        OctreeNodeSerializable serializable_node = {};
        std::optional<ChunkSerializable> serializable_points = nullopt;
        std::optional<ChunkSerializable> serializable_voxels = nullopt;

        Entry(){}
		/// A constructor from an existing node
		Entry(const OctreeNode* node);
		Entry(const std::shared_ptr<HostStorageNode> node);
        /// A constructor which is deserialized from an aabb
        static Entry deserialize(const IdAABB& aabb_index);

		/// Builds an octree node from an entry
		OctreeNode* toLeafNode() const;
	};

    /// The size of the cache
	const uint32_t CACHE_SIZE;

    /// The global cache
    CDoubleLinkedList<const Entry*> cache = {};
	CHashMap<IdAABB, CDoubleLinkedList<const Entry*>::Iterator*> cache_map = {};

    /// Creates a cache given its size
    CPUFallbackCache(uint32_t cache_size);
    ~CPUFallbackCache(){
        cache = {};
        cache_map = {};
    }

    /// Add a node to the cache
    /// Optionally return the node that was removed from the cache after the insertion
    /// Note that new entries should overwrite its previous version if the node was already in cache
    const Entry* add(const Entry* new_entry);

    /// Get a node from the cache
    const Entry* get(const IdAABB& aabb_index);

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
void storeOctree(const OctreeNode* node);

/// Load an octree from a file
/// Recursively loads all root node's children
OctreeNode* loadOctree(const IdAABB& root_aabb_index);

/// Add nodes to the updates cache after octree update
void updateUpdatesCache(OctreeNode* root_octree);





///////////////////////////////////////////////////////////////////////////////
////////////////////////////// HELPER FUNCTIONS ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

std::string getFileName(const AABB& aabb);
std::string getNodeFilePath(const AABB& aabb);
std::string getOccupancyFilePath(const AABB& aabb);
std::string getChunkFilePath(const AABB& aabb, bool is_voxel);
std::string getNodeFilePathV2(const CIdAABB& aabb_index);
std::string getOccupancyFilePathV2(const CIdAABB& aabb_index);
std::string getChunkFilePathV2(const CIdAABB& aabb_index, bool is_voxel);
