#pragma once

#include "gpuVersion.h"



///////////////////////////////////////////////////////////////////////////////
////////////////////////////// HELPER FUNCTIONS ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// `.points' (`CPoint', 16 bytes each) or `.voxels' (`CVoxel', 8 bytes each: cell + color; the cells
/// are also the occupancy grid of the node, there is no separate `.grid' file anymore)
/// There is no `.node' file anymore either: the properties of a stored node live in RAM
/// (`OctreeNodeSerializable::metadata')
std::string getChunkFilePath(const CIdAABB& aabb_index, bool is_voxel);



///////////////////////////////////////////////////////////////////////////////
//////////////////////////////// CHUNK HANDLER ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// Append-only file of points (`CPoint') or voxels (`CVoxel')
struct ChunkSerializable {
    template <typename TElement>
    static void serialize(
        const std::string& filepath, const TElement* elements,
        uint32_t nb_old_elements, uint32_t nb_new_elements
    );
    template <typename TElement>
    static void deserialize(
        const std::string& filepath, TElement* elements,
        uint32_t nb_elements, const std::string& msg
    );
};



///////////////////////////////////////////////////////////////////////////////
//////////////////////////////// NODE HANDLER /////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// Properties of a node whose data is on disk (what the former `.node' file held: 12 bytes per node,
/// opened, truncated, written and closed at every serialisation)
/// The counters are the numbers of elements of the `.points' / `.voxels' files (append-only)
struct StoredNodeMetadata {
    uint32_t children_ids = 0;
    uint32_t points_counter = 0;
    uint32_t voxels_counter = 0;
    bool is_stored = false;
};

struct OctreeNodeSerializable {
    /// One entry per node id (`MAX_NB_NODES': 16 bytes each), indexed by id
    /// Entry `id' is only accessed under `GpuVersion::storedNodesMtx[id]', like the files of the node
    static inline std::vector<StoredNodeMetadata> metadata = {};

    static void init(uint32_t max_nb_nodes);
    static void destroy();

    /// Append the elements of the node not written yet (`*_last_stored' to `*_counter') and update its
    /// metadata. The caller holds `GpuVersion::storedNodesMtx[id]'
    static void serialize(
        const std::shared_ptr<HostStorageNode> node
    );
    /// Main thread: set a new host node up from the metadata of its id, WITHOUT reading anything: its
    /// counters, children, and empty RAM windows starting at the counts on disk
    /// (`HostStorageNode::*_valid_from'). A node never stored gets empty counters
    /// The octree update never needs the stored points again (the device only receives new points):
    /// they are only read by the visibility update, from the files
    static void attach(HostStorageNode* node, const CIdAABB& aabb_index);
    /// Any thread: read the voxels the host node lacks (`[0, voxels_valid_from)', all of them on
    /// disk) into their place in its buffer. Needed when the node is loaded to the device (its
    /// occupancy grid is rebuilt from them). The caller moves `voxels_valid_from' to 0 on the main
    /// thread once the read is complete. Writes nothing the visibility update reads (it copies
    /// elements from `voxels_valid_from' on)
    static void readMissingVoxels(HostStorageNode* node, const std::string& msg = "");
};