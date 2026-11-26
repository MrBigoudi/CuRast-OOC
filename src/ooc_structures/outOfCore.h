#pragma once

#include "gpuVersion.h"



///////////////////////////////////////////////////////////////////////////////
////////////////////////////// HELPER FUNCTIONS ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

std::string getNodeFilePath(const CIdAABB& aabb_index);
/// `.points' (`CPoint', 16 bytes each) or `.voxels' (`CVoxel', 8 bytes each: cell + color; the cells
/// are also the occupancy grid of the node, there is no separate `.grid' file anymore)
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

struct OctreeNodeSerializable {
    static void serialize(
        const std::shared_ptr<HostStorageNode> node
    );
    static void deserialize(
        HostStorageNode* node, const CIdAABB& aabb_index, 
        const std::string& msg = ""
    );
};