#include "outOfCore.h"


///////////////////////////////////////////////////////////////////////////////
////////////////////////////// HELPER FUNCTIONS ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

std::string getChunkFilePath(const CIdAABB& aabb_index, bool is_voxel){
    return format( "{}/{}.{}",
        OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY, aabb_index,
        is_voxel ? "voxels" : "points"
    );
}



///////////////////////////////////////////////////////////////////////////////
//////////////////////////////// CHUNK HANDLER ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename TElement>
void ChunkSerializable::serialize(
    const std::string& filepath, const TElement* elements,
    uint32_t nb_old_elements, uint32_t nb_new_elements
){
    ofstream file(filepath, ios::binary | std::ios::app);
    if(!file.is_open()){
        println("Failed to open the file {} to serialize a chunk", filepath);
        if(GpuVersion::mainLoopIsTerminating){return;}else{throw(EXIT_FAILURE);}
    }

    file.write(
        reinterpret_cast<const char*>(elements + nb_old_elements),
        uint64_t(nb_new_elements) * sizeof(TElement)
    );

    file.close();
}


template <typename TElement>
void ChunkSerializable::deserialize(
    const std::string& filepath, TElement* elements,
    uint32_t nb_elements, const std::string& msg
){
    ifstream file(filepath, ios::binary);
    if(!file.is_open()){
        println("Failed to open the file {} to deserialize a chunk: {}", filepath, msg);
        if(GpuVersion::mainLoopIsTerminating){return;}else{throw(EXIT_FAILURE);}
    }

    file.read(
        reinterpret_cast<char*>(elements),
        uint64_t(nb_elements) * sizeof(TElement)
    );

    file.close();
}

template void ChunkSerializable::serialize<CPoint>(const std::string&, const CPoint*, uint32_t, uint32_t);
template void ChunkSerializable::serialize<CVoxel>(const std::string&, const CVoxel*, uint32_t, uint32_t);
template void ChunkSerializable::deserialize<CPoint>(const std::string&, CPoint*, uint32_t, const std::string&);
template void ChunkSerializable::deserialize<CVoxel>(const std::string&, CVoxel*, uint32_t, const std::string&);



///////////////////////////////////////////////////////////////////////////////
//////////////////////////////// NODE HANDLER /////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void OctreeNodeSerializable::init(uint32_t max_nb_nodes){
    metadata = std::vector<StoredNodeMetadata>(max_nb_nodes);
}

void OctreeNodeSerializable::destroy(){
    metadata = {};
    metadata.shrink_to_fit();
}


void OctreeNodeSerializable::serialize(
    const std::shared_ptr<HostStorageNode> node
){
    const CIdAABB id = node->node.aabb_index;

    // Store points (only the new ones: the file is append-only; no file is opened for nothing)
    {
        uint32_t nb_old_points = node->node.points_last_stored;
        uint32_t nb_new_points = node->node.points_counter - nb_old_points;
        if(nb_new_points > 0){
            ChunkSerializable::serialize(getChunkFilePath(id, false), node->points,
                nb_old_points, nb_new_points
            );
        }
    }

    // Store voxels (their cells are the occupancy grid of the node)
    {
        uint32_t nb_old_voxels = node->node.voxels_last_stored;
        uint32_t nb_new_voxels = node->node.voxels_counter - nb_old_voxels;
        if(nb_new_voxels > 0){
            ChunkSerializable::serialize(getChunkFilePath(id, true), node->voxels,
                nb_old_voxels, nb_new_voxels
            );
        }
    }

    // Properties of the node, once its data is written (formerly the `.node' file)
    StoredNodeMetadata& entry = metadata[id];
    entry.children_ids = node->node.children_ids;
    entry.points_counter = node->node.points_counter;
    entry.voxels_counter = node->node.voxels_counter;
    entry.is_stored = true;

    node->node.points_last_stored = node->node.points_counter;
    node->node.voxels_last_stored = node->node.voxels_counter;
}


void OctreeNodeSerializable::attach(HostStorageNode* node, const CIdAABB& aabb_index){
    node->node.aabb_index = aabb_index;

    // Properties of the node (formerly the `.node' file); a node never stored has empty ones
    StoredNodeMetadata entry = {};
    {
        std::lock_guard<std::mutex> lock(GpuVersion::storedNodesMtx[aabb_index]);
        entry = metadata[aabb_index];
    }
    if(!entry.is_stored){ entry = StoredNodeMetadata{}; }

    // The host buffers hold MAX_POINTS_PER_LEAF elements: a larger node would be written past them
    if(entry.points_counter > OocSimLodSettings::MAX_POINTS_PER_LEAF
        || entry.voxels_counter > OocSimLodSettings::MAX_POINTS_PER_LEAF
    ){
        println("FATAL: attach, node {} has {} points / {} voxels, host buffers hold {}",
            aabb_index, entry.points_counter, entry.voxels_counter, OocSimLodSettings::MAX_POINTS_PER_LEAF
        );
        std::abort();
    }

    node->node.children_ids = entry.children_ids;
    node->node.points_counter = entry.points_counter;
    node->node.voxels_counter = entry.voxels_counter;
    // Everything counted so far is on disk...
    node->node.points_last_stored = entry.points_counter;
    node->node.voxels_last_stored = entry.voxels_counter;
    // ... and nothing of it is in host RAM: the RAM windows start where the files end
    node->points_valid_from = entry.points_counter;
    node->voxels_valid_from = entry.voxels_counter;
    // Empty RAM windows: complete
    node->readable_points = entry.points_counter;
    node->readable_voxels = entry.voxels_counter;
}


void OctreeNodeSerializable::readMissingVoxels(HostStorageNode* node, const std::string& msg){
    const uint32_t nb_voxels = node->voxels_valid_from;
    if(nb_voxels == 0){return;}
    // The voxel file is append-only and not written meanwhile (a node is only written when it leaves
    // the host cache, and the octree update waits for that): its first `nb_voxels' are the missing ones
    ChunkSerializable::deserialize(getChunkFilePath(node->node.aabb_index, true), node->voxels, nb_voxels, msg);
}