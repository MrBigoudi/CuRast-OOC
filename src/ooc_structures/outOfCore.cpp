#include "outOfCore.h"


///////////////////////////////////////////////////////////////////////////////
////////////////////////////// HELPER FUNCTIONS ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

std::string getNodeFilePath(const CIdAABB& aabb_index){
    return format("{}/{}.node", OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY, aabb_index);
}


std::string getOccupancyFilePath(const CIdAABB& aabb_index){
    return format("{}/{}.grid", OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY, aabb_index);
}


std::string getChunkFilePath(const CIdAABB& aabb_index, bool is_voxel){
    return format( "{}/{}.{}", 
        OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY, aabb_index,
        is_voxel ? "voxels" : "points"
    );
}



///////////////////////////////////////////////////////////////////////////////
//////////////////////////////// CHUNK HANDLER ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void ChunkSerializable::serialize(
    const std::string& filepath, const CPoint* points, 
    uint32_t nb_old_points, uint32_t nb_new_points
){
    ofstream file(filepath, ios::binary | std::ios::app);
    if(!file.is_open()){
        println("Failed to open the file {} to serialize a chunk", filepath);
        if(GpuVersion::mainLoopIsTerminating){return;}else{throw(EXIT_FAILURE);}
    }

    file.write(
        reinterpret_cast<const char*>(points + nb_old_points),
        nb_new_points * sizeof(CPoint)
    );

    file.close();
}


void ChunkSerializable::deserialize(
    const std::string& filepath, CPoint* points, 
    uint32_t nb_points, const std::string& msg
){
    ifstream file(filepath, ios::binary);
    if(!file.is_open()){
        println("Failed to open the file {} to deserialize a chunk: {}", filepath, msg);
        if(GpuVersion::mainLoopIsTerminating){return;}else{throw(EXIT_FAILURE);}
    }

    file.read(
        reinterpret_cast<char*>(points),
        nb_points * sizeof(CPoint)
    );
    
    file.close();
}



///////////////////////////////////////////////////////////////////////////////
//////////////////////////////// GRID HANDLER /////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void OccupancyGridSerializable::serialize(
    const std::string& filepath, const uint64_t* indices, 
    uint32_t nb_old_voxels, uint32_t nb_new_voxels
){
    ofstream file(filepath, ios::binary | std::ios::app);
    if(!file.is_open()){
        println("Failed to open the file {} to serialize a grid", filepath);
        if(GpuVersion::mainLoopIsTerminating){return;}else{throw(EXIT_FAILURE);}
    }

    file.write(
        reinterpret_cast<const char*>(indices + nb_old_voxels),
        nb_new_voxels * sizeof(uint64_t)
    );

    file.close();
}


void OccupancyGridSerializable::deserialize(
    const std::string& filepath, uint64_t* indices, 
    uint32_t nb_voxels, const std::string& msg
){
    ifstream file(filepath, ios::binary);
    if(!file.is_open()){
        println("Failed to open the file {} to deserialize a grid: {}", filepath, msg);
        if(GpuVersion::mainLoopIsTerminating){return;}else{throw(EXIT_FAILURE);}
    }

    file.read(
        reinterpret_cast<char*>(indices),
        nb_voxels * sizeof(uint64_t)
    );
    
    file.close();
}



///////////////////////////////////////////////////////////////////////////////
//////////////////////////////// NODE HANDLER /////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void OctreeNodeSerializable::serialize(
    const std::shared_ptr<HostStorageNode> node
){
    // Store node
    {
        std::string node_filepath = getNodeFilePath(node->node.aabb_index);
        std::ofstream file(node_filepath, std::ios::binary | std::ios::trunc);
        if(!file.is_open()){
            println("Failed to open the file {} to serialize an octree node", node_filepath);
            if(!GpuVersion::mainLoopIsTerminating){
                throw(EXIT_FAILURE);
            }
        }
        file.write(reinterpret_cast<const char*>(&node->node.aabb_index), sizeof(CIdAABB));
        file.write(reinterpret_cast<const char*>(&node->node.children_ids), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&node->node.points_counter), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&node->node.voxels_counter), sizeof(uint32_t));
        file.close();
    }

    // Store points
    if(node->node.points_counter > 0){
        std::string points_filepath = getChunkFilePath(node->node.aabb_index, false);
        uint32_t nb_old_points = node->node.points_last_stored;
        uint32_t nb_new_points = node->node.points_counter - nb_old_points;

        // ChunkSerializable::serialize(points_filepath, node->points.data(), 
        ChunkSerializable::serialize(points_filepath, node->points,
            nb_old_points, nb_new_points
        );
    }

    // Store voxels
    if(node->node.voxels_counter > 0){
        std::string voxels_filepath = getChunkFilePath(node->node.aabb_index, true);
        std::string grid_filepath = getOccupancyFilePath(node->node.aabb_index);

        uint32_t nb_old_voxels = node->node.voxels_last_stored;
        uint32_t nb_new_voxels = node->node.voxels_counter - nb_old_voxels;
        
        ChunkSerializable::serialize(voxels_filepath, node->voxels, nb_old_voxels, nb_new_voxels);
        OccupancyGridSerializable::serialize(grid_filepath, node->occupancy_indices, nb_old_voxels, nb_new_voxels);
    }

    node->node.points_last_stored = node->node.points_counter;
    node->node.voxels_last_stored = node->node.voxels_counter;
}


void OctreeNodeSerializable::deserialize(
    HostStorageNode* node, const CIdAABB& aabb_index, 
    const std::string& msg
){
    node->node.aabb_index = aabb_index;
    
    // Read node
    {
        std::string node_filepath = getNodeFilePath(node->node.aabb_index);
        std::ifstream file(node_filepath, std::ios::binary);
        if(!file.is_open()){
            println("Failed to open the file {} to deserialize an octree node: {}", node_filepath, msg);
            if(!GpuVersion::mainLoopIsTerminating){
                throw(EXIT_FAILURE);
            }
        }
        file.read(reinterpret_cast<char*>(&node->node.aabb_index), sizeof(CIdAABB));
        file.read(reinterpret_cast<char*>(&node->node.children_ids), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&node->node.points_counter), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&node->node.voxels_counter), sizeof(uint32_t));
        file.close();
    }
    node->node.points_last_stored = node->node.points_counter;
    node->node.voxels_last_stored = node->node.voxels_counter;

    // The host buffers hold MAX_POINTS_PER_LEAF elements: a larger read would write past them
    if(node->node.points_counter > OocSimLodSettings::MAX_POINTS_PER_LEAF
        || node->node.voxels_counter > OocSimLodSettings::MAX_POINTS_PER_LEAF
    ){
        println("FATAL: deserialize, node {} has {} points / {} voxels, host buffers hold {} ({})",
            aabb_index, node->node.points_counter, node->node.voxels_counter,
            OocSimLodSettings::MAX_POINTS_PER_LEAF, msg
        );
        std::abort();
    }

    // Read points
    if(node->node.points_counter > 0){
        std::string points_filepath = getChunkFilePath(node->node.aabb_index, false);
        ChunkSerializable::deserialize(points_filepath, node->points, node->node.points_counter, msg);
    }

    // Read voxels
    if(node->node.voxels_counter > 0){
        std::string voxels_filepath = getChunkFilePath(node->node.aabb_index, true);
        std::string grid_filepath = getOccupancyFilePath(node->node.aabb_index);
        ChunkSerializable::deserialize(voxels_filepath, node->voxels, node->node.voxels_counter, msg);
        OccupancyGridSerializable::deserialize(grid_filepath, node->occupancy_indices, node->node.voxels_counter, msg);
    }
}



