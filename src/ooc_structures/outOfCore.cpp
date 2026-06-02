#include "outOfCore.h"

#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////
////////////////////////////// HELPER FUNCTIONS ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

std::string getFileName(const AABB& aabb){
    auto encodeFloat = [](float value) -> std::string {
        float abs_value = std::abs(value);
        std::ostringstream ss;
        ss << std::setprecision(50) << std::defaultfloat << abs_value;
        std::string res = ss.str();

        // Replace '.' with '_'
        std::replace(res.begin(), res.end(), '.', '_');

        // Prefix negative numbers with 'n'
        if(value < 0.f){res = "n" + res;}
        
        return res;
    };

    std::string res = format("{}__{}__{}__{}__{}__{}",
        encodeFloat(aabb.mins.x),
        encodeFloat(aabb.mins.y),
        encodeFloat(aabb.mins.z),
        encodeFloat(aabb.maxs.x),
        encodeFloat(aabb.maxs.y),
        encodeFloat(aabb.maxs.z)
    );

    return res;
}
std::string getNodeFilePath(const AABB& aabb){
    return format("{}/{}.node", TEMPORARY_DIRECTORY, getFileName(aabb));
}
std::string getOccupancyFilePath(const AABB& aabb){
    return format("{}/{}.grid", TEMPORARY_DIRECTORY, getFileName(aabb));
}
std::string getChunkFilePath(const AABB& aabb, bool is_voxel){
    return format( "{}/{}.{}", 
        TEMPORARY_DIRECTORY, getFileName(aabb),
        is_voxel ? "voxels" : "points"
    );
}




///////////////////////////////////////////////////////////////////////////////
///////////////////////////// CHUNK SERIALIZATION /////////////////////////////
///////////////////////////////////////////////////////////////////////////////

ChunkSerializable::ChunkSerializable(const std::shared_ptr<Chunk>& root_chunk){
    Chunk* cur_chunk = root_chunk.get();
    while(cur_chunk){
        // Copy points
        uint32_t cur_size = cur_chunk->size;
        sizes.push_back(cur_size);

        std::array<Point, POINTS_PER_CHUNK> new_points = {};
        for(uint32_t i=0; i<cur_size; i++){
            new_points[i] = cur_chunk->points[i];
        }
        points.push_back(new_points);

        cur_chunk = cur_chunk->next.get();
    }
}

void ChunkSerializable::serialize(const std::string& filepath) const {
    // https://www.geeksforgeeks.org/cpp/serialize-and-deserialize-an-object-in-cpp/
    ofstream file(filepath, ios::binary);
    if(!file.is_open()){
        println("Failed to open the file {} to serialize a chunk", filepath);
        exit(EXIT_FAILURE);
    }
    size_t nb_chunks = points.size();
    file.write(reinterpret_cast<const char*>(&nb_chunks), sizeof(nb_chunks));

    for(size_t i = 0; i < nb_chunks; ++i) {
        file.write(reinterpret_cast<const char*>(&sizes[i]), sizeof(uint32_t));

        file.write(
            reinterpret_cast<const char*>(points[i].data()),
            POINTS_PER_CHUNK * sizeof(Point)
        );
    }

    file.close();
}

ChunkSerializable ChunkSerializable::deserialize(const std::string& filepath){
    ChunkSerializable new_chunk = {};

    // https://www.geeksforgeeks.org/cpp/serialize-and-deserialize-an-object-in-cpp/
    ifstream file(filepath, ios::binary);

    size_t nb_chunks;
    file.read(reinterpret_cast<char*>(&nb_chunks), sizeof(nb_chunks));

    new_chunk.sizes.resize(nb_chunks);
    new_chunk.points.resize(nb_chunks);

    for(size_t i = 0; i < nb_chunks; ++i) {
        file.read(reinterpret_cast<char*>(&new_chunk.sizes[i]),
                  sizeof(uint32_t));

        file.read(
            reinterpret_cast<char*>(new_chunk.points[i].data()),
            POINTS_PER_CHUNK * sizeof(Point)
        );
    }
    
    file.close();

    return new_chunk;
}

std::shared_ptr<Chunk> ChunkSerializable::toChunk() const{
    std::shared_ptr<Chunk> root_chunk = nullptr;
    uint32_t nb_chunks = sizes.size();

    for(int32_t chunk_id=nb_chunks-1; chunk_id>=0; chunk_id--){
        uint32_t cur_size = sizes[chunk_id];
        const std::array<Point, POINTS_PER_CHUNK>& cur_points = points[chunk_id];

        std::shared_ptr<Chunk> new_chunk = std::make_shared<Chunk>();
        new_chunk->size = cur_size;
        for(uint32_t point_id = 0; point_id < cur_size; point_id++){
            new_chunk->points[point_id] = cur_points[point_id];
        }
        new_chunk->next = root_chunk;
        root_chunk = new_chunk;
    }

    return root_chunk;
}




///////////////////////////////////////////////////////////////////////////////
//////////////////////////// OCTREE SERIALIZATION /////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void OctreeNodeSerializable::init(
    const AABB& root_aabb, 
    const std::shared_ptr<OctreeNode>& root_node
){
    std::function<void (const AABB&, const std::shared_ptr<OctreeNode>&)> recursion = 
        [&](const AABB& cur_aabb, const std::shared_ptr<OctreeNode>& cur_node){
            OctreeNodeSerializable new_node = {};
            new_node.counter = cur_node->counter;
            new_node.children_ids = cur_node->children_ids;

            if(cur_node->points){
                new_node.points = getChunkFilePath(cur_aabb, false);
                ChunkSerializable serializable = ChunkSerializable(cur_node->points);
                serializable.serialize(new_node.points);
            }
            if(cur_node->voxels){
                new_node.voxels = getChunkFilePath(cur_aabb, true);
                ChunkSerializable serializable = ChunkSerializable(cur_node->voxels);
                serializable.serialize(new_node.voxels);
            }

            // TODO: recreate occupancy grid
            // if(cur_node->occupancy){
            //     new_node.occupancy = getOccupancyFilePath(cur_aabb);
            //     // TODO: occupancy grid
            // }
            
            for(uint32_t child_id = 0; child_id < 8; child_id++){
                if(cur_node->children[child_id]){
                    new_node.children |= (0x01 << child_id);

                    AABB child_aabb = cur_aabb;
                    child_aabb.shrink((NodePosition)child_id);
                    std::string new_filepath = getNodeFilePath(child_aabb);
                    recursion(child_aabb, cur_node->children[child_id]);
                }
            }

            new_node.serialize(getNodeFilePath(cur_aabb));
    };

    // root_node->display();
    recursion(root_aabb, root_node);
}

void OctreeNodeSerializable::serialize(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        println("Failed to open the file {} to serialize an octree node", filepath);
        std::exit(EXIT_FAILURE);
    }

    // Write fixed-size members
    file.write(reinterpret_cast<const char*>(&counter), sizeof(counter));
    file.write(reinterpret_cast<const char*>(&children), sizeof(children));
    file.write(reinterpret_cast<const char*>(&children_ids), sizeof(children_ids));

    // Write points string
    uint64_t points_size = points.size();
    file.write(reinterpret_cast<const char*>(&points_size), sizeof(points_size));
    file.write(points.data(), points_size);

    // Write voxels string
    uint64_t voxels_size = voxels.size();
    file.write(reinterpret_cast<const char*>(&voxels_size), sizeof(voxels_size));
    file.write(voxels.data(), voxels_size);

    file.close();
}

OctreeNodeSerializable OctreeNodeSerializable::deserialize(const std::string& filepath) {
    OctreeNodeSerializable new_node = {};

    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        println("Failed to open the file {} to deserialize an octree node", filepath);
        std::exit(EXIT_FAILURE);
    }

    // Read fixed-size members
    file.read(reinterpret_cast<char*>(&new_node.counter), sizeof(new_node.counter));
    file.read(reinterpret_cast<char*>(&new_node.children), sizeof(new_node.children));
    file.read(reinterpret_cast<char*>(&new_node.children_ids), sizeof(new_node.children_ids));

    // Read points string
    uint64_t points_size = 0;
    file.read(reinterpret_cast<char*>(&points_size), sizeof(points_size));
    new_node.points.resize(points_size);
    file.read(new_node.points.data(), points_size);

    // Read voxels string
    uint64_t voxels_size = 0;
    file.read(reinterpret_cast<char*>(&voxels_size), sizeof(voxels_size));
    new_node.voxels.resize(voxels_size);
    file.read(new_node.voxels.data(), voxels_size);

    return new_node;
}

std::shared_ptr<OctreeNode> OctreeNodeSerializable::toLeafNode(const AABB& node_aabb) const{
    std::shared_ptr<OctreeNode> new_node = std::make_shared<OctreeNode>();
    new_node->counter = counter;
    new_node->children_ids = children_ids;

    if(points != ""){
        ChunkSerializable points_deserialized = ChunkSerializable::deserialize(
            getChunkFilePath(node_aabb, false)
        );
        new_node->points = points_deserialized.toChunk();
    }

    if(voxels != ""){
        ChunkSerializable voxels_deserialized = ChunkSerializable::deserialize(
            getChunkFilePath(node_aabb, true)
        );
        new_node->voxels = voxels_deserialized.toChunk();
    }

    // TODO: occupancy grid

    return new_node;
}

std::shared_ptr<OctreeNode> OctreeNodeSerializable::toOctreeNodes(
    const AABB& root_aabb
){
    std::unordered_map<std::string, std::shared_ptr<OctreeNode>> map = {};
    
    // Load all nodes indepentenly
    std::function<void(const AABB&, uint32_t, uint32_t)> recursion = 
        [&](const AABB& cur_aabb, uint32_t id, uint32_t level) {

        std::string filepath = getNodeFilePath(cur_aabb);
        OctreeNodeSerializable new_serializable_node = OctreeNodeSerializable::deserialize(filepath);
        std::shared_ptr<OctreeNode> new_node = new_serializable_node.toLeafNode(cur_aabb);

        // new_node->display(id, level, true);

        for(uint32_t child_id = 0; child_id < 8; child_id++){
            if(new_serializable_node.children & (0x01 << child_id)){
                AABB child_aabb = cur_aabb;
                child_aabb.shrink((NodePosition)child_id);
                
                recursion(child_aabb, child_id, level+1);
                // Fill up children pointers
                std::string child_filepath = getNodeFilePath(child_aabb);
                new_node->children[child_id] = map[child_filepath];
            }
        }

        map[filepath] = {new_node};
    };

    recursion(root_aabb, 0, 0);
    return map[getNodeFilePath(root_aabb)];
}




void storeOctree(const AABB& root_aabb, const std::shared_ptr<OctreeNode>& root_node){
    OctreeNodeSerializable::init(root_aabb, root_node);
    println("Done storing octree");
}

std::shared_ptr<OctreeNode> loadOctree(const AABB& root_aabb){
    println("Start loading octree");
    auto res = OctreeNodeSerializable::toOctreeNodes(root_aabb);
    println("Done loading octree");
    return res;
}