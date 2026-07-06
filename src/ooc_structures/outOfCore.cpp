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
    return format("{}/{}.node", OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY, getFileName(aabb));
}
std::string getOccupancyFilePath(const AABB& aabb){
    return format("{}/{}.grid", OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY, getFileName(aabb));
}
std::string getChunkFilePath(const AABB& aabb, bool is_voxel){
    return format( "{}/{}.{}", 
        OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY, getFileName(aabb),
        is_voxel ? "voxels" : "points"
    );
}




/// A constructor from an existing node
CPUFallbackCache::Entry::Entry(const OctreeNode* node){
    serializable_node = {};
    serializable_node.counter = node->counter;
    serializable_node.children_ids = node->children_ids;
    serializable_node.aabb_index = node->aabb_index;

    const AABB& aabb = GlobalVariables::getAABB(node->aabb_index);
    if(node->points){
        serializable_node.points = getChunkFilePath(aabb, false);
        serializable_points = ChunkSerializable(node->points);
    }
    if(node->voxels){
        serializable_node.voxels = getChunkFilePath(aabb, true);
        serializable_voxels = ChunkSerializable(node->voxels);
    }
}

/// A constructor which is deserialized from an aabb
CPUFallbackCache::Entry::Entry(const IdAABB& aabb_index){
    const AABB& aabb = GlobalVariables::getAABB(aabb_index);

    serializable_node = OctreeNodeSerializable::deserialize(getNodeFilePath(aabb));
    if(serializable_node.points != ""){
        ChunkSerializable points_deserialized = ChunkSerializable::deserialize(
            getChunkFilePath(aabb, false)
        );
        serializable_points = points_deserialized.toChunk();
    }

    if(serializable_node.voxels != ""){
        ChunkSerializable voxels_deserialized = ChunkSerializable::deserialize(
            getChunkFilePath(aabb, true)
        );
        serializable_voxels = voxels_deserialized.toChunk();
    }
}

/// Builds an octree node from an entry
OctreeNode* CPUFallbackCache::Entry::toLeafNode() const {
    OctreeNode* new_node = new OctreeNode(serializable_node.aabb_index);
    new_node->counter = serializable_node.counter;
    new_node->children_ids = serializable_node.children_ids;

    if(serializable_points.has_value()){
        new_node->points = serializable_points->toChunk();
    }

    if(serializable_voxels.has_value()){
        new_node->voxels = serializable_voxels->toChunk();
    }

    new_node->rebuildOccupancy();
    return new_node;
}

CPUFallbackCache::CPUFallbackCache(uint32_t cache_size): CACHE_SIZE(cache_size){}

std::shared_ptr<CPUFallbackCache::Entry> CPUFallbackCache::add(const std::shared_ptr<Entry>& new_entry){
    auto it = cache_map.find(new_entry->serializable_node.aabb_index);

    // If the AABB was already in cache, remove its old version from the list
    if(it != cache_map.end()){
        cache.erase(it->second);
        cache_map.erase(it);
    }

    std::shared_ptr<Entry> old_entry = nullptr;

    // If the cache is full, remove the last node
    if(cache_map.size() >= CACHE_SIZE){
        old_entry = cache.back();
        cache.pop_back();
        cache_map.erase(old_entry->serializable_node.aabb_index);
    }

    // Insert the new node at the front of the list
    cache.push_front(new_entry);
    cache_map[new_entry->serializable_node.aabb_index] = cache.begin();

    return old_entry;
}

std::shared_ptr<CPUFallbackCache::Entry> CPUFallbackCache::get(const IdAABB& aabb_index) {
    auto it = cache_map.find(aabb_index);

    if(it != cache_map.end()){
        std::shared_ptr<CPUFallbackCache::Entry> output = *it->second;
        return output;
    }

    return nullptr;
}




///////////////////////////////////////////////////////////////////////////////
///////////////////////////// CHUNK SERIALIZATION /////////////////////////////
///////////////////////////////////////////////////////////////////////////////

ChunkSerializable::ChunkSerializable(const Chunk* root_chunk){
    const Chunk* cur_chunk = root_chunk;
    while(cur_chunk){
        // Copy points
        uint32_t cur_size = cur_chunk->size;
        sizes.push_back(cur_size);

        std::array<Point, OocSimLodSettings::NB_POINTS_PER_CHUNK> new_points = {};
        for(uint32_t i=0; i<cur_size; i++){
            new_points[i] = cur_chunk->points[i];
        }
        points.push_back(new_points);

        cur_chunk = cur_chunk->next;
    }
}

void ChunkSerializable::serialize(const std::string& filepath) const {
    // https://www.geeksforgeeks.org/cpp/serialize-and-deserialize-an-object-in-cpp/
    ofstream file(filepath, ios::binary | std::ios::trunc);
    if(!file.is_open()){
        println("Failed to open the file {} to serialize a chunk", filepath);
        if(!GlobalVariables::mainLoopIsTerminating){
            exit(EXIT_FAILURE);
        }
    }
    size_t nb_chunks = points.size();
    file.write(reinterpret_cast<const char*>(&nb_chunks), sizeof(nb_chunks));

    for(size_t i = 0; i < nb_chunks; ++i) {
        file.write(reinterpret_cast<const char*>(&sizes[i]), sizeof(uint32_t));

        file.write(
            reinterpret_cast<const char*>(points[i].data()),
            OocSimLodSettings::NB_POINTS_PER_CHUNK * sizeof(Point)
        );
    }

    file.close();
}

ChunkSerializable ChunkSerializable::deserialize(const std::string& filepath){
    ChunkSerializable new_chunk = {};

    // https://www.geeksforgeeks.org/cpp/serialize-and-deserialize-an-object-in-cpp/
    ifstream file(filepath, ios::binary);
    if(!file.is_open()){
        println("Failed to open the file {} to deserialize a chunk", filepath);
        if(!GlobalVariables::mainLoopIsTerminating){
            exit(EXIT_FAILURE);
        }
    }

    size_t nb_chunks = 0;
    file.read(reinterpret_cast<char*>(&nb_chunks), sizeof(nb_chunks));

    new_chunk.sizes.resize(nb_chunks);
    new_chunk.points.resize(nb_chunks);

    for(size_t i = 0; i < nb_chunks; ++i) {
        file.read(reinterpret_cast<char*>(&new_chunk.sizes[i]),
                  sizeof(uint32_t));

        file.read(
            reinterpret_cast<char*>(new_chunk.points[i].data()),
            OocSimLodSettings::NB_POINTS_PER_CHUNK * sizeof(Point)
        );
    }
    
    file.close();

    return new_chunk;
}

Chunk* ChunkSerializable::toChunk() const{
    Chunk* root_chunk = nullptr;
    uint32_t nb_chunks = sizes.size();

    for(int32_t chunk_id=nb_chunks-1; chunk_id>=0; chunk_id--){
        uint32_t cur_size = sizes[chunk_id];
        const std::array<Point, OocSimLodSettings::NB_POINTS_PER_CHUNK>& cur_points = points[chunk_id];

        Chunk* new_chunk = new Chunk();
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

void OctreeNodeSerializable::serialize(const OctreeNode* node){
    // Add the nodes to the CPU cache
    std::unordered_map<IdAABB, std::shared_ptr<CPUFallbackCache::Entry>> to_store = {};
    std::function<void (const OctreeNode*)> recursion = [&](const OctreeNode* cur_node){
        to_store.erase(cur_node->aabb_index);
        std::shared_ptr<CPUFallbackCache::Entry> new_entry = std::make_shared<CPUFallbackCache::Entry>(cur_node);
        std::shared_ptr<CPUFallbackCache::Entry> entry = GlobalVariables::cpuCache->add(new_entry);
        if(entry){
            to_store[entry->serializable_node.aabb_index] = entry;
        }
    };
    recursion(node);

    // Store the nodes that need to be on disk
    std::lock_guard<std::mutex> lock(GlobalVariables::mainLoopIsTerminatingMtx);
    for(auto [aabb_index, entry] : to_store){

        OctreeNodeSerializable& new_node = entry->serializable_node;
        if(entry->serializable_points.has_value()){
            ChunkSerializable& serializable = entry->serializable_points.value();
            serializable.serialize(new_node.points);
        }
        if(entry->serializable_voxels.has_value()){
            ChunkSerializable& serializable = entry->serializable_voxels.value();
            serializable.serialize(new_node.voxels);
        }

        const AABB& aabb = GlobalVariables::getAABB(aabb_index);
        new_node.serialize(getNodeFilePath(aabb));
    };
}

void OctreeNodeSerializable::serialize(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary | std::ios::trunc);

    if (!file.is_open()) {
        println("Failed to open the file {} to serialize an octree node", filepath);
        if(!GlobalVariables::mainLoopIsTerminating){
            exit(EXIT_FAILURE);
        }
    }

    // Write fixed-size members
    file.write(reinterpret_cast<const char*>(&counter), sizeof(counter));
    file.write(reinterpret_cast<const char*>(&children_ids), sizeof(children_ids));

    // Write points string
    uint64_t points_size = points.size();
    file.write(reinterpret_cast<const char*>(&points_size), sizeof(points_size));
    file.write(points.data(), points_size);

    // Write voxels string
    uint64_t voxels_size = voxels.size();
    file.write(reinterpret_cast<const char*>(&voxels_size), sizeof(voxels_size));
    file.write(voxels.data(), voxels_size);

    // Write aabb
    file.write(reinterpret_cast<const char*>(&aabb_index), sizeof(aabb_index));

    file.close();
}

OctreeNodeSerializable OctreeNodeSerializable::deserialize(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(GlobalVariables::mainLoopIsTerminatingMtx);
    OctreeNodeSerializable new_node = {};

    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        println("Failed to open the file {} to deserialize an octree node", filepath);
        if(!GlobalVariables::mainLoopIsTerminating){
            exit(EXIT_FAILURE);
        }
    }

    // Read fixed-size members
    file.read(reinterpret_cast<char*>(&new_node.counter), sizeof(new_node.counter));
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

    // Read aabb
    file.read(reinterpret_cast<char*>(&new_node.aabb_index), sizeof(aabb_index));

    return new_node;
}

OctreeNode* OctreeNodeSerializable::toLeafNode(const IdAABB& node_aabb_index) const {
    OctreeNode* new_node = new OctreeNode(node_aabb_index);
    new_node->counter = counter;
    new_node->children_ids = children_ids;

    const AABB& node_aabb = GlobalVariables::getAABB(node_aabb_index);
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

    new_node->rebuildOccupancy();
    return new_node;
}

OctreeNode* OctreeNodeSerializable::toOctreeNodes(const IdAABB& root_aabb_index){
    std::unordered_map<IdAABB, std::shared_ptr<CPUFallbackCache::Entry>> to_store = {};
    
    // Load all nodes indepentenly
    std::function<OctreeNode*(const IdAABB&, uint32_t, uint32_t)> recursion = 
        [&](const IdAABB& cur_aabb_index, uint32_t id, uint32_t level) {

        // Check if the node is in CPU cache
        to_store.erase(cur_aabb_index);
        std::shared_ptr<CPUFallbackCache::Entry> entry = GlobalVariables::cpuCache->get(cur_aabb_index);
        if(!entry){
            // If the node is not in CPU cache, load it from disk
            std::shared_ptr<CPUFallbackCache::Entry> new_entry = std::make_shared<CPUFallbackCache::Entry>(cur_aabb_index);
            std::shared_ptr<CPUFallbackCache::Entry> old_entry = GlobalVariables::cpuCache->add(new_entry);
            if(old_entry){
                to_store[old_entry->serializable_node.aabb_index] = old_entry;
            }
            entry = new_entry;
        }
        OctreeNode* new_node = entry->toLeafNode();

        return new_node;
    };

    OctreeNode* root = recursion(root_aabb_index, 0, 0);

    // Store the nodes that need to be on disk
    std::lock_guard<std::mutex> lock(GlobalVariables::mainLoopIsTerminatingMtx);
    for(auto [aabb_index, entry] : to_store){

        OctreeNodeSerializable& new_node = entry->serializable_node;
        if(entry->serializable_points.has_value()){
            ChunkSerializable& serializable = entry->serializable_points.value();
            serializable.serialize(new_node.points);
        }
        if(entry->serializable_voxels.has_value()){
            ChunkSerializable& serializable = entry->serializable_voxels.value();
            serializable.serialize(new_node.voxels);
        }

        const AABB& aabb = GlobalVariables::getAABB(aabb_index);
        new_node.serialize(getNodeFilePath(aabb));
    };

    return root;
}




void storeOctree(const OctreeNode* node){
    OctreeNodeSerializable::serialize(node);
    // println("Done storing octree");
}

OctreeNode* loadOctree(const IdAABB& root_aabb_index){
    // println("Start loading octree");
    OctreeNode* res = OctreeNodeSerializable::toOctreeNodes(root_aabb_index);
    // println("Done loading octree");
    return res;
}


/// Add nodes to cache after octree update
void updateUpdatesCache(OctreeNode* root_octree){
    std::lock_guard<std::mutex> lock(LRUCache::caches_sync_mtx);

    // Traverse octree and add newly updated aabbs to the cache
    std::function<void(const OctreeNode*)> recursionAddToCache = [&](const OctreeNode* cur_node){
        for(uint32_t child_id = 0; child_id < 8; child_id++){
            if(cur_node->children[child_id]){
                recursionAddToCache(cur_node->children[child_id]);
            }
        }

        if(cur_node->updated){
            GlobalVariables::updatesCache->add(cur_node->aabb_index);
        }

    };

    recursionAddToCache(root_octree);

    // Traverse octree and remove nodes that were just serialized
    std::function<bool(OctreeNode*, uint32_t, uint32_t)> 
        recursionRemoveNodes = 
            [&](OctreeNode* cur_node, uint32_t id, uint32_t level) -> bool
    {
        for(uint32_t child_id = 0; child_id < 8; child_id++){
            if(cur_node->children[child_id]){
                if(recursionRemoveNodes(cur_node->children[child_id], child_id, level+1)){
                    delete(cur_node->children[child_id]);
                    cur_node->children[child_id] = nullptr;
                }
            }
        }

        bool is_in_cache = GlobalVariables::updatesCache->contains(cur_node->aabb_index);
        bool to_remove = false;
        if(!is_in_cache){
            storeOctree(cur_node);
            to_remove = !GlobalVariables::visibilityCache->contains(cur_node->aabb_index);
        }

        cur_node->updated = false;
        return to_remove;
    };

    recursionRemoveNodes(root_octree, 0, 0);
}