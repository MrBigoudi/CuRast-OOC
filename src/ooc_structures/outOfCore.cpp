#include "outOfCore.h"
#include "allocator.h"

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

std::string getNodeFilePathV2(const CIdAABB& aabb_index){
    return format("{}/{}.node", OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY, aabb_index);
}
std::string getOccupancyFilePathV2(const CIdAABB& aabb_index){
    return format("{}/{}.grid", OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY, aabb_index);
}
std::string getChunkFilePathV2(const CIdAABB& aabb_index, bool is_voxel){
    return format( "{}/{}.{}", 
        OocSimLodSettings::TEMPORARY_NODE_STORAGE_DIRECTORY, aabb_index,
        is_voxel ? "voxels" : "points"
    );
}




/// A constructor from an existing node
CPUFallbackCache::Entry::Entry(const OctreeNode* node){
    serializable_node = {};
    serializable_node.points_counter = node->counter.load();
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

/// A constructor from an existing node
CPUFallbackCache::Entry::Entry(const COctreeNode* node, 
    const std::vector<CPoint>& points, 
    const std::vector<CPoint>& voxels
){
    serializable_node = {};
    serializable_node.points_counter = node->points_counter;
    serializable_node.voxels_counter = node->voxels_counter;
    serializable_node.children_ids = node->children_ids;
    serializable_node.aabb_index = node->aabb_index;
    serializable_node.aabb = AABB();
    serializable_node.aabb.maxs = node->aabb.maxs;
    serializable_node.aabb.mins = node->aabb.mins;

    if(!points.empty()){
        serializable_node.points = getChunkFilePathV2(serializable_node.aabb_index, false);
        serializable_points = ChunkSerializable(points);
    }
    if(!voxels.empty()){
        serializable_node.voxels = getChunkFilePathV2(serializable_node.aabb_index, true);
        serializable_voxels = ChunkSerializable(voxels);
    }
}

/// A constructor which is deserialized from an aabb
CPUFallbackCache::Entry CPUFallbackCache::Entry::deserialize(const IdAABB& aabb_index){
    const AABB& aabb = GlobalVariables::getAABB(aabb_index);
    Entry new_entry = {};
    new_entry.serializable_node = OctreeNodeSerializable::deserialize(getNodeFilePath(aabb));
    if(new_entry.serializable_node.points != ""){
        ChunkSerializable points_deserialized = ChunkSerializable::deserialize(
            getChunkFilePath(aabb, false)
        );
        new_entry.serializable_points = std::optional<ChunkSerializable>(points_deserialized);
    }

    if(new_entry.serializable_node.voxels != ""){
        ChunkSerializable voxels_deserialized = ChunkSerializable::deserialize(
            getChunkFilePath(aabb, true)
        );
        new_entry.serializable_voxels = std::optional<ChunkSerializable>(voxels_deserialized);
    }

    return new_entry;
}

/// A constructor which is deserialized from an aabb
CPUFallbackCache::Entry CPUFallbackCache::Entry::deserializeV2(const CIdAABB& aabb_index){
    Entry new_entry = {};
    new_entry.serializable_node = OctreeNodeSerializable::deserialize(getNodeFilePathV2(aabb_index));
    if(new_entry.serializable_node.points != ""){
        ChunkSerializable points_deserialized = ChunkSerializable::deserialize(
            getChunkFilePathV2(aabb_index, false)
        );
        new_entry.serializable_points = std::optional<ChunkSerializable>(points_deserialized);
    }

    if(new_entry.serializable_node.voxels != ""){
        ChunkSerializable voxels_deserialized = ChunkSerializable::deserialize(
            getChunkFilePathV2(aabb_index, true)
        );
        new_entry.serializable_voxels = std::optional<ChunkSerializable>(voxels_deserialized);
    }

    return new_entry;
}

/// Builds an octree node from an entry
OctreeNode* CPUFallbackCache::Entry::toLeafNode() const {
    // OctreeNode* new_node = new OctreeNode(serializable_node.aabb_index);
    OctreeNode* new_node = MemoryAllocator::newOctreeNode(serializable_node.aabb_index);
    
    new_node->counter.store(serializable_node.points_counter);
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

CPUFallbackCache::CPUFallbackCache(uint32_t cache_size): CACHE_SIZE(cache_size){
    cache.init();
    cache_map.init(cache_size);
}

const CPUFallbackCache::Entry* CPUFallbackCache::add(const Entry* new_entry){
    CDoubleLinkedList<const Entry*>::Iterator** it = cache_map.find(new_entry->serializable_node.aabb_index);

    // If the AABB was already in cache, remove its old version from the list
    if(it){
        CDoubleLinkedList<const Entry*>::Iterator* list_it = *it;
        cache.erase(list_it);
        cache_map.erase(list_it->value->serializable_node.aabb_index);
        delete(list_it);
    }

    const Entry* old_entry = nullptr;

    // If the cache is full, remove the last node
    if(cache_map.size >= CACHE_SIZE){
        old_entry = *cache.back();
        cache.popBack();
        cache_map.erase(old_entry->serializable_node.aabb_index);
    }

    // Insert the new node at the front of the list
    cache.pushFront(new_entry);
    cache_map[new_entry->serializable_node.aabb_index] = cache.begin();

    return old_entry;
}

const CPUFallbackCache::Entry* CPUFallbackCache::get(const IdAABB& aabb_index) {    
    CDoubleLinkedList<const Entry*>::Iterator** it = cache_map.find(aabb_index);

    if(it){
        CDoubleLinkedList<const Entry*>::Iterator* list_it = *it;
        const Entry* output = list_it->value;
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

ChunkSerializable::ChunkSerializable(const std::vector<CPoint>& root_points){
    uint32_t total_size = root_points.size();
    if(!total_size){return;}
    uint32_t nb_chunks = (total_size + OocSimLodSettings::NB_POINTS_PER_CHUNK - 1) / OocSimLodSettings::NB_POINTS_PER_CHUNK;
    uint32_t last_size = total_size % OocSimLodSettings::NB_POINTS_PER_CHUNK;
    
    uint32_t cur_index = 0;
    for(uint32_t i=0; i<nb_chunks; i++){
        std::array<Point, OocSimLodSettings::NB_POINTS_PER_CHUNK> new_points = {};
        sizes.push_back(OocSimLodSettings::NB_POINTS_PER_CHUNK);
        for(uint32_t j=0; j<OocSimLodSettings::NB_POINTS_PER_CHUNK; j++){
            if(cur_index >= root_points.size()){break;}
            new_points[j] = {};
            new_points[j].position = root_points[cur_index].position;
            new_points[j].color[0] = uint8_t((root_points[cur_index].color << 24) >> 24);
            new_points[j].color[1] = uint8_t(((root_points[cur_index].color >> 8) << 24) >> 24);
            new_points[j].color[2] = uint8_t(((root_points[cur_index].color >> 16) << 24) >> 24);
            new_points[j].color[3] = uint8_t(root_points[cur_index].color >> 24);
            cur_index++;
        }
        points.push_back(new_points);
    }
    sizes.back() = last_size;
}

void ChunkSerializable::serialize(const std::string& filepath) const {
    std::lock_guard<std::mutex> lock(GlobalVariables::mainLoopIsTerminatingMtx);

    // https://www.geeksforgeeks.org/cpp/serialize-and-deserialize-an-object-in-cpp/
    ofstream file(filepath, ios::binary | std::ios::trunc);
    if(!file.is_open()){
        println("Failed to open the file {} to serialize a chunk", filepath);
        if(!GlobalVariables::mainLoopIsTerminating){
            throw(EXIT_FAILURE);
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

    std::lock_guard<std::mutex> lock(GlobalVariables::mainLoopIsTerminatingMtx);

    // https://www.geeksforgeeks.org/cpp/serialize-and-deserialize-an-object-in-cpp/
    ifstream file(filepath, ios::binary);
    if(!file.is_open()){
        println("Failed to open the file {} to deserialize a chunk", filepath);
        if(!GlobalVariables::mainLoopIsTerminating){
            throw(EXIT_FAILURE);
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

        // Chunk* new_chunk = new Chunk();
        Chunk* new_chunk = MemoryAllocator::newChunk();
        
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
    // std::lock_guard<std::mutex> lock_cache(GlobalVariables::cpuCache->mutex);

    // TODO: rework cache version
    // // Add the nodes to the CPU cache
    // const CPUFallbackCache::Entry* new_entry = new CPUFallbackCache::Entry(node);
    // const CPUFallbackCache::Entry* to_store = GlobalVariables::cpuCache->add(new_entry);

    // // Store node if removed from cache
    // std::lock_guard<std::mutex> lock(GlobalVariables::mainLoopIsTerminatingMtx);
    // if(to_store){
    //     const IdAABB& aabb_index = to_store->serializable_node.aabb_index;
    //     const OctreeNodeSerializable& new_node = to_store->serializable_node;
    //     if(to_store->serializable_points.has_value()){
    //         const ChunkSerializable& serializable = to_store->serializable_points.value();
    //         serializable.serialize(new_node.points);
    //     }
    //     if(to_store->serializable_voxels.has_value()){
    //         const ChunkSerializable& serializable = to_store->serializable_voxels.value();
    //         serializable.serialize(new_node.voxels);
    //     }

    //     const AABB& aabb = GlobalVariables::getAABB(aabb_index);
    //     new_node.serialize(getNodeFilePath(aabb));

    //     delete(to_store);
    //     to_store = nullptr;
    // };

    const CPUFallbackCache::Entry to_store = CPUFallbackCache::Entry(node);
    const OctreeNodeSerializable& stored_node = to_store.serializable_node;
    if(to_store.serializable_points.has_value()){
        const ChunkSerializable& serializable = to_store.serializable_points.value();
        serializable.serialize(stored_node.points);
    }
    if(to_store.serializable_voxels.has_value()){
        const ChunkSerializable& serializable = to_store.serializable_voxels.value();
        serializable.serialize(stored_node.voxels);
    }
    const AABB& aabb = GlobalVariables::getAABB(to_store.serializable_node.aabb_index);
    stored_node.serialize(getNodeFilePath(aabb));
}

void OctreeNodeSerializable::serializeV2(const COctreeNode* node, 
    const std::vector<CPoint>& points, 
    const std::vector<CPoint>& voxels
){
    const CPUFallbackCache::Entry to_store = CPUFallbackCache::Entry(node, points, voxels);
    const OctreeNodeSerializable& stored_node = to_store.serializable_node;
    if(to_store.serializable_points.has_value()){
        const ChunkSerializable& serializable = to_store.serializable_points.value();
        serializable.serialize(stored_node.points);
    }
    if(to_store.serializable_voxels.has_value()){
        const ChunkSerializable& serializable = to_store.serializable_voxels.value();
        serializable.serialize(stored_node.voxels);
    }
    stored_node.serialize(getNodeFilePathV2(node->aabb_index));
}

void OctreeNodeSerializable::serialize(const std::string& filepath) const {
    std::lock_guard<std::mutex> lock(GlobalVariables::mainLoopIsTerminatingMtx);

    std::ofstream file(filepath, std::ios::binary | std::ios::trunc);

    if (!file.is_open()) {
        println("Failed to open the file {} to serialize an octree node", filepath);
        if(!GlobalVariables::mainLoopIsTerminating){
            throw(EXIT_FAILURE);
        }
    }

    // Write fixed-size members
    file.write(reinterpret_cast<const char*>(&points_counter), sizeof(points_counter));
    file.write(reinterpret_cast<const char*>(&voxels_counter), sizeof(voxels_counter));
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
    file.write(reinterpret_cast<const char*>(&aabb.mins.x), sizeof(float));
    file.write(reinterpret_cast<const char*>(&aabb.mins.y), sizeof(float));
    file.write(reinterpret_cast<const char*>(&aabb.mins.z), sizeof(float));
    file.write(reinterpret_cast<const char*>(&aabb.maxs.x), sizeof(float));
    file.write(reinterpret_cast<const char*>(&aabb.maxs.y), sizeof(float));
    file.write(reinterpret_cast<const char*>(&aabb.maxs.z), sizeof(float));

    file.close();
}

OctreeNodeSerializable OctreeNodeSerializable::deserialize(const std::string& filepath) {
    OctreeNodeSerializable new_node = {};

    std::lock_guard<std::mutex> lock(GlobalVariables::mainLoopIsTerminatingMtx);
    
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        println("Failed to open the file {} to deserialize an octree node", filepath);
        if(!GlobalVariables::mainLoopIsTerminating){
            throw(EXIT_FAILURE);
        }
    }

    // Read fixed-size members
    file.read(reinterpret_cast<char*>(&new_node.points_counter), sizeof(new_node.points_counter));
    file.read(reinterpret_cast<char*>(&new_node.voxels_counter), sizeof(new_node.voxels_counter));
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
    file.read(reinterpret_cast<char*>(&new_node.aabb.mins.x), sizeof(float));
    file.read(reinterpret_cast<char*>(&new_node.aabb.mins.y), sizeof(float));
    file.read(reinterpret_cast<char*>(&new_node.aabb.mins.z), sizeof(float));
    file.read(reinterpret_cast<char*>(&new_node.aabb.maxs.x), sizeof(float));
    file.read(reinterpret_cast<char*>(&new_node.aabb.maxs.y), sizeof(float));
    file.read(reinterpret_cast<char*>(&new_node.aabb.maxs.z), sizeof(float));

    return new_node;
}


OctreeNode* OctreeNodeSerializable::toOctreeNode(const IdAABB& root_aabb_index){
    // std::lock_guard<std::mutex> lock_cache(GlobalVariables::cpuCache->mutex);

    // // TODO: rework cache version
    // const CPUFallbackCache::Entry* to_store = nullptr;
    // // Check if the node is in CPU cache
    // const CPUFallbackCache::Entry* entry = GlobalVariables::cpuCache->get(root_aabb_index);
    // if(!entry){
    //     // If the node is not in CPU cache, load it from disk
    //     CPUFallbackCache::Entry* new_entry = new CPUFallbackCache::Entry();
    //     *new_entry = CPUFallbackCache::Entry::deserialize(root_aabb_index);
    //     entry = new_entry;
    //     to_store = GlobalVariables::cpuCache->add(entry);
    // }
    // OctreeNode* root = entry->toLeafNode();

    // // Store node if removed from cache
    // std::lock_guard<std::mutex> lock(GlobalVariables::mainLoopIsTerminatingMtx);
    // if(to_store){
    //     const IdAABB& aabb_index = to_store->serializable_node.aabb_index;
    //     const OctreeNodeSerializable& new_node = to_store->serializable_node;
    //     if(to_store->serializable_points.has_value()){
    //         const ChunkSerializable& serializable = to_store->serializable_points.value();
    //         serializable.serialize(new_node.points);
    //     }
    //     if(to_store->serializable_voxels.has_value()){
    //         const ChunkSerializable& serializable = to_store->serializable_voxels.value();
    //         serializable.serialize(new_node.voxels);
    //     }

    //     const AABB& aabb = GlobalVariables::getAABB(aabb_index);
    //     new_node.serialize(getNodeFilePath(aabb));

    //     delete(to_store);
    //     to_store = nullptr;
    // };
    // return root;

    CPUFallbackCache::Entry loaded_entry = CPUFallbackCache::Entry::deserialize(root_aabb_index);
    return loaded_entry.toLeafNode();
}




void storeOctree(const OctreeNode* node){
    std::lock_guard<std::mutex> lock(GlobalVariables::aabbMutexMap[node->aabb_index]);
    OctreeNodeSerializable::serialize(node);
}

OctreeNode* loadOctree(const IdAABB& root_aabb_index){
    std::lock_guard<std::mutex> lock(GlobalVariables::aabbMutexMap[root_aabb_index]);
    return OctreeNodeSerializable::toOctreeNode(root_aabb_index);
}


/// Add nodes to cache after octree update
void updateUpdatesCache(OctreeNode* root_octree){
	if(!root_octree){return;}

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
                    // delete(cur_node->children[child_id]);
                    MemoryAllocator::delOctreeNode(cur_node->children[child_id]);
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