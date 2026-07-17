#pragma once

#include "globals.h"
#include "kernels/ooc/GpuVersionStructs.h"


/// A pre-allocated pool of elements
template <typename T>
struct AllocatorPool {
    /// An entry in the double linked list of pre-allocated elements
    struct Entry {
        /// True if the entry is available for use
        bool is_free = true;
        /// A pointer to its inner element
        T* value = nullptr;
    };

    /// The pool capacity
    const uint32_t CAPACITY;
    /// The actual number of allocated elements
    std::atomic<uint32_t> nb_allocated_elements = 0;
    /// A pointer to the contiguous pre-allocated memory
    T* allocated_memory = nullptr;

    /// The double linked list of pre-allocated elements
    /// The last element of the list should always be available (unless the list is full)
    CDoubleLinkedList<std::shared_ptr<Entry>> elements = {};
    /// A map to easily find the next free entry
	CHashMap<T*, typename CDoubleLinkedList<std::shared_ptr<Entry>>::Iterator*> elements_map = {};


    /// For stats
    std::atomic<uint64_t> total_nb_allocation = 0;
    std::atomic<uint64_t> total_nb_deallocation = 0;
    uint64_t total_allocated_size = 0;

    /// To synchronise allocations
    std::mutex mtx;


    /// A basic constructor with its corresponding capacity
    AllocatorPool(uint32_t capacity) : CAPACITY(capacity) {
        allocated_memory = static_cast<T*>(::operator new[](CAPACITY * sizeof(T), std::align_val_t(alignof(T))));
        total_allocated_size = CAPACITY * sizeof(T);

        elements.init();
        elements_map.init(CAPACITY);

        for (uint32_t i = 0; i < CAPACITY; i++) {
            std::shared_ptr<Entry> entry = std::make_shared<Entry>();

            entry->value = allocated_memory + i;

            elements.pushFront(entry);
            elements_map[entry->value] = elements.begin();
        }
    }

    ~AllocatorPool() {
        typename CDoubleLinkedList<std::shared_ptr<Entry>>::Iterator* loop_it = elements.begin();
        while(loop_it){
            std::shared_ptr<Entry>& entry = loop_it->value;
            if(!entry->is_free){entry->value->~T();}
            loop_it = loop_it->next;
        }
        ::operator delete[](allocated_memory, std::align_val_t(alignof(T)));
    }


    uint64_t getSize() const {
        return total_allocated_size 
            + elements.size * sizeof(Entry)
            + elements_map.capacity * (sizeof(T*) + sizeof(typename CDoubleLinkedList<std::shared_ptr<Entry>>::Iterator*))
            + sizeof(AllocatorPool<T>);
    }


    /// Create a new entry
    /// Pick the first available element
    T* allocate(bool auto_sync = true) {
        // TODO: for now just crash if list is full
        if(nb_allocated_elements.load() == CAPACITY){
            println("ERROR: can't allocate more `{}' elements", typeid(T).name());
            throw(EXIT_FAILURE);
        }

        nb_allocated_elements++;
        total_nb_allocation++;

        auto lock = auto_sync ? std::unique_lock<std::mutex>(mtx) : std::unique_lock<std::mutex>();

        // Get the first free element of the list
        std::shared_ptr<Entry> entry = *elements.back();
        if(!entry->is_free){
            println("ERROR: this should not be possible");
            throw(EXIT_FAILURE);
        }
        entry->is_free = false;

        // Update the element position in the list and in the map
        elements_map.erase(entry->value);
        elements.popBack();
        elements.pushFront(entry);
        elements_map[entry->value] = elements.begin();

        new (entry->value) T();

        return entry->value;
    }


    /// Free an existing entry
    void deallocate(T* entry_id, bool auto_sync = true){
        if(!entry_id){return;}

        // TODO: for now just crash if list is empty
        if(nb_allocated_elements == 0){
            println("ERROR: can't deallocate empty `{}' elements", typeid(T).name());
            throw(EXIT_FAILURE);
        }

        nb_allocated_elements--;
        total_nb_deallocation++;

        auto lock = auto_sync ? std::unique_lock<std::mutex>(mtx) : std::unique_lock<std::mutex>();

        typename CDoubleLinkedList<std::shared_ptr<Entry>>::Iterator** it = elements_map.find(entry_id);
        if(!it){
            println("ERROR: can't deallocate an unknown `{}' element", typeid(T).name());
            throw(EXIT_FAILURE);
        }

        // Reset the element
        entry_id->~T();

        typename CDoubleLinkedList<std::shared_ptr<Entry>>::Iterator* list_it = *it;
        std::shared_ptr<Entry> real_entry = list_it->value;
        if(real_entry->is_free){
            println("ERROR: double free of `{}' element {}", typeid(T).name(), (void*)entry_id);
            throw(EXIT_FAILURE);
        }
        real_entry->is_free = true;

        // Remove the element and put it in the back
        elements_map.erase(entry_id);
        elements.erase(list_it);
        delete(list_it);
        elements.pushBack(real_entry);
        elements_map[entry_id] = elements.end();
    }

    void displayInfo() {
        std::lock_guard<std::mutex> lock(mtx);
        uint32_t nb_allocated = nb_allocated_elements.load();
        uint32_t total_allocation = total_nb_allocation.load();
        uint32_t total_deallocation = total_nb_deallocation.load();

        uint32_t real_nb_free = 0;

        typename CDoubleLinkedList<std::shared_ptr<Entry>>::Iterator* loop_it = elements.begin();
        while(loop_it){
            const std::shared_ptr<Entry>& entry = loop_it->value;
            real_nb_free += entry->is_free ? 1 : 0;
            loop_it = loop_it->next;
        }

        println("    - memory consumption: {}", GlobalVariables::formatMemSize(total_allocated_size));
        println("    - capacity: {}, available: {}, used: {}", CAPACITY, CAPACITY - nb_allocated, nb_allocated);
        println("    - really used: {}, really available: {}", CAPACITY - real_nb_free, real_nb_free);
        println("    - total nb allocations: {}, total nb deallocation: {}", total_allocation, total_deallocation);
    }
};




struct MemoryAllocator {
    /// The Chunk allocator
    static inline std::shared_ptr<AllocatorPool<Chunk>> chunksAllocator = nullptr;
    /// The Occupancy grid allocator
    static inline std::shared_ptr<AllocatorPool<OccupancyGrid>> gridsAllocator = nullptr;
    /// The Node allocator
    static inline std::shared_ptr<AllocatorPool<OctreeNode>> nodesAllocator = nullptr;

    static void init() {
        chunksAllocator = std::make_shared<AllocatorPool<Chunk>>(OocSimLodSettings::NB_ALLOCATED_CHUNKS);
        gridsAllocator = std::make_shared<AllocatorPool<OccupancyGrid>>(OocSimLodSettings::NB_ALLOCATED_GRIDS);
        nodesAllocator = std::make_shared<AllocatorPool<OctreeNode>>(OocSimLodSettings::NB_ALLOCATED_NODES);
    }
    static uint64_t getSize(){
        return chunksAllocator->getSize() + gridsAllocator->getSize() + nodesAllocator->getSize();
    }
    static void displayInfo() {
        println("///////////////////////////////////////////////////");
        println("//////////////////// Allocator ////////////////////");
        println("///////////////////////////////////////////////////\n");

        println("Chunks:");
        chunksAllocator->displayInfo();

        println("\nOccupancy grids:");
        gridsAllocator->displayInfo();

        println("\nNodes:");
        nodesAllocator->displayInfo();

        println("\nTotal Memory Usage: {}", GlobalVariables::formatMemSize(getSize()));

        println("\n///////////////////////////////////////////////////");
        println("///////////////////////////////////////////////////");
        println("///////////////////////////////////////////////////\n");
    }

    ////////////////////////////////////////////////////////////
    ////////////////////////// CHUNKS //////////////////////////
    ////////////////////////////////////////////////////////////

    /// Allocate a new chunk
    static Chunk* newChunk(){
        return chunksAllocator->allocate();
    }

    /// Allocate a new chunk and make a copy of the given chunk
    static Chunk* newChunkCpy(const Chunk* cpy){
        Chunk* chunk = newChunk();
        chunk->size = cpy->size;
        for(uint32_t i=0; i<OocSimLodSettings::NB_POINTS_PER_CHUNK; i++){
            chunk->points[i] = cpy->points[i];
        }
        if(cpy->next){
            chunk->next = newChunkCpy(cpy->next);
        }
        return chunk;
    }

    /// Deallocate a chunk and all it's children
    static void delChunk(Chunk* chunk){
        if(!chunk){return;}

        // Get the chunks to deallocate
        std::vector<Chunk*> to_destroy = {};
        Chunk* cur_chunk = chunk;
        while(cur_chunk){
            to_destroy.push_back(cur_chunk);
            cur_chunk = cur_chunk->next;
        }

        std::lock_guard<std::mutex> lock(chunksAllocator->mtx);
        std::for_each(to_destroy.begin(), to_destroy.end(), [&](Chunk* cur_chunk){
            chunksAllocator->deallocate(cur_chunk, false);
        });
    }



    ////////////////////////////////////////////////////////////
    ////////////////////////// GRIDS ///////////////////////////
    ////////////////////////////////////////////////////////////
    
    /// Allocate a new occupancy grid
    static OccupancyGrid* newOccupancyGrid(){
        return gridsAllocator->allocate();
    }

    /// Allocate a new occupancy grid and make a copy of the given occupancy grid
    static OccupancyGrid* newOccupancyGridCpy(const OccupancyGrid* cpy){
        OccupancyGrid* grid = newOccupancyGrid();
        for(uint32_t i=0; i<OocSimLodSettings::GRID_SIZE / 32; i++){
            grid->values[i] = cpy->values[i].load();
        }
        return grid;
    }

    /// Deallocate an occupancy grid
    static void delOccupancyGrid(OccupancyGrid* grid){
        gridsAllocator->deallocate(grid);
        grid = nullptr;
    }



    ////////////////////////////////////////////////////////////
    ////////////////////////// NODES ///////////////////////////
    ////////////////////////////////////////////////////////////
    
    /// Allocate a new node
    static OctreeNode* newOctreeNode(IdAABB aabb_index){
        OctreeNode* node = nodesAllocator->allocate();
        node->aabb_index = aabb_index;
        return node;
    }

    /// Allocate a new node and make a copy of the given node
    static OctreeNode* newOctreeNodeCpy(const OctreeNode* cpy, bool node_only = false){
        OctreeNode* node = newOctreeNode(cpy->aabb_index);

        node->children_ids = cpy->children_ids;
        node->counter.store(cpy->counter.load());
        node->points = cpy->points ? newChunkCpy(cpy->points) : nullptr;
        node->voxels = cpy->voxels ? newChunkCpy(cpy->voxels) : nullptr;
        node->occupancy = cpy->occupancy ? newOccupancyGridCpy(cpy->occupancy) : nullptr;

        if(!node_only){
            for(uint32_t child = 0; child < 8; child++){
                if(cpy->children[child]){
                    node->children[child] = newOctreeNodeCpy(cpy->children[child]);
                }
            }
        }

        return node;
    }


    /// Allocate a new node and make a copy of the given node
    /// The new node will keep children of the partial node if the copy doesn't introduce new children
    static OctreeNode* newOctreeNodePartialCpy(const OctreeNode* cpy, const OctreeNode* partial){
        if(!partial){return newOctreeNodeCpy(cpy);}

        OctreeNode* node = newOctreeNode(cpy->aabb_index);

        node->children_ids = cpy->children_ids;
        node->counter.store(cpy->counter.load());
        node->points = cpy->points ? newChunkCpy(cpy->points) : nullptr;
        node->voxels = cpy->voxels ? newChunkCpy(cpy->voxels) : nullptr;
        node->occupancy = cpy->occupancy ? newOccupancyGridCpy(cpy->occupancy) : nullptr;

        for(uint32_t child = 0; child < 8; child++){
            const OctreeNode* next_partial = partial->children[child] ? partial->children[child] : nullptr;
            if(cpy->children[child]){
                if(next_partial){
                    node->children[child] = newOctreeNodePartialCpy(cpy->children[child], next_partial);
                } else {
                    node->children[child] = newOctreeNodeCpy(cpy->children[child]);
                }
            } else if(next_partial) {
                node->children[child] = newOctreeNodeCpy(next_partial);
            }
        }

        return node;
    }


    /// Deallocate a node
    static void delOctreeNode(OctreeNode* node){
        if(!node){return;}

        std::vector<Chunk*> chunks_to_delete = {};
        std::vector<OccupancyGrid*> grids_to_delete = {};
        std::vector<OctreeNode*> nodes_to_delete = {};

        std::function<void(OctreeNode*)> get_pointers = [&](OctreeNode* cur_node){
            if(!cur_node){return;}
            nodes_to_delete.push_back(cur_node);

            if(cur_node->points){
                Chunk* cur_chunk = cur_node->points;
                while(cur_chunk){
                    chunks_to_delete.push_back(cur_chunk);
                    cur_chunk = cur_chunk->next;
                }
            }
            if(cur_node->voxels){
                Chunk* cur_chunk = cur_node->voxels;
                while(cur_chunk){
                    chunks_to_delete.push_back(cur_chunk);
                    cur_chunk = cur_chunk->next;
                }
            }
            if(cur_node->occupancy){
                grids_to_delete.push_back(cur_node->occupancy);
            }
            for(uint32_t i=0; i<8; i++){
                get_pointers(cur_node->children[i]);
            }
        };

        get_pointers(node);

        // Delete chunks
        {
            std::lock_guard<std::mutex> lock(chunksAllocator->mtx);
            std::for_each(chunks_to_delete.begin(), chunks_to_delete.end(), [&](Chunk* cur_chunk){
                chunksAllocator->deallocate(cur_chunk, false);
            });
        }

        // Delete grids
        {
            std::lock_guard<std::mutex> lock(gridsAllocator->mtx);
            std::for_each(grids_to_delete.begin(), grids_to_delete.end(), [&](OccupancyGrid* cur_grid){
                gridsAllocator->deallocate(cur_grid, false);
            });
        }

        // Delete nodes
        {
            std::lock_guard<std::mutex> lock(nodesAllocator->mtx);
            std::for_each(nodes_to_delete.begin(), nodes_to_delete.end(), [&](OctreeNode* cur_node){
                nodesAllocator->deallocate(cur_node, false);
            });
        }
    }

};