#pragma once

#include "./GpuVersionInterface.h"
#include "./GpuVersionStructs.h"

enum AllocatorId {
    ChunkAllocator,
    OccupancyGridAllocator,
    OctreeNodeAllocator,
};



/// A pre-allocated pool of elements
template <typename T>
struct CAllocatorPool {
    /// An entry in the double linked list of pre-allocated elements
    struct Entry {
        /// True if the entry is available for use
        bool is_free = true;
        /// A pointer to its inner element
        T* value = nullptr;
    };

    /// The pool capacity
    const uint32_t CAPACITY;
    const AllocatorId ALLOCATOR_ID;

    /// The actual number of allocated elements
    uint32_t nb_allocated_elements = 0;
    /// A pointer to the contiguous pre-allocated memory
    T* allocated_memory = nullptr;

    /// The double linked list of pre-allocated elements
    /// The last element of the list should always be available (unless the list is full)
    CDoubleLinkedList<Entry*> elements = {};
    /// A map to easily find the next free entry
	CHashMap<T*, typename CDoubleLinkedList<Entry*>::Iterator*> elements_map = {};

    CAllocatorPool(uint32_t capacity, AllocatorId id) : CAPACITY(capacity), ALLOCATOR_ID(id){}

    /// Create a new entry
    /// Pick the first available element
    T* allocate(bool auto_sync = true) {
        // TODO: for now just crash if list is full
        if(nb_allocated_elements == CAPACITY){
            switch(ALLOCATOR_ID){
                case ChunkAllocator:
                    printf("ERROR: can't allocate more `Chunk' elements\n");
                    break;
                case OccupancyGridAllocator:
                    printf("ERROR: can't allocate more `OccupancyGrid' elements\n");
                    break;
                case OctreeNodeAllocator:
                    printf("ERROR: can't allocate more `OctreeNode' elements\n");
                    break;
            }
            // __trap();
        }

        nb_allocated_elements++;

        // auto lock = auto_sync ? std::unique_lock<std::mutex>(mtx) : std::unique_lock<std::mutex>();

        // Get the first free element of the list
        typename CDoubleLinkedList<Entry*>::Iterator* list_it = elements.end(); 
        Entry* entry = list_it->value;
        if(!entry->is_free){
            printf("ERROR: the last element of an allocator should be free if the allocator is not full\n");
            // __trap();
        }
        entry->is_free = false;

        // Update the element position in the list
        // No need to update the map as the iterator pointer is unchanged
        elements.moveBegin(list_it);
        new (entry->value) T();

        return entry->value;
    }


    /// Free an existing entry
    void deallocate(T* entry_id, bool auto_sync = true){
        if(!entry_id){return;}

        // TODO: for now just crash if list is empty
        if(nb_allocated_elements == 0){
            switch(ALLOCATOR_ID){
                case ChunkAllocator:
                    printf("ERROR: can't deallocate empty `Chunk' elements\n");
                    break;
                case OccupancyGridAllocator:
                    printf("ERROR: can't deallocate empty `OccupancyGrid' elements\n");
                    break;
                case OctreeNodeAllocator:
                    printf("ERROR: can't deallocate empty `OctreeNode' elements\n");
                    break;
            }
            // __trap();
        }

        nb_allocated_elements--;

        // auto lock = auto_sync ? std::unique_lock<std::mutex>(mtx) : std::unique_lock<std::mutex>();

        typename CDoubleLinkedList<Entry*>::Iterator** it = elements_map.find(entry_id);
        if(!it){
            printf("ERROR: can't deallocate an unknown element\n");
            // __trap();
        }
        // Reset the element
        entry_id->~T();

        typename CDoubleLinkedList<Entry*>::Iterator* list_it = *it;
        Entry* entry = list_it->value;
        if(entry->is_free){
            printf("ERROR: double free of `%d' element %p\n", ALLOCATOR_ID, (void*)entry_id);
            // __trap();
        }
        entry->is_free = true;

        // Remove the element and put it in the back
        // No need to update the map as the iterator pointer is unchanged
        elements.moveEnd(list_it);
    }

};




struct CMemoryAllocator {
    /// The CChunk allocator
    CAllocatorPool<CChunk>* chunksAllocator = nullptr;
    /// The Occupancy grid allocator
    CAllocatorPool<COccupancyGrid>* gridsAllocator = nullptr;
    /// The Node allocator
    CAllocatorPool<COctreeNode>* nodesAllocator = nullptr;

    ////////////////////////////////////////////////////////////
    ////////////////////////// CHUNKS //////////////////////////
    ////////////////////////////////////////////////////////////

    /// Allocate a new chunk
    CChunk* newChunk(){
        return chunksAllocator->allocate();
    }

    /// Allocate a new chunk and make a copy of the given chunk
    CChunk* newChunkCpy(const CChunk* cpy){
        CChunk* chunk = newChunk();
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
    void delChunk(CChunk* chunk){
        if(!chunk){return;}

        // Get the chunks to deallocate
        CDoubleLinkedList<CChunk*> to_destroy = {};
        to_destroy.init();
        CChunk* cur_chunk = chunk;
        while(cur_chunk){
            to_destroy.pushBack(cur_chunk);
            cur_chunk = cur_chunk->next;
        }

        // std::lock_guard<std::mutex> lock(chunksAllocator->mtx);
        CDoubleLinkedList<CChunk*>::Iterator* it = to_destroy.begin();
        while(it != nullptr){
            chunksAllocator->deallocate(it->value, false);
            it = it->next;
        }
    }



    ////////////////////////////////////////////////////////////
    ////////////////////////// GRIDS ///////////////////////////
    ////////////////////////////////////////////////////////////
    
    /// Allocate a new occupancy grid
    COccupancyGrid* newOccupancyGrid(){
        return gridsAllocator->allocate();
    }

    /// Allocate a new occupancy grid and make a copy of the given occupancy grid
    COccupancyGrid* newOccupancyGridCpy(const COccupancyGrid* cpy){
        COccupancyGrid* grid = newOccupancyGrid();
        for(uint32_t i=0; i<OocSimLodSettings::GRID_SIZE / 32; i++){
            grid->values[i] = cpy->values[i];
        }
        return grid;
    }

    /// Deallocate an occupancy grid
    void delOccupancyGrid(COccupancyGrid* grid){
        gridsAllocator->deallocate(grid);
        grid = nullptr;
    }



    ////////////////////////////////////////////////////////////
    ////////////////////////// NODES ///////////////////////////
    ////////////////////////////////////////////////////////////
    
    /// Allocate a new node
    COctreeNode* newOctreeNode(CIdAABB aabb_index){
        COctreeNode* node = nodesAllocator->allocate();
        node->aabb_index = aabb_index;
        return node;
    }

    /// Allocate a new node and make a copy of the given node
    COctreeNode* newOctreeNodeCpy(const COctreeNode* cpy, bool node_only = false){
        COctreeNode* node = newOctreeNode(cpy->aabb_index);

        node->children_ids = cpy->children_ids;
        node->counter = cpy->counter;
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
    COctreeNode* newOctreeNodePartialCpy(const COctreeNode* cpy, const COctreeNode* partial){
        if(!partial){return newOctreeNodeCpy(cpy);}

        COctreeNode* node = newOctreeNode(cpy->aabb_index);

        node->children_ids = cpy->children_ids;
        node->counter = cpy->counter;
        node->points = cpy->points ? newChunkCpy(cpy->points) : nullptr;
        node->voxels = cpy->voxels ? newChunkCpy(cpy->voxels) : nullptr;
        node->occupancy = cpy->occupancy ? newOccupancyGridCpy(cpy->occupancy) : nullptr;

        for(uint32_t child = 0; child < 8; child++){
            const COctreeNode* next_partial = partial->children[child] ? partial->children[child] : nullptr;
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
    void delOctreeNode(COctreeNode* node){
        if(!node){return;}

        CDoubleLinkedList<CChunk*> chunks_to_delete = {};
        chunks_to_delete.init();
        CDoubleLinkedList<COccupancyGrid*> grids_to_delete = {};
        grids_to_delete.init();
        CDoubleLinkedList<COctreeNode*> nodes_to_delete = {};
        nodes_to_delete.init();

        CDoubleLinkedList<COctreeNode*> to_visit = {};
        while(!to_visit.isEmpty()){
            COctreeNode* cur_node = *to_visit.front();
            to_visit.popFront();

            nodes_to_delete.pushBack(cur_node);
            if(cur_node->points){
                CChunk* cur_chunk = cur_node->points;
                while(cur_chunk){
                    chunks_to_delete.pushBack(cur_chunk);
                    cur_chunk = cur_chunk->next;
                }
            }
            if(cur_node->voxels){
                CChunk* cur_chunk = cur_node->voxels;
                while(cur_chunk){
                    chunks_to_delete.pushBack(cur_chunk);
                    cur_chunk = cur_chunk->next;
                }
            }
            if(cur_node->occupancy){
                grids_to_delete.pushBack(cur_node->occupancy);
            }
            for(uint32_t i=0; i<8; i++){
                if(cur_node->children[i]){
                    to_visit.pushBack(cur_node->children[i]);
                }
            }
        }

        // Delete chunks
        {
            // std::lock_guard<std::mutex> lock(chunksAllocator->mtx);
            CDoubleLinkedList<CChunk*>::Iterator* it = chunks_to_delete.begin();
            while(it != nullptr){
                chunksAllocator->deallocate(it->value, false);
                it = it->next;
            }
        }

        // Delete grids
        {
            // std::lock_guard<std::mutex> lock(gridsAllocator->mtx);
            CDoubleLinkedList<COccupancyGrid*>::Iterator* it = grids_to_delete.begin();
            while(it != nullptr){
                gridsAllocator->deallocate(it->value, false);
                it = it->next;
            }
        }

        // Delete nodes
        {
            // std::lock_guard<std::mutex> lock(nodesAllocator->mtx);
            CDoubleLinkedList<COctreeNode*>::Iterator* it = nodes_to_delete.begin();
            while(it != nullptr){
                nodesAllocator->deallocate(it->value, false);
                it = it->next;
            }
        }
    }

};