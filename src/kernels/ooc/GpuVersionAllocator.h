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
    CDoubleLinkedList<Entry*>* elements = nullptr;
    /// A map to easily find the next free entry
	CHashMap<T*, typename CDoubleLinkedList<Entry*>::Iterator*>* elements_map = nullptr;

    CAllocatorPool(uint32_t capacity, AllocatorId id) : CAPACITY(capacity), ALLOCATOR_ID(id){}





    /// A temporary allocation counter
    /// This should be reset before and after each kernel launch
    /// The reset functions should be call a single time
    uint32_t tmp_allocation_counter = 0;
    uint32_t tmp_deallocation_counter = 0;
    /// An array of pointers to the deallocated elements
    typename CDoubleLinkedList<Entry*>::Iterator** deallocated_memory = nullptr;

#ifdef __CUDACC__
    __device__ void reset_temporary_allocations(){
        uint32_t counter = __nv_atomic_load_n(&tmp_allocation_counter, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        if(counter == 0){return;}
        // Update the element position in the list
        // No need to update the map as the iterator pointer is unchanged
        typename CDoubleLinkedList<Entry*>::Iterator* list_it = elements->end();
        for(uint32_t i=0; i<(counter-1); i++){
            list_it = list_it->prev;
        }
        elements->moveBeginWithNexts(list_it);

        __nv_atomic_sub(&tmp_allocation_counter, counter, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    }

    __device__ void reset_temporary_deallocations(){
        uint32_t counter = __nv_atomic_load_n(&tmp_deallocation_counter, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        if(counter == 0){return;}
        
        // Remove the element and put it in the back
        // No need to update the map as the iterator pointer is unchanged
        for(uint32_t i=0; i<counter; i++){
            typename CDoubleLinkedList<Entry*>::Iterator* list_it = deallocated_memory[i];
            elements->moveEnd(list_it);
        }

        __nv_atomic_sub(&tmp_deallocation_counter, counter, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    }
    




    /// Create a new entry
    /// Pick the first available element
    __device__ T* allocate(bool will_run_in_parallel) {
        // TODO: for now just crash if list is full
        uint32_t new_counter = __nv_atomic_fetch_add(&nb_allocated_elements, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        new_counter++;
        if(new_counter >= CAPACITY){
            switch(ALLOCATOR_ID){
                case ChunkAllocator:
                    printf("ERROR: can't allocate more `Chunk' elements\n");
                    customAssert();
                case OccupancyGridAllocator:
                    printf("ERROR: can't allocate more `OccupancyGrid' elements\n");
                    customAssert();
                case OctreeNodeAllocator:
                    printf("ERROR: can't allocate more `OctreeNode' elements\n");
                    customAssert();
            }
        }

        // Get the first free element of the list
        typename CDoubleLinkedList<Entry*>::Iterator* list_it = elements->end();
        if(will_run_in_parallel){
            uint32_t counter = __nv_atomic_fetch_add(&tmp_allocation_counter, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            for(uint32_t i=0; i<counter; i++){
                list_it = list_it->prev;
            }
        }

        Entry* entry = list_it->value;
        if(!entry->is_free){
            switch(ALLOCATOR_ID){
                case ChunkAllocator:
                    printf("ERROR `Chunk' allocator: the last element of an allocator should be free if the allocator is not full\n");
                    customAssert();
                case OccupancyGridAllocator:
                    printf("ERROR `OccupancyGrid' allocator: the last element of an allocator should be free if the allocator is not full\n");
                    customAssert();
                case OctreeNodeAllocator:
                    printf("ERROR `OctreeNode' allocator: the last element of an allocator should be free if the allocator is not full\n");
                    customAssert();
            }
        }
        entry->is_free = false;
        new (entry->value) T();

        if(!will_run_in_parallel){
            elements->moveBegin(list_it);
        }

        return entry->value;
    }


    /// Free an existing entry
    __device__ void deallocate(T* entry_id, bool will_run_in_parallel){
        if(!entry_id){return;}

        // TODO: for now just crash if list is empty
        uint32_t old_counter = __nv_atomic_fetch_sub(&nb_allocated_elements, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        if(old_counter == 0 || old_counter > CAPACITY){ // To handle underflow
            switch(ALLOCATOR_ID){
                case ChunkAllocator:
                    printf("ERROR: can't deallocate empty `Chunk' elements\n");
                    customAssert();
                case OccupancyGridAllocator:
                    printf("ERROR: can't deallocate empty `OccupancyGrid' elements\n");
                    customAssert();
                case OctreeNodeAllocator:
                    printf("ERROR: can't deallocate empty `OctreeNode' elements\n");
                    customAssert();
            }
        }

        typename CDoubleLinkedList<Entry*>::Iterator** it = elements_map->find(entry_id);
        if(!it){
            switch(ALLOCATOR_ID){
                case ChunkAllocator:
                    printf("ERROR `Chunk' allocator: can't deallocate an unknown element\n");
                    customAssert();
                case OccupancyGridAllocator:
                    printf("ERROR `OccupancyGrid' allocator: can't deallocate an unknown element\n");
                    customAssert();
                case OctreeNodeAllocator:
                    printf("ERROR `OctreeNode' allocator: can't deallocate an unknown element\n");
                    customAssert();
            }
        }
        // Reset the element
        entry_id->~T();

        typename CDoubleLinkedList<Entry*>::Iterator* list_it = *it;
        Entry* entry = list_it->value;
        if(entry->is_free){
            switch(ALLOCATOR_ID){
                case ChunkAllocator:
                    printf("ERROR `Chunk' allocator: double free of element %p\n", (void*)entry_id);
                    customAssert();
                case OccupancyGridAllocator:
                    printf("ERROR `OccupancyGrid' allocator: double free of element %p\n", (void*)entry_id);
                    customAssert();
                case OctreeNodeAllocator:
                    printf("ERROR `OctreeNode' allocator: double free of element %p\n", (void*)entry_id);
                    customAssert();
            }
        }
        entry->is_free = true;

        if(will_run_in_parallel){
            uint32_t counter = __nv_atomic_fetch_add(&tmp_deallocation_counter, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            deallocated_memory[counter] = list_it;
        } else {
            elements->moveEnd(list_it);
        }
    }
#endif // __CUDACC__

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

#ifdef __CUDACC__

    /// Allocate a new chunk
    __device__ CChunk* newChunk(bool will_run_in_parallel){
        return chunksAllocator->allocate(will_run_in_parallel);
    }

    /// Deallocate a chunk and all it's children
    __device__ void delChunk(CChunk* chunk, bool will_run_in_parallel){
        if(!chunk){return;}

        // Get the chunks to deallocate
        CChunk* cur_chunk = chunk;
        while(cur_chunk){
            CChunk* next = cur_chunk->next;
            cur_chunk->next = nullptr;
            chunksAllocator->deallocate(cur_chunk, will_run_in_parallel);
            cur_chunk = next;
        }
    }



    ////////////////////////////////////////////////////////////
    ////////////////////////// GRIDS ///////////////////////////
    ////////////////////////////////////////////////////////////
    
    /// Allocate a new occupancy grid
    __device__ COccupancyGrid* newOccupancyGrid(bool will_run_in_parallel){
        return gridsAllocator->allocate(will_run_in_parallel);
    }

    /// Deallocate an occupancy grid
    __device__ void delOccupancyGrid(COccupancyGrid* grid, bool will_run_in_parallel){
        if(!grid){return;}
        gridsAllocator->deallocate(grid, will_run_in_parallel);
    }



    ////////////////////////////////////////////////////////////
    ////////////////////////// NODES ///////////////////////////
    ////////////////////////////////////////////////////////////
    
    /// Allocate a new node
    __device__ COctreeNode* newOctreeNode(CIdAABB aabb_index, bool will_run_in_parallel){
        COctreeNode* node = nodesAllocator->allocate(will_run_in_parallel);
        node->aabb_index = aabb_index;
        return node;
    }


    /// Deallocate a node
    __device__ void delOctreeNode(COctreeNode* node, bool node_only, bool will_run_in_parallel){
        if(!node){return;}

        delChunk(node->points, will_run_in_parallel);
        node->points = nullptr;
        delChunk(node->voxels, will_run_in_parallel);
        node->voxels = nullptr;
        delOccupancyGrid(node->occupancy, will_run_in_parallel);
        node->occupancy = nullptr;

        for(uint32_t i=0; i<8; i++){
            if(!node_only){delOctreeNode(node->children[i], false, will_run_in_parallel);}
            node->children[i] = nullptr;
        }
    }
    
#endif // __CUDACC__

};