#pragma once

#include "./GpuVersionInterface.h"
#include "./GpuVersionStructs.h"
#include "./GpuVersionGlobals.h"

enum AllocatorId {
    ChunkAllocator,
    OccupancyGridAllocator,
    OctreeNodeAllocator,
};



/// A pre-allocated pool of elements
template <typename T>
struct CAllocatorPool {
    /// The pool capacity
    const uint32_t CAPACITY;
    const AllocatorId ALLOCATOR_ID;

    /// A pointer to the contiguous pre-allocated memory
    T* allocated_memory = nullptr;
    /// The stack of free indices
    uint32_t* free_stack;
    /// The atomic pointer to the top of the stack 
    uint32_t free_top;

    /// The actual number of allocated elements
    uint32_t nb_allocated_elements = 0;
    
#ifdef __CUDACC__
    // Pop from stack
    __device__ T* allocate() {
        uint32_t stack_pos = __nv_atomic_fetch_sub(&free_top, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        if(stack_pos == 0 || stack_pos > CAPACITY) {
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
        uint32_t idx = free_stack[stack_pos - 1];
        __nv_atomic_add(&nb_allocated_elements, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        return &allocated_memory[idx];
    }

    // Push back into stack
    __device__ void deallocate(T* ptr) {
        uint32_t idx = (uint32_t)(ptr - allocated_memory);
        uint32_t stack_pos = __nv_atomic_fetch_add(&free_top, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        free_stack[stack_pos] = idx;
        __nv_atomic_sub(&nb_allocated_elements, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
    }
#endif //__CUDACC__
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
    __device__ CChunk* newChunk(bool will_update_metrics = true){
        // UI values
        if(will_update_metrics){
            __nv_atomic_add(&globalVariables.nbNewChunksThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.nbTotalNewChunks, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.currentNbChunks, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        }

        CChunk* new_chunk = chunksAllocator->allocate();
        new_chunk->size = 0;
        new_chunk->next = nullptr;
        return new_chunk;
    }

    /// Deallocate a chunk and all it's children
    __device__ void delChunk(CChunk* chunk, bool will_update_metrics = true){
        if(!chunk){return;}

        // Get the chunks to deallocate
        CChunk* cur_chunk = chunk;
        while(cur_chunk){
            CChunk* next = cur_chunk->next;
            cur_chunk->next = nullptr;
            cur_chunk->size = 0;
            chunksAllocator->deallocate(cur_chunk);
            cur_chunk = next;

            // UI values
            if(will_update_metrics){
                __nv_atomic_add(&globalVariables.nbDeletedChunksThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                __nv_atomic_add(&globalVariables.nbTotalDeletedChunks, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
                __nv_atomic_sub(&globalVariables.currentNbChunks, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            }
        }
    }
    __device__ void delChunkSingle(CChunk* chunk, bool will_update_metrics = true){
        if(!chunk){return;}

        // Get the chunks to deallocate
        CChunk* cur_chunk = chunk;
        cur_chunk->next = nullptr;
        cur_chunk->size = 0;
        chunksAllocator->deallocate(cur_chunk);

        // UI values
        if(will_update_metrics){
            __nv_atomic_add(&globalVariables.nbDeletedChunksThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.nbTotalDeletedChunks, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_sub(&globalVariables.currentNbChunks, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        }
    }



    ////////////////////////////////////////////////////////////
    ////////////////////////// GRIDS ///////////////////////////
    ////////////////////////////////////////////////////////////
    
    /// Allocate a new occupancy grid
    __device__ COccupancyGrid* newOccupancyGrid(bool will_update_metrics = true){
        // UI values
        if(will_update_metrics){
            __nv_atomic_add(&globalVariables.nbNewGridsThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.nbTotalNewGrids, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.currentNbGrids, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        }
        return gridsAllocator->allocate();
    }

    /// Deallocate an occupancy grid
    __device__ void delOccupancyGrid(COccupancyGrid* grid, bool will_update_metrics = true){
        if(!grid){return;}

        // UI values
        if(will_update_metrics){
            __nv_atomic_add(&globalVariables.nbDeletedGridsThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.nbTotalDeletedGrids, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_sub(&globalVariables.currentNbGrids, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        }
        gridsAllocator->deallocate(grid);
    }



    ////////////////////////////////////////////////////////////
    ////////////////////////// NODES ///////////////////////////
    ////////////////////////////////////////////////////////////
    
    /// Allocate a new node
    __device__ COctreeNode* newOctreeNode(CIdAABB aabb_index, bool will_update_metrics = true){
        // UI values
        if(will_update_metrics){
            __nv_atomic_add(&globalVariables.nbNewNodesThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.nbTotalNewNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        }
        COctreeNode* node = nodesAllocator->allocate();
        new (node) COctreeNode();
        node->aabb_index = aabb_index;
        return node;
    }


    /// Deallocate a node
    __device__ void delOctreeNode(COctreeNode* node, bool node_only, bool will_update_metrics = true){
        if(!node){return;}

        // UI values
        if(will_update_metrics){
            __nv_atomic_add(&globalVariables.nbDeletedNodesThisUpdate, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
            __nv_atomic_add(&globalVariables.nbTotalDeletedNodes, 1, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);
        }

        delChunk(node->points, will_update_metrics);
        node->points = nullptr;
        delChunk(node->voxels, will_update_metrics);
        node->voxels = nullptr;
        delOccupancyGrid(node->occupancy, will_update_metrics);
        node->occupancy = nullptr;

        for(uint32_t i=0; i<8; i++){
            if(!node_only){
                delOctreeNode(node->children[i], false, will_update_metrics);
            }
            node->children[i] = nullptr;
        }

        node->aabb_index = CINVALID_ID;
        node->points_counter = 0;
        node->voxels_counter = 0;
        node->points_stored = 0;
        node->points_last_stored = 0;
        node->voxels_last_stored = 0;
        node->voxels_stored = 0;
        node->children_ids = 0;
        node->level = 0;
        node->cur_id = 0;
        node->flags = 0;

        nodesAllocator->deallocate(node);
    }
    
#endif // __CUDACC__

};