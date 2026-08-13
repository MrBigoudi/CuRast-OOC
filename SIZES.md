# Memory Footprint Overview

## Baseline Type Sizes

| Type | Size | Align |
|---|---|---|
| `CIdAABB` | 4 b | 4 b |
| `CNodePosition` | 4 b | 4 b |
| `CAABB` | 24 b | 4 b |
| `CPoint` | 16 b | 4 b |
| `COccupancyGrid` | 262 kb 144 b | 4 b |
| `CChunk` | 16 kb 400 b | 8 b |
| `COctreeNode` | 144 b | 8 b |
| `Relationship` | 36 b | 8 b |
| `T*` | 8 b | 4 b |

## Generic Container Formulas

**`CDoubleLinkedList<T>(k elems)`**
```
size = 16b + k * (sizeof(T) + 16b)
```

**`CAllocatorPool<T>(k elems)`**
```
size = 12b
     + elements       = 16b + k * 24b
     + elements_map    = 20b + k * 16b + k * (8b + 8b + 8b + 16b) = 20b + k * 56b
     + allocated memory = k * (sizeof(T) + alignof(T))
     = 48b + k * (80b + sizeof(T) + alignof(T))
```

---

## Default Constants

| Constant | Default Value |
|---|---|
| `MAX_NB_NODES` | 1,000,000 |
| `MAX_NB_NODES_TO_EXCHANGE` | 256 |
| `MAX_POINTS_PER_LEAF` | 65,536 |
| `MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE` | 256 |
| `NB_POINTS_PER_CHUNK` | 1,024 |
| `MAX_BATCHES_PER_OCTREE_UPDATE` | 4 |
| `MAX_POINTS_PER_BATCHES` | 1,000,000 |
| `MAX_NB_RENDERED_POINTS` | 5,000,000 |
| `MAX_NB_RENDERED_VOXELS` | 5,000,000 |
| `LRU_UPDATES_CACHE_SIZE` | 1,024 |
| `LRU_VISIBILITY_CACHE_SIZE` | 1,024 |
| `MAX_NB_SPILLING_POINTS` | 1,000,000 |
| `MAX_NB_BACKLOG_VOXELS` | 5,000,000 |
| `NB_ALLOCABLE_CHUNKS` | 32,768 |
| `NB_ALLOCABLE_GRIDS` | 1,024 |
| `NB_ALLOCABLE_NODES` | 8,192 |

---

## Buffer Sizes

### Core Node Data

| Buffer | Formula | Default | Size |
|---|---|---|---|
| `relationshipMap` | `MAX_NB_NODES * 36b` | 1,000,000 | 36 Mb |
| `nodesFlags` | `MAX_NB_NODES * 4b` | 1,000,000 | 4 Mb |
| `packedNodes` | `MAX_NB_NODES * 8b` | 1,000,000 | 8 Mb |
| `renderingPackedNodes` | `MAX_NB_NODES * 8b` | 1,000,000 | 8 Mb |
| `renderingPackedNodesTmp` | `MAX_NB_NODES * 8b` | 1,000,000 | 8 Mb |
| `gridsToInit` | `MAX_NB_NODES * 8b` | 1,000,000 | 8 Mb |

### Exchange Buffers

| Buffer | Formula | Default | Size |
|---|---|---|---|
| `exchangedAABBIndices` | `MAX_NB_NODES_TO_EXCHANGE * 4b` | 256 | 1 kb |
| `exchangedAABBs` | `MAX_NB_NODES_TO_EXCHANGE * 24b` | 256 | 6 kb |
| `exchangedChildrenIds` | `MAX_NB_NODES_TO_EXCHANGE * 4b` | 256 | 1 kb |
| `exchangedPointsCounters` | `MAX_NB_NODES_TO_EXCHANGE * 4b` | 256 | 1 kb |
| `exchangedVoxelsCounters` | `MAX_NB_NODES_TO_EXCHANGE * 4b` | 256 | 1 kb |
| `exchangedPoints` | `MAX_NB_NODES_TO_EXCHANGE * MAX_POINTS_PER_LEAF * 16b` | 256 * 65,536 | 268 Mb |
| `exchangedVoxels` | `MAX_NB_NODES_TO_EXCHANGE * MAX_NB_VOXELS_CHUNKS_TO_EXCHANGE * NB_POINTS_PER_CHUNK * 16b` | 256 * 256 * 1,024 | 1.073 Gb |

### Batch Buffers

| Buffer | Formula | Default | Size |
|---|---|---|---|
| `batchesAddedMask` | `MAX_BATCHES_PER_OCTREE_UPDATE * 4b` | 4 | 16 b |
| `batchesToAddCounts` | `MAX_BATCHES_PER_OCTREE_UPDATE * 4b` | 4 | 16 b |
| `batchesToAddPoints` | `MAX_BATCHES_PER_OCTREE_UPDATE * MAX_POINTS_PER_BATCHES` | 4 * 1,000,000 | 4 Mb |

### Rendering Buffers

| Buffer | Formula | Default | Size |
|---|---|---|---|
| `renderedPoints` | `MAX_NB_RENDERED_POINTS * 16b` | 5,000,000 | 80 Mb |
| `renderedVoxels` | `MAX_NB_RENDERED_VOXELS * 16b` | 5,000,000 | 80 Mb |
| `renderedVoxelsSizes` | `MAX_NB_RENDERED_VOXELS * 16b` | 5,000,000 | 80 Mb |
| `renderedVoxelsNextChildIndex` | `MAX_NB_RENDERED_VOXELS * 4b` | 5,000,000 | 20 Mb |
| `renderedVoxelsNodes` | `MAX_NB_RENDERED_VOXELS * 4b` | 5,000,000 | 20 Mb |

### Caches

| Buffer | Formula | Default | Size |
|---|---|---|---|
| `updatesCache` | `(LRU_UPDATES_CACHE_SIZE + 1) * 52b` | 1,024 | 52 kb |
| `visibilityCache` | `LRU_VISIBILITY_CACHE_SIZE * 4b` | 1,024 | 4 kb |

### Spilling

| Buffer | Formula | Default | Size |
|---|---|---|---|
| `spilledPoints` | `MAX_NB_SPILLING_POINTS * 16b` | 1,000,000 | 16 Mb |
| `spillingNodes` | `MAX_NB_SPILLING_POINTS * 8b` | 1,000,000 | 8 Mb |

### Backlog

| Buffer | Formula | Default | Size |
|---|---|---|---|
| `backlogVoxels` | `MAX_NB_BACKLOG_VOXELS * 16b` | 5,000,000 | 80 Mb |
| `backlogVoxelsNodes` | `MAX_NB_BACKLOG_VOXELS * 8b` | 5,000,000 | 40 Mb |

### Allocators

| Buffer | Formula | Default | Size |
|---|---|---|---|
| `chunksAllocator` | `NB_ALLOCABLE_CHUNKS * (80b + 16400b + 8b) + 48b` | 32,768 * 16,488b + 48b | 540 Mb |
| `gridsAllocator` | `NB_ALLOCABLE_GRIDS * (80b + 262kb + 4b) + 48b` | 1,024 * 262,228b + 48b | 268 Mb |
| `nodesAllocator` | `NB_ALLOCABLE_NODES * (80b + 144b + 8b) + 48b` | 8,192 * 232b + 48b | 1.9 Mb |

---

## Total

```
36Mb + 4Mb + 8Mb + 8Mb + 8Mb + 8Mb + 1kb + 6kb + 1kb + 1kb + 1kb + 268Mb + 1.073Gb
+ 16b + 16b + 4Mb + 80Mb + 80Mb + 80Mb + 20Mb + 20Mb + 52kb + 4kb + 16Mb + 8Mb
+ 80Mb + 40Mb + 540Mb + 268Mb + 1.9
= 2.65 Gb
```

| | |
|---|---|
| **Total Memory Footprint** | **2.65 Gb** |