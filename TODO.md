# TODO list

## Next steps (in priority order)

- Delay load / store of nodes as waiting for loading and storing won't be possible on GPU side
- Only send new AABBs to the GPU (send deltas more generally, is this worth it ?)
- Fix the huge CPU cache
- Better voxel rendering

## Coding part

- Clean the code + improve comments
- Automatically fetch CPU / GPU capacities to set the constants
- Fix Vulkan segfault on quit
- supress warnings
- Replace list with fixed size array to prepare for GPU side


## Research part

- Find a way to only send the delta
- Find which node to store (is LRU best strategy)
- Find a way to compress stored nodes
- Improve Color-filtering
- Find a way to load closest batches first
- Improve on linked-list approach ?


## Report part

- Update the latex algorithm
- Write down pipeline / method somewhere in a .md file