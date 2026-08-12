# TODO list

- Display number of points + flow rate

- Automatically fetch CPU / GPU capacities to set the constants + measure size of each structs + each buffers on GPU side
- Use nsight systems to find the bottlenecks and fix them


## Next steps (in priority order)

- Use cuda graphs to combine repetitive kernel launches
- Check all globalVariables and their initial values (rename some, destroy some, ...)

- Delay load / store of nodes as waiting for loading and storing won't be possible on GPU side
- Only send new AABBs to the GPU (send deltas more generally, is this worth it ?)
- Fix the huge CPU cache
- Better voxel rendering

## Coding part

- Modify caches to avoid unecessary new / delete in the double linked list (see hashmap version)
- Clean the code + improve comments
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