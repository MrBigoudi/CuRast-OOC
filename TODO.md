# TODO list

## Next steps (in priority order)

- Rework the maps aabbRelationshipMap and aabbParentMap
- Measure timings and memory usage more precisely 
- Clean the code + improve comments
- Allocate memory once on CPU side to avoid new/delete and reuse it on demand


## Coding part

- Automatically fetch CPU / GPU capacities to set the constants
- Replace exit(EXIT_FAILURE) with exception raise to catch it in the main and clear the temporary folder
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

- Write down pipeline / method somewhere in a .md file