# TODO list

## Next steps (in priority order)

- Implement frustum culling on GPU side (or just disable non-visible nodes)
- Add a settings file to avoid recompiling everytime
- Clean the code + improve comments
- Create uniform structure for better structuring in cuda code / gui
- Measure timings and memory usage more precisely 
- Batch the memcpy on loadingtogpu (ie separate memalloc from memcpy)

## Coding part

- Add parameters to the UI (max nb batches, points per batches, etc...)
- Automatically fetch CPU / GPU capacities to set the constants
- Add a way to (manually ?) select the node to store / load
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