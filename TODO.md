# TODO list

## Next small steps

- Batch the memcpy on loadingtogpu
- Replace semaphores by setting context as current (be carefull of gpu memory free then)
- Clean the code + improve comments
- Implement frustum culling
- Try out big scenes, load / unload part of the scene on the fly
- Automatically get CPU / GPU capcatities to set the constants

## Next big steps

- Replace list with fixed size array to prepare for GPU side
- Add a way to flag not visible nodes from GPU side

## Coding part

- Add parameters to the UI (max nb batches, points per batches, etc...)
- Create uniform structure for better structuring in cuda code / gui
- Add a way to (manually ?) select the node to store / load
- Fix Vulkan segfault on quit
- supress warnings

## Research part

- Find a way to only send the delta
- Find which node to store (is LRU best strategy)
- Find a way to compress stored nodes
- Improve Color-filtering
- Find a way to not use occupancy grid
- Find a way to load closest batches first
- Improve on linked-list approach ?

## Report part

- Write down pipeline / method somewhere in a .md file