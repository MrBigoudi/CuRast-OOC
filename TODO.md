# TODO list

## Next small steps

- Batch the memcpy on loadingtogpu
- Clean the code + improve comments
- Implement frustum culling
- Automatically get CPU / GPU capacities to set the constants

## Next big steps

- Replace list with fixed size array to prepare for GPU side
- Add a way to flag not visible nodes from GPU side

## Coding part

- Add a settings file to avoid recompiling everytime
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