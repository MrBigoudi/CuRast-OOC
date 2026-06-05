# TODO list

## Next small steps

- Add a way to (manually ?) select the node to store / load
- Add a way to flag not visible nodes from GPU side
- Add a way to reload a node
- Add parameters to the UI (max nb batches, points per batches, etc...)
- Implement frustum culling
- Create uniform structure for better structuring in cuda code / gui
- supress warnings

## Next big steps

- Implement LRU caching

## Coding part

- Fix Vulkan segfault on quit
- Rebuild occupancy grid on load

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