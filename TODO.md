# TODO list

## Next small steps

- Compare octrees before / after load
- Add parameters to the UI (max nb batches, points per batches, etc...)
- Rebuild occupancy grid on load
- Implement frustum culling
- Create uniform structure for better structuring in cuda code / gui
- supress warnings

## Next big steps

- Implement LRU caching

## Coding part

- Fix Vulkan segfault on quit
- Add a way to load / store nodes synchronously

## Research part

- Find a way to only send the delta
- Find which node to store (is LRU best strategy)
- Find a way to compress stored nodes
- Color-filtering
- Find a way to not use occupancy grid
- Find a way to load closest batches first
- Improve on linked-list approach ?

## Report part

- Write down pipeline / method somewhere in a .md file