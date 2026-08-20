# TODO list


- Make things faster:
    - Make the loading happen in another thread ?
    - Remove spilled points + do not load points, just the counter (this will reduce bandwith + memory on device side)
    - Better handling for voxels as well ?? maybe a compressed octree (building DAGs on CPU side in the background) ?
    - Restore the CPU cache to avoid loading from disk
    - Try nsight compute to detect slow parts
    - Try to display timings to detect slower parts

    - Make the "exchangedVoxels / Points" unrelated to the number of nodes that can be exchanged; instead just have a fixed number of exchangeable chunks (this might reduce the sizes of the structures, allowing for more nbExchangeable nodes)

    - Find better launching values


- Restore visibility cache

- Aim for 100M points / seconds ??

- Add preprocessing macros to remove asserts on RELEASE mode (plus easy way to flag/unflag it for testing)




## Next steps (in priority order)

- Use cuda graphs to combine repetitive kernel launches
- Check all globalVariables and their initial values (rename some, destroy some, ...)
- Better voxel rendering



## Coding part

- Clean the code + improve comments
- Fix Vulkan segfault on quit
- supress warnings



## Research part

- Find which node to store (is LRU best strategy)
- Find a way to compress stored nodes
- Improve Color-filtering
- Find a way to load closest batches first
- Improve on linked-list approach ?




## Report part

- Update the latex algorithm
- Write down pipeline / method somewhere in a .md file