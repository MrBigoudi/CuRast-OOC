# TODO list


- Make things faster:
    - Find better launching values
    - Check if some COPY_TO / COPY_FROM can be made async


- Aim for 100M points / seconds ??




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