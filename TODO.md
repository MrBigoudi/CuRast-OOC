# TODO list


## Before next week's meeting

### Today 

<!-- - Fix rendering -->
<!-- - Create UI window with zoom on each file (store their AABB) -->
- Move the vis update on the device side
- Fix camera controls
- Implement vis cache

### This weekend / Monday

- Check all globalVariables and their initial values (rename some, destroy some, ...)
- Send globalVariables buffers as kernel input
- Fix loader + improve loading speed ? (maybe pre-read first points for first batch creation)

- Fix IO contention
- Optimise kernels (rendering and update)
- What about storing all nodes info in global variables -> avoid loading entirely + avoid duplicate occupancy on host side on store
- Fix random crops in model training




## Longer term goals

- Find a better dataset
- Aim for 100M points / seconds ?? (getting closer)



## Research part

- Find which node to store (is LRU best strategy)
- Find a way to compress stored nodes
- Improve Color-filtering
- Find a way to load closest batches first
- Improve on linked-list approach ?


## Report part

- Update the latex algorithm
- Write down pipeline / method somewhere in a .md file


<!-- regex:^(?!kernel_clearFramebuffer$|kernel_dummy$|kernel_resolve_colorbuffer_to_opengl_2D$|kernel_resolve_visbuffer_to_colorbuffer2D$|kernel_init_availableMcuSlots$).*kernel_ -->