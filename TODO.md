# TODO list



## Last week

- [x] Implement visibility cache (for disk loading)
- [x] Fix rendering when using the multipass loader
- [x] Measure bottlenecks (nsys + ncu + superluminal) (still vis cache exchanges ??, not sure..., don't know...)
- [x] Fix depth store
- [x] Generate new simple dataset

- [x] Fix nodes sorting (wrong sorting creates random holes)
- [x] Improve loading speed ? (maybe pre-read first points for first batch creation)
- [x] Fix original bounding box
- [x] Fix priority loading
- [x] Fix huge segfault (caused by intel CPU multithreading on too many cores apparently...)
- [x] Fix UI values

- [x] Downgrade rendering to a simple octree cut to avoid having things diseapearing
- [x] Fix rendering of just unloaded nodes / updating nodes (from device to disk to device)



## Today

- [x] Allow dropping folders directly

<!-- VERY HIGH PRIORITY -->
- [ ] Slider to change the points budget
- [ ] Coloring per node
- [ ] Coloring per level
- [ ] Change how voxels are rendered (quad )
- [ ] Load immediately when zooming in a region 



## This month

<!-- Quality of life -->
- [ ] Rewrite everything more cleanly (delete unnecessary parts + better comments + refactor settings)
- [ ] Check all globalVariables and their initial values (rename some, destroy some, ...)
- [ ] Store all remaining nodes on quit (or on button pressed) + load them to init the octree / scene
- [ ] Properly put down the entire method
- [ ] Update the latex algorithms
  
<!-- Bug fixing -->
- [ ] Rethink point distribution
- [ ] Fix memory limitations (do not crash if too much RAM is used)
- [ ] What about storage limitations (do not crash if too much Disk Space is used)

<!-- Optimizations -->
- [ ] Send globalVariables buffers as kernel input
- [ ] Optimise kernels (rendering and update)
- [ ] Fix IO contention

<!-- New contributions -->
- [ ] Find a way to compress stored nodes
- [ ] Find a way to average colors in the voxels (or other color-filtering algorithms)

<!-- Neural Network -->
- [ ] Improve inference speed
- [ ] Script to automatically generate a better dataset



<!-- regex:^(?!kernel_clearFramebuffer$|kernel_dummy$|kernel_resolve_colorbuffer_to_opengl_2D$|kernel_resolve_visbuffer_to_colorbuffer2D$|kernel_init_availableMcuSlots$).*kernel_ -->