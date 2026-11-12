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



## Today

- [ ] Fix rendering of just unloaded nodes / updating nodes (from device to disk to device)
- [ ] Downgrade rendering to a simple octree cut to avoid having things diseapearing



## This month

- [ ] Check all globalVariables and their initial values (rename some, destroy some, ...)
- [ ] Send globalVariables buffers as kernel input

- [ ] Fix memory limitations
- [ ] Find a way to compress stored nodes
- [ ] What about storage limitations

- [ ] Fix IO contention
- [ ] Optimise kernels (rendering and update)
- [ ] Store all remaining nodes on quit (or on button pressed) + load them to init the octree / scene

- [ ] Improve inference speed
- [ ] Script to automatically generate a better dataset

- [ ] Properly put down the entire method
- [ ] Rewrite everything more cleanly
- [ ] Update the latex algorithms
- [ ] Improve Color-filtering


<!-- regex:^(?!kernel_clearFramebuffer$|kernel_dummy$|kernel_resolve_colorbuffer_to_opengl_2D$|kernel_resolve_visbuffer_to_colorbuffer2D$|kernel_init_availableMcuSlots$).*kernel_ -->