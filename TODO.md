# TODO list


## Tomorrow


### Morning

- [x] Implement visibility cache (for disk loading)
- [x] Fix rendering when using the multipass loader
- [x] Measure bottlenecks (nsys + ncu + superluminal) (still vis cache exchanges ??, not sure..., don't know...)
- [ ] Implement visibility cache (for node rendering)

### Afternoon

- [ ] Fix depth store
- [ ] Generate new simple dataset
- [ ] Try running a training session on cluster (or locally if cluster not available)
- [ ] Clean model architecture / training script


### Unrelated

- [ ]  Prepare papers for seminar


## This week

- [ ] Fix rendering of just unloaded nodes / updating nodes
- [ ] Fix nodes sorting (wrong sorting creates random holes)
- [ ] Fix priority loading


- [ ] Implement depth mask for inference (find depth with "an hierarchical coverage-based depth upsampling")
- [ ] Generate dataset for TAA
- [ ] Train with TAA

- [ ] Fix UI values
- [ ] Store all remaining nodes on quit
- [ ] Improve loading speed ? (maybe pre-read first points for first batch creation)


- [ ] Check all globalVariables and their initial values (rename some, destroy some, ...)
- [ ] Send globalVariables buffers as kernel input
- [ ] What about storing all nodes info in global variables -> avoid loading entirely + avoid duplicate occupancy on host side on store



## This month

- [ ] Fix memory limitations
- [ ] What about storage limitations

- [ ] Fix IO contention
- [ ] Optimise kernels (rendering and update)
- [ ] Improve inference speed

- [ ] Properly put down the entire method
- [ ] Rewrite everything more cleanly
- [ ] Update the latex algorithms
- [ ] Find a way to compress stored nodes
- [ ] Improve Color-filtering


<!-- regex:^(?!kernel_clearFramebuffer$|kernel_dummy$|kernel_resolve_colorbuffer_to_opengl_2D$|kernel_resolve_visbuffer_to_colorbuffer2D$|kernel_init_availableMcuSlots$).*kernel_ -->