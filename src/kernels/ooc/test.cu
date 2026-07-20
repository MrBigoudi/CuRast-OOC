#include "utils.cuh"

extern "C" __global__
void kernel_test(){
    int* p = new int(42);
    printf("new test: %d\n", p ? *p : -1);
    delete p;
}