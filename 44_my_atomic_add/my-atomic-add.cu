#include "../include/freshman.h"
#include <stdio.h>
#include <cuda_runtime.h>

/**
 * This example illustrates implementation of custom atomic operations using
 * CUDA's built-in atomicCAS function to implement atomic signed 32-bit integer
 * addition.
 **/

__device__ int myAtomicAdd(int *address, int incr)
{
    // Create an initial guess for the value stored at *address.
    int guess = *address;
    int oldValue = atomicCAS(address, guess, guess + incr);

    // Loop while the guess is incorrect.
    // if first atomicCAS succeed,pass the while
    while (oldValue != guess)
    {
        guess = oldValue;
        oldValue = atomicCAS(address, guess, guess + incr);
    }
    return oldValue;
}

__device__ float myAtomicAdd2(float* address, float incr)
{
    unsigned int* typeAddress = (unsigned int*)address;
 
    float currentVal = *address;
 
    unsigned int expected = __float2uint_rn(currentVal);
 
    unsigned int desired = __float2uint_rn(currentVal + incr);
 
    int oldIntValue = atomicCAS(typeAddress, expected, desired);
 
    while(oldIntValue != expected)
    {
        expected = oldIntValue;
 
        desired = __float2uint_rn(__uint2float_rn(oldIntValue) + incr);
        oldIntValue = atomicCAS(typeAddress, expected, desired);
 
    }
 
    return __uint2float_rn(oldIntValue);
}

__global__ void kernel(int *sharedInteger)
{
    myAtomicAdd(sharedInteger, 1);
}

int main(int argc, char **argv)
{
    int h_sharedInteger;
    int *d_sharedInteger;
    CHECK(cudaMalloc((void **)&d_sharedInteger, sizeof(int)));
    CHECK(cudaMemset(d_sharedInteger, 0x00, sizeof(int)));

    kernel<<<4, 128>>>(d_sharedInteger);

    CHECK(cudaMemcpy(&h_sharedInteger, d_sharedInteger, sizeof(int),
                     cudaMemcpyDeviceToHost));
    printf("4 x 128 increments led to value of %d\n", h_sharedInteger);

    cudaFree(d_sharedInteger);

    return 0;
}
