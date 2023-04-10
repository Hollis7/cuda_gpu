#include<cuda_runtime.h>
#include<stdio.h>
#include"freshman.h"
int main(int argc,char **argv){
    cudaFuncCache cacheConfig=cudaFuncCachePreferShared;
    CHECK(cudaDeviceSetCacheConfig(cacheConfig));
    printf("cudaDeviceSetCacheConfig %s\n",cudaGetErrorString(cudaDeviceSetCacheConfig(cacheConfig)));
    return 0;

}