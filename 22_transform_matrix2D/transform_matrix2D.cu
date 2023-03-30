#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
// cpu transform
void transformMatrix2D_CPU(float *out, float *in, int nx, int ny)
{
  for (int j = 0; j < ny; j++)
  {
    for (int i = 0; i < nx; i++)
    {
      out[i * ny + j] = in[j * nx + i];
    }
  }
}
__global__ void copyRow(float *out, float *in, int nx, int ny)
{
  int ix = threadIdx.x + blockDim.x * blockIdx.x;
  int iy = threadIdx.y + blockDim.y * blockIdx.y;
  int idx = ix + iy * nx;
  if (ix < nx && iy < ny)
  {
    out[idx] = in[idx];
  }
}
__global__ void copyCol(float *out, float *in, int nx, int ny)
{
  int ix = threadIdx.x + blockDim.x * blockIdx.x;
  int iy = threadIdx.y + blockDim.y * blockIdx.y;
  int idx = ix * ny + iy;
  if (ix < nx && iy < ny)
  {
    out[idx] = in[idx];
  }
}
__global__ void transformNaiveRow(float *out, float *in, int nx, int ny)
{
  int ix = threadIdx.x + blockDim.x * blockIdx.x;
  int iy = threadIdx.y + blockDim.y * blockIdx.y;
  int idx_row = ix + iy * nx;
  int idx_col = ix * ny + iy;
  if (ix < nx && iy < ny)
  {
    out[idx_col] = in[idx_row];
  }
}
__global__ void transformNaiveCol(float *out, float *in, int nx, int ny)
{
  int ix = threadIdx.x + blockDim.x * blockIdx.x;
  int iy = threadIdx.y + blockDim.y * blockIdx.y;
  int idx_row = ix + iy * nx;
  int idx_col = ix * ny + iy;
  if (ix < nx && iy < ny)
  {
    out[idx_row] = in[idx_col];
  }
}
__global__ void transformNaiveRowUnroll(float *out, float *in, int nx, int ny)
{
  int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
  int iy = threadIdx.y + blockDim.y * blockIdx.y;
  int idx_row = ix + iy * nx;
  int idx_col = ix * ny + iy;
  if (ix+blockDim.x*3 < nx && iy < ny)
  {
    out[idx_col] = in[idx_row];
    out[idx_col + ny * 1 * blockDim.x] = in[idx_row + 1 * blockDim.x];
    out[idx_col + ny * 2 * blockDim.x] = in[idx_row + 2 * blockDim.x];
    out[idx_col + ny * 3 * blockDim.x] = in[idx_row + 3 * blockDim.x];
  }
}
__global__ void transformNaiveColUnroll(float *out, float *in, int nx, int ny)
{
  int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
  int iy = threadIdx.y + blockDim.y * blockIdx.y;
  int idx_row = ix + iy * nx;
  int idx_col = ix * ny + iy;
  if (ix +blockDim.x*3< nx && iy < ny)
  {
    out[idx_row] = in[idx_col];
    out[idx_row + 1 * blockDim.x] = in[idx_col + ny * 1 * blockDim.x];
    out[idx_row + 2 * blockDim.x] = in[idx_col + ny * 2 * blockDim.x];
    out[idx_row + 3 * blockDim.x] = in[idx_col + ny * 3 * blockDim.x];
  }
}
__global__ void transformNaiveRowDiagonal(float *out, float *in, int nx, int ny)
{
  int block_y = blockIdx.x;
  int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
  int ix = threadIdx.x + blockDim.x * block_x;
  int iy = threadIdx.y + blockDim.y * block_y;
  int idx_row = ix + iy * nx;
  int idx_col = ix * ny + iy;
  if (ix < nx && iy < ny)
  {
    out[idx_col] = in[idx_row];
  }
}
__global__ void transformNaiveColDiagonal(float *out, float *in, int nx, int ny)
{
  int block_y = blockIdx.x;
  int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
  int ix = threadIdx.x + blockDim.x * block_x;
  int iy = threadIdx.y + blockDim.y * block_y;
  int idx_row = ix + iy * nx;
  int idx_col = ix * ny + iy;
  if (ix < nx && iy < ny)
  {
    out[idx_row] = in[idx_col];
  }
}

int main(int argc, char **argv)
{
  printf("strating...\n");
  initDevice(0);
  int nx = 1 << 12;
  int ny = 1 << 12;
  int dimx = 32;
  int dimy = 32;
  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);
  int transform_kernel = 0;
  if (argc == 2)
    transform_kernel = atoi(argv[1]);
  if (argc >= 4)
  {
    transform_kernel = atoi(argv[1]);
    dimx = atoi(argv[2]);
    dimy = atoi(argv[3]);
  }

  // Malloc
  float *A_host = (float *)malloc(nBytes);
  float *B_host = (float *)malloc(nBytes);
  float *A_from_gpu = (float *)malloc(nBytes);
  memset(A_from_gpu, 0, nBytes);
  initialData(A_host, nxy);

  // cudaMalloc
  float *A_dev = NULL;
  float *B_dev = NULL;
  CHECK(cudaMalloc((void **)&A_dev, nBytes));
  CHECK(cudaMalloc((void **)&B_dev, nBytes));

  CHECK(cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(B_dev, 0, nBytes));

  // cpu compute
  double iStart = cpuSecond();
  transformMatrix2D_CPU(A_host, B_host, nx, ny);
  double iElaps = cpuSecond() - iStart;
  printf("CPU Execution Time elapsed %f sec\n", iElaps);

  // 2d block and 2d grid
  dim3 block(dimx, dimy);
  //the grid has the similar shape of matrics
  dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);
  dim3 block_1(dimx, dimy);
  dim3 grid_1((nx - 1) / (block_1.x * 4) + 1, (ny - 1) / block_1.y + 1);
  const char *p = NULL;
  iStart = cpuSecond();
  switch (transform_kernel)
  {
  case 0:
    copyRow<<<grid, block>>>(B_dev, A_dev, nx, ny);
    break;
  case 1:
    copyCol<<<grid, block>>>(B_dev, A_dev, nx, ny);
    break;
  case 2:
    transformNaiveRow<<<grid, block>>>(A_dev, B_dev, nx, ny);
    p="transformNaiveRow";
    break;
  case 3:
    transformNaiveCol<<<grid, block>>>(A_dev, B_dev, nx, ny);
    p="transformNaiveCol";
    break;
  case 4:
    transformNaiveColUnroll<<<grid_1, block_1>>>(A_dev, B_dev, nx, ny);
    p="transformNaiveColUnroll";
    break;
  case 5:
    transformNaiveColUnroll<<<grid_1, block_1>>>(A_dev, B_dev, nx, ny);
    p="transformNaiveColUnroll";
    break;
  case 6:
    transformNaiveRowDiagonal<<<grid, block>>>(A_dev, B_dev, nx, ny);
    p="transformNaiveRowDiagonal";
    break;
  case 7:
    transformNaiveColDiagonal<<<grid, block>>>(A_dev, B_dev, nx, ny);
    break;
  default:
    break;
  }
  CHECK(cudaDeviceSynchronize());
  iElaps = cpuSecond() - iStart;
  printf(" (kernal %s)Time elapsed %f sec\n", p,iElaps);
  CHECK(cudaMemcpy(A_from_gpu, A_dev, nBytes, cudaMemcpyDeviceToHost));
  checkResult(A_from_gpu, A_host, nxy);

  cudaFree(A_dev);
  cudaFree(B_dev);
  free(A_host);
  free(B_host);
  free(A_from_gpu);
  cudaDeviceReset();
  return 0;
}
