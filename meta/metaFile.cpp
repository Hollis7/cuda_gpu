#include <iostream>
#include <string>
#include <cstdlib>
void transformMatrix2D_CPU(float *MatA, float *MatB, int nx, int ny)
{
    for (int j = 0; j < ny; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            MatB[i * ny + j] = MatA[j * nx + i];
        }
    }
}
int main()
{
    float* arr = new float[12];
    for (int i = 0; i < 12; i++)
    {
        arr[i] = i + 1;
    }
    float *res = new float[12];
    transformMatrix2D_CPU(arr,res,4,3);
    printf("\n");
    for(int i=0;i<12;i++)
    {
        printf("%f ",arr[i]);
    }
    printf("\n------------------------------------\n");
    for(int i=0;i<12;i++)
    printf("%f ",res[i]);
    printf("\n");

    return 0;
}