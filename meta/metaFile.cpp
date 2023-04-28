#include <iostream>
#include <string>
int main()
{
    double a = 3.1415927f;
    double b = 3.1415928f;
    if (a == b)
    {
        printf("equal \n");
    }
    else
    {
        printf("not equal");
    }
    return 0;
}