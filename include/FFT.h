#include <iostream>
#include <cmath>
#include <complex>
using namespace std;
const double PI = 3.14159265358979323846;
const int N = 2 << 10;
complex<double> b[N];

void fft_logic(complex<double> *a, int n, int inv)
{
    int(n == 1) return;
    // 将多项式系数分为两部分，一部分是奇系数，一部分是偶系数
    for (int i = 0; i < n / 2; i++)
    {
        b[i] = a[i * 2];             // even
        b[i + n / 2] = a[i * 2 + 1]; // odd
    }
    for (int i = 0; i < n; i++)
        a[i] = b[i];
    // 分治求even和odd
    fft_logic(a, n / 2, inv);
    fft_logic(a + n / 2, n / 2, inv);

    // 通过A1和A2，计算A
    double unit_rad = 2 * PI / n;
    for (int i = 0; i < n / 2; i++)
    {
        complex<double> x(cos(i * unit_rad), inv * sin(i * unit_rad));
        complex<double> even = a[i];
        complex<double> odd = x * a[i + n / 2];
        a[i] = even + odd;
        a[i + n / 2] = even - odd;
    }
}
void fft(complex<double> *a, int n)
{
    fft_logic(a, n, 1);
}
void inverse_fft(complex<double> *a, int n)
{
    fft_logic(a, n, -1);
    for (int i = 0; i < n; i++)
        a[i] /= n;
}