#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void mat_kernel(
    __global const float* A,   // M x N row-major
    __global const float* x,   // length N
    __global const float* B,   // length N
    __global float* y,         // length M
    const int M,
    const int N
) {
    const int row = get_global_id(0);
    if (row >= M) return;
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += A[row * N + i] * x[i];
    }
    y[row] = sum + B[row];
}

__kernel void elu_kernel(__global const float* input, __global float* output, const int N) {
    int idx = get_global_id(0);
    if (idx < N) {
        float x = input[idx];
        if(input[idx] > 0)  output[idx] = x;
        else    output[idx] = exp(x) - 1.0f;
    }
}