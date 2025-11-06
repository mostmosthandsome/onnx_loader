#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void pure_mat_kernel(
    __global const float* input,   // length N
    __global const float* A,       // M x N row-major
    __global const float* B,       // length M (修正注释)
    __global float* y,             // length M
    const int M,
    const int N
) {
    const int row = get_global_id(0);
    if (row >= M) return;

    float sum = B[row];
    for (int i = 0; i < N; ++i) {
        sum += A[row * N + i] * input[i];
    }
    y[row] = sum;
}


__kernel void mat_slice_kernel(
    __global const float* input,   // length N
    __global const float* A,       // M x N row-major
    __global const float* B,       // length M (修正注释)
    __global float* y,             // length M
    const int M,
    const int N,
    const int slice_position
) {
    const int row = get_global_id(0);
    if (row >= M) return;
    int row_real_position;
    if(row >= M - slice_position)   row_real_position = row - M + slice_position;
    else row_real_position = row + slice_position;
    float sum = B[row_real_position];
    for (int i = 0; i < N; ++i) {
        sum += A[row_real_position * N + i] * input[i];
    }
    y[row] = sum;
}

__kernel void mat_elu_kernel(
    __global const float* input,   // length N
    __global const float* A,       // M x N row-major
    __global const float* B,       // length M
    __global float* y,             // length M
    const int M,
    const int N
) {
    const int row = get_global_id(0);
    if (row >= M) return;

    float sum = B[row];
    for (int i = 0; i < N; ++i) {
        sum += A[row * N + i] * input[i];
    }

    if (sum > 0.0f)
        y[row] = sum;
    else
        y[row] = exp(sum) - 1.0f;
}

__kernel void mat_clip_kernel(
    __global const float* input,   // length N
    __global const float* A,       // M x N row-major
    __global const float* B,       // length M (修正注释)
    __global float* y,             // length M
    const int M,
    const int N,
    const int limit
) {
    const int row = get_global_id(0);
    if (row >= M) return;

    float sum = B[row];
    for (int i = 0; i < N; ++i) {
        sum += A[row * N + i] * input[i];
    }
    if (sum > limit)
        y[row] = limit;
    else if (sum < -limit)
        y[row] = -limit;
    else
        y[row] = sum;
    
}

__kernel void ori_kernel(
    __global const float* input,
    __global const float* Mul,
    __global const float* Add,
    __global float* y,
    const int M
) {
    int row = get_global_id(0);
    float sum = 0.0f,sum_pow = 0.0;
    if(row >= M) return;
    // === 阶段 2: 求和 ===
    for(int i = 0; i < M; ++i) sum += input[i],sum_pow += input[i] * input[i];
    // === 阶段 3: 计算平均值并减去 ===
    float mean = sum / (float)M,mean_pow = sum_pow / (float)M;
    y[row] = (input[row] - mean) / sqrt(mean_pow - mean * mean + 0.00001);
    y[row] = y[row] * Mul[row] + Add[row];
    if (y[row] < 0.0f)   y[row] = exp(y[row]) - 1.0f;
    return;
}

__kernel void test_kernel(
    __global const float* input,
    __global const float* Mul,
    __global const float* Add,
    __global float* y,
    const int M
) {
    int row = get_global_id(0);
    float sum = 0.0f,sum_pow = 0.0;
    if(row >= M) return;
    // === 阶段 2: 求和 ===
    for(int i = 0; i < M; ++i) sum += input[i],sum_pow += input[i] * input[i];
    // === 阶段 3: 计算平均值并减去 ===
    float mean = sum / (float)M,mean_pow = sum_pow / (float)M;
    y[row] = (input[row] - mean) / sqrt(mean_pow - mean * mean + 0.00001);
    y[row] = y[row] * Mul[row] + Add[row];
    if (y[row] < 0.0f)   y[row] = exp(y[row]) - 1.0f;
    return;
}
