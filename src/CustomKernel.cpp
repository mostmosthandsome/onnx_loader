#include "CustomKernel.h"


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
using namespace handsome;



MlpDataMemory::~MlpDataMemory()
{
    for(int i = 0; i < num_layers; ++i)  clReleaseMemObject(weight_buff[i]);
    for(int i = 0; i < num_layers; ++i)  clReleaseMemObject(bias_buff[i]);
}

class CustomKernel::CustomKernelPrivate
{
public:
    cl_device_id device;
    cl_platform_id platform;
    cl_program program{NULL};
    cl_kernel pure_mat_kernel{NULL},mat_elu_kernel{NULL},mat_clip_kernel{NULL}, mat_slice_kernel{NULL},ori_kernel{NULL};
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int err;
    cl_mem temp_output_buff[2];


    //cl_mem
    cl_mem encoder_out_buff,body_vel_out_buff,fc_mu_out_buff,actor_in_buff, final_out_buff;


    /**
     * @brief do a inference of mlp_data_ptr, output will be put in output_buff
     # TODO 加入自动识别功能，自动识别最后一层是不是elu
    */
    void InferenceMlp(cl_command_queue &queue, cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr);
    void inference_single_end_clip(cl_command_queue &queue, cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr, float clip_limit);

    void copy_cl_mem(cl_command_queue &queue, cl_mem &src_mem,cl_mem &dst_mem, int copy_size);

    void concat(cl_command_queue &queue, cl_mem &src1, int size1, cl_mem &src2, int size2, cl_mem &src3, int size3, cl_mem &dst);

    void concat(cl_command_queue &queue, cl_mem &src1, int size1, cl_mem &src2, int size2, cl_mem &dst);

    void inference_ori_actor(cl_command_queue &queue, cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr);

    void inference_with_slice_changed(cl_command_queue &queue, cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr, int slice_position);

};

CustomKernel::CustomKernel():data_ptr(std::make_unique<CustomKernelPrivate>())
{

    /* 获取平台设备信息 */
    data_ptr->err = clGetPlatformIDs(1, &data_ptr->platform, &data_ptr->ret_num_platforms);
    data_ptr->err = clGetDeviceIDs(data_ptr->platform, CL_DEVICE_TYPE_GPU, 1, &data_ptr->device, &data_ptr->ret_num_devices);

    /* 创建 OpenCL 上下文 */
    context = clCreateContext( NULL, 1, &data_ptr->device, NULL, NULL, &data_ptr->err);

    /* 创建命令队列 */
    queue = clCreateCommandQueue(context, data_ptr->device, 0, &data_ptr->err);

    data_ptr->temp_output_buff[0] = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(float) * 512,
        NULL,
        &data_ptr->err
    );
    data_ptr->temp_output_buff[1] = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(float) * 512,
        NULL,
        &data_ptr->err
    );
    if (data_ptr->err < 0) { perror("Couldn't create res buffer"); exit(1); }

}

CustomKernel::~CustomKernel()
{
    
      /* 終了処理 */
    data_ptr->err = clFlush(queue);
    data_ptr->err = clFinish(queue);
    data_ptr->err = clReleaseKernel(data_ptr->pure_mat_kernel);
    data_ptr->err = clReleaseKernel(data_ptr->mat_elu_kernel);
    data_ptr->err = clReleaseKernel(data_ptr->mat_clip_kernel);
    data_ptr->err = clReleaseKernel(data_ptr->mat_clip_kernel);
    data_ptr->err = clReleaseKernel(data_ptr->mat_slice_kernel);


    
    data_ptr->err = clReleaseProgram(data_ptr->program);

    data_ptr->err = clReleaseCommandQueue(queue);

    data_ptr->err = clReleaseContext(context);
    data_ptr->err = clReleaseMemObject(data_ptr->temp_output_buff[0]),data_ptr->err = clReleaseMemObject(data_ptr->temp_output_buff[1]);


}

void CustomKernel::load_openCL_code(std::string file_name)
{
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size; 

        /* Read program file and place content into buffer */
    program_handle = fopen(file_name.c_str(), "r");
    if(program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);   
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    //对读取到的代码进行编译，创建Program
    /* Create program from file */
    data_ptr->program = clCreateProgramWithSource(context, 1, 
        (const char**)&program_buffer, &program_size, &data_ptr->err);
    if(data_ptr->err < 0) {
        perror("Couldn't create the program");
        exit(1);   
    }
    free(program_buffer);

    /* Build program */
    data_ptr->err = clBuildProgram(data_ptr->program, 0, NULL, NULL, NULL, NULL);
    if(data_ptr->err < 0) {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(data_ptr->program, data_ptr->device, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(data_ptr->program, data_ptr->device, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    /* Create kernel for the kernel function */
    data_ptr->pure_mat_kernel = clCreateKernel(data_ptr->program, "pure_mat_kernel", &data_ptr->err);
    
    if(data_ptr->err < 0) {
        perror("Couldn't create the mat kernel");
        exit(1);   
    }

    data_ptr->mat_elu_kernel = clCreateKernel(data_ptr->program, "mat_elu_kernel", &data_ptr->err);

    if(data_ptr->err < 0) {
        perror("Couldn't create the elu kernel");
        exit(1);   
    }
    
    data_ptr->mat_clip_kernel = clCreateKernel(data_ptr->program, "mat_clip_kernel", &data_ptr->err);
    
    if(data_ptr->err < 0) {
        perror("Couldn't create the clip kernel");
        exit(1);   
    }

    data_ptr->mat_slice_kernel = clCreateKernel(data_ptr->program, "mat_slice_kernel", &data_ptr->err);
    
    if(data_ptr->err < 0) {
        perror("Couldn't create the slice kernel");
        exit(1);   
    }

    data_ptr->ori_kernel = clCreateKernel(data_ptr->program, "ori_kernel", &data_ptr->err);
    
    if(data_ptr->err < 0) {
        std::cout << "Couldn't create the ori kernel, err_code = " << data_ptr->err << std::endl;
        exit(1);   
    }
    return;
}


void CustomKernel::concat(cl_mem &src1, int size1, cl_mem &src2, int size2, cl_mem &src3, int size3, cl_mem &dst)
{
    data_ptr->concat(queue, src1, size1, src2, size2, src3, size3, dst);
}

void CustomKernel::concat(cl_mem &src1, int size1, cl_mem &src2, int size2, cl_mem &dst)
{
    data_ptr->concat(queue, src1, size1, src2, size2, dst);
}

void CustomKernel::inference_mlp(cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr)
{
    data_ptr->InferenceMlp(queue, input_buff, output_buff, mlp_data_ptr);
}

void CustomKernel::inference_single_end_clip(cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr, float clip_limit)
{
    data_ptr->inference_single_end_clip(queue, input_buff, output_buff, mlp_data_ptr, clip_limit);
}

void CustomKernel::inference_ori_actor(cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr)
{
    data_ptr->inference_ori_actor(queue, input_buff, output_buff, mlp_data_ptr);
}

void CustomKernel::inference_with_slice_changed(cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr, int slice_position)
{
    data_ptr->inference_with_slice_changed(queue, input_buff, output_buff, mlp_data_ptr, slice_position);
}

///////////////////////////////////////////////////////////////////


void CustomKernel::load_mlp_params(std::shared_ptr<OnnxLoader> model_ptr,std::shared_ptr<MlpDataMemory> mlp_ptr,std::string mlp_name)
{
    //先从创建好的模型中读取mlp数据
    std::shared_ptr<MlpParam> mlp_param_data = std::make_shared<MlpParam>();
    model_ptr->load_mlp_param(mlp_param_data,mlp_name);
    int num_layers = mlp_param_data->num_layers;
    mlp_ptr->num_layers = num_layers;
    mlp_ptr->weight_buff.resize(num_layers), mlp_ptr->bias_buff.resize(num_layers),mlp_ptr->rows.resize(num_layers),mlp_ptr->cols.resize(num_layers);
    mlp_ptr->input_dim = mlp_param_data->cols[0], mlp_ptr->output_dim = mlp_param_data->rows[num_layers - 1];

    //创建cl_mem
    for (int i = 0; i < num_layers; ++i) {
        int out_dim = mlp_param_data->rows[i];
        int in_dim  = mlp_param_data->cols[i];
        mlp_ptr->rows[i] =  mlp_param_data->rows[i],mlp_ptr->cols[i] = mlp_param_data->cols[i];
        // 展平权重矩阵
        std::vector<float> flat_weight;
        flat_weight.resize(out_dim * in_dim);
        for(int j = 0; j < out_dim; ++j)
            for(int k = 0; k < in_dim; ++k) 
                flat_weight[j * in_dim + k] = mlp_param_data->weights[i][j][k];
        // 创建权重 buffer
        mlp_ptr->weight_buff[i] = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * out_dim * in_dim,
            flat_weight.data(),
            &data_ptr->err
        );
        if (data_ptr->err < 0) {
            perror(("Couldn't create weight_buff[" + std::to_string(i) + "]").c_str());
            exit(1);
        }

        // 创建 bias buffer
         mlp_ptr->bias_buff[i] = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * mlp_param_data->biases[i].size(),
            mlp_param_data->biases[i].data(),
            &data_ptr->err
        );
        if (data_ptr->err < 0) {
            perror(("Couldn't create bias_buff[" + std::to_string(i) + "]").c_str());
            exit(1);
        }   
    }
    if(mlp_param_data->mul_param.size() > 0)
    {
        //mul param
        mlp_ptr->mul_buff.resize(num_layers - 1),mlp_ptr->add_buff.resize(num_layers - 1);
        for(int i = 0; i < num_layers - 1; ++i)
        {
            int out_dim = mlp_param_data->rows[i];
            // 创建 mul buffer
            mlp_ptr->mul_buff[i] = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * out_dim,
                mlp_param_data->mul_param[i].data(),
                &data_ptr->err
            );
            if (data_ptr->err < 0) {
                perror(("Couldn't create mul_buff[" + std::to_string(i) + "]").c_str());
                exit(1);
            }   
            // 创建 add buffer
             mlp_ptr->add_buff[i] = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * out_dim,
                mlp_param_data->add_param[i].data(),
                &data_ptr->err
            );
            if( data_ptr->err < 0) {
                perror(("Couldn't create add_buff[" + std::to_string(i) + "]").c_str());
                exit(1);
            }
        }

    }
}


void CustomKernel::CustomKernelPrivate::InferenceMlp(cl_command_queue &queue, cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr)
{
    // size_t global_size[2];
    size_t global_size;
    cl_event kernel_event;
    bool temp_out_id = 0;
    cl_mem *now_input_buff = &input_buff;
    for(int i = 0; i < mlp_data_ptr->num_layers; ++i)
    {
        // global_size[0] = mlp_data_ptr->rows[i];
        // global_size[1] = mlp_data_ptr->cols[i]; 
        global_size = mlp_data_ptr->rows[i];  // 每个 work-item 负责一行        
        if(i == mlp_data_ptr->num_layers - 1) //最后一层只有gemm，且直接输出到Output
        {
            clSetKernelArg(pure_mat_kernel, 0, sizeof(cl_mem), now_input_buff);
            //设置 kernel 参数
            clSetKernelArg(pure_mat_kernel, 1, sizeof(cl_mem), &mlp_data_ptr->weight_buff[i]);
            clSetKernelArg(pure_mat_kernel, 2, sizeof(cl_mem), &mlp_data_ptr->bias_buff[i]);
            clSetKernelArg(pure_mat_kernel, 3, sizeof(cl_mem), &output_buff);
            clSetKernelArg(pure_mat_kernel, 4, sizeof(int), &mlp_data_ptr->rows[i]);
            clSetKernelArg(pure_mat_kernel, 5, sizeof(int), &mlp_data_ptr->cols[i]);
            err = (clEnqueueNDRangeKernel)(queue, pure_mat_kernel, 1, NULL,
                                   &global_size, NULL, 0, NULL, 
            &kernel_event);
            // 等待 kernel 完成
            clWaitForEvents(1, &kernel_event);
        }
        else
        {
            clSetKernelArg(mat_elu_kernel, 0, sizeof(cl_mem), now_input_buff);
            //设置 kernel 参数
            clSetKernelArg(mat_elu_kernel, 1, sizeof(cl_mem), &mlp_data_ptr->weight_buff[i]);
            clSetKernelArg(mat_elu_kernel, 2, sizeof(cl_mem), &mlp_data_ptr->bias_buff[i]);
            clSetKernelArg(mat_elu_kernel, 3, sizeof(cl_mem), &temp_output_buff[temp_out_id]);
            clSetKernelArg(mat_elu_kernel, 4, sizeof(int), &mlp_data_ptr->rows[i]);
            clSetKernelArg(mat_elu_kernel, 5, sizeof(int), &mlp_data_ptr->cols[i]);
            err = (clEnqueueNDRangeKernel)(queue, mat_elu_kernel, 1, NULL,
                                    &global_size, NULL, 0, NULL, 
            &kernel_event);

            // 等待 kernel 完成
            clWaitForEvents(1, &kernel_event);
            now_input_buff = &temp_output_buff[temp_out_id],temp_out_id = !temp_out_id;
        }
      
    }
}

void CustomKernel::CustomKernelPrivate::copy_cl_mem(cl_command_queue &queue, cl_mem &src_mem, cl_mem &dest_mem, int copy_size)
{
    // 从 temp_output_buff[0] 拷贝前 output_dim 部分到 encoder_out_buff
    err = clEnqueueCopyBuffer(
        queue,                 // 命令队列
        src_mem,      // 源缓冲区
        dest_mem,                // 目标缓冲区
        0,                               // 源偏移
        0,                               // 目标偏移
        sizeof(float) * copy_size,                       // 拷贝大小（字节）
        0, nullptr, nullptr              // 同步选项
    );

    if (err != CL_SUCCESS) printf("Failed to copy buffer: %d\n", err);

    // 可选：等待执行完
    clFinish(queue);
}

void CustomKernel::CustomKernelPrivate::concat(
    cl_command_queue &queue,
    cl_mem &src1, int size1,
    cl_mem &src2, int size2,
    cl_mem &src3, int size3,
    cl_mem &dst)
{
    cl_int err;
    size_t offset = 0;

    // 元素字节大小（假设 float）
    const size_t elem_size = sizeof(float);

    // ---- 第1段 ----
    err = clEnqueueCopyBuffer(
        queue,
        src1,
        dst,
        0,
        offset,
        elem_size * size1,    // ✅ 拷贝字节数
        0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        fprintf(stderr, "[concat] Copy src1 failed, err=%d\n", err);
    offset += elem_size * size1;

    // ---- 第2段 ----
    err = clEnqueueCopyBuffer(
        queue,
        src2,
        dst,
        0,
        offset,
        elem_size * size2,    // ✅ 拷贝字节数
        0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        fprintf(stderr, "[concat] Copy src2 failed, err=%d\n", err);
    offset += elem_size * size2;

    // ---- 第3段 ----
    err = clEnqueueCopyBuffer(
        queue,
        src3,
        dst,
        0,
        offset,
        elem_size * size3,    // ✅ 拷贝字节数
        0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        fprintf(stderr, "[concat] Copy src3 failed, err=%d\n", err);

    // 等待执行完毕
    clFinish(queue);
}


void CustomKernel::CustomKernelPrivate::concat(
    cl_command_queue &queue,
    cl_mem &src1, int size1,
    cl_mem &src2, int size2,
    cl_mem &dst)
{
    cl_int err;
    size_t offset = 0;

    // 元素字节大小（假设 float）
    const size_t elem_size = sizeof(float);

    // ---- 第1段 ----
    err = clEnqueueCopyBuffer(
        queue,
        src1,
        dst,
        0,
        offset,
        elem_size * size1,    // ✅ 拷贝字节数
        0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        fprintf(stderr, "[concat] Copy src1 failed, err=%d\n", err);
    offset += elem_size * size1;

    // ---- 第2段 ----
    err = clEnqueueCopyBuffer(
        queue,
        src2,
        dst,
        0,
        offset,
        elem_size * size2,    // ✅ 拷贝字节数
        0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        fprintf(stderr, "[concat] Copy src2 failed, err=%d\n", err);
    offset += elem_size * size2;

    // 等待执行完毕
    clFinish(queue);
}

void CustomKernel::CustomKernelPrivate::inference_single_end_clip(cl_command_queue &queue, cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr,float clip_limit)
{
    size_t global_size = mlp_data_ptr->rows[0];
    cl_event kernel_event;
    clSetKernelArg(mat_clip_kernel, 0, sizeof(cl_mem), &input_buff);
    //设置 kernel 参数
    clSetKernelArg(mat_clip_kernel, 1, sizeof(cl_mem), &mlp_data_ptr->weight_buff[0]);
    clSetKernelArg(mat_clip_kernel, 2, sizeof(cl_mem), &mlp_data_ptr->bias_buff[0]);
    clSetKernelArg(mat_clip_kernel, 3, sizeof(cl_mem), &output_buff);
    clSetKernelArg(mat_clip_kernel, 4, sizeof(int), &mlp_data_ptr->rows[0]);
    clSetKernelArg(mat_clip_kernel, 5, sizeof(int), &mlp_data_ptr->cols[0]);
    clSetKernelArg(mat_clip_kernel, 6, sizeof(float), &clip_limit);

    err = (clEnqueueNDRangeKernel)(queue, mat_clip_kernel, 1, NULL,
                            &global_size, NULL, 0, NULL, 
    &kernel_event);
    // 等待 kernel 完成
    clWaitForEvents(1, &kernel_event);
}

void CustomKernel::CustomKernelPrivate::inference_ori_actor(cl_command_queue &queue, cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr)
{
     // size_t global_size[2];
    size_t global_size;
    cl_event kernel_event;
    cl_mem *now_input_buff = &input_buff;
    for(int i = 0; i < mlp_data_ptr->num_layers; ++i)
    {
        // global_size[0] = mlp_data_ptr->rows[i];
        // global_size[1] = mlp_data_ptr->cols[i]; 
        global_size = mlp_data_ptr->rows[i];  // 每个 work-item 负责一行        
        if(i == mlp_data_ptr->num_layers - 1) //最后一层只有gemm，且直接输出到Output
        {
            clSetKernelArg(pure_mat_kernel, 0, sizeof(cl_mem), now_input_buff);
            //设置 kernel 参数
            clSetKernelArg(pure_mat_kernel, 1, sizeof(cl_mem), &mlp_data_ptr->weight_buff[i]);
            clSetKernelArg(pure_mat_kernel, 2, sizeof(cl_mem), &mlp_data_ptr->bias_buff[i]);
            clSetKernelArg(pure_mat_kernel, 3, sizeof(cl_mem), &output_buff);
            clSetKernelArg(pure_mat_kernel, 4, sizeof(int), &mlp_data_ptr->rows[i]);
            clSetKernelArg(pure_mat_kernel, 5, sizeof(int), &mlp_data_ptr->cols[i]);
            err = (clEnqueueNDRangeKernel)(queue, pure_mat_kernel, 1, NULL,
                                   &global_size, NULL, 0, NULL, 
            &kernel_event);
            // 等待 kernel 完成
            clWaitForEvents(1, &kernel_event);
        }
        else
        {
            //先进行矩阵运算
            clSetKernelArg(pure_mat_kernel, 0, sizeof(cl_mem), now_input_buff);
            //设置 kernel 参数
            clSetKernelArg(pure_mat_kernel, 1, sizeof(cl_mem), &mlp_data_ptr->weight_buff[i]);
            clSetKernelArg(pure_mat_kernel, 2, sizeof(cl_mem), &mlp_data_ptr->bias_buff[i]);
            clSetKernelArg(pure_mat_kernel, 3, sizeof(cl_mem), &temp_output_buff[0]);    
            clSetKernelArg(pure_mat_kernel, 4, sizeof(int), &mlp_data_ptr->rows[i]);
            clSetKernelArg(pure_mat_kernel, 5, sizeof(int), &mlp_data_ptr->cols[i]);
            err = (clEnqueueNDRangeKernel)(queue, pure_mat_kernel, 1, NULL,
                                   &global_size, NULL, 0, NULL, 
            &kernel_event);
            // 等待 kernel 完成
            clWaitForEvents(1, &kernel_event);

            //再进行 ori 操作
            
            clSetKernelArg(ori_kernel, 0, sizeof(cl_mem), &temp_output_buff[0]);
            //设置 kernel 参数
            clSetKernelArg(ori_kernel, 1, sizeof(cl_mem), &mlp_data_ptr->mul_buff[i]);
            clSetKernelArg(ori_kernel, 2, sizeof(cl_mem), &mlp_data_ptr->add_buff[i]);

            clSetKernelArg(ori_kernel, 3, sizeof(cl_mem), &temp_output_buff[1]);
            clSetKernelArg(ori_kernel, 4, sizeof(int), &mlp_data_ptr->rows[i]);
            err = (clEnqueueNDRangeKernel)(queue, ori_kernel, 1, NULL,
                                    &global_size, NULL, 0, NULL, 
            &kernel_event);

            // 等待 kernel 完成
            clWaitForEvents(1, &kernel_event);
            now_input_buff = &temp_output_buff[1];
        }
      
    }
}

void CustomKernel::CustomKernelPrivate::inference_with_slice_changed(cl_command_queue &queue, cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr, int slice_position)
{
    // size_t global_size[2];
    size_t global_size;
    cl_event kernel_event;
    bool temp_out_id = 0;
    cl_mem *now_input_buff = &input_buff;
    for(int i = 0; i < mlp_data_ptr->num_layers; ++i)
    {
        // global_size[0] = mlp_data_ptr->rows[i];
        // global_size[1] = mlp_data_ptr->cols[i]; 
        global_size = mlp_data_ptr->rows[i];  // 每个 work-item 负责一行        
        if(i == mlp_data_ptr->num_layers - 1) //最后一层只有gemm，且直接输出到Output
        {
            clSetKernelArg(mat_slice_kernel, 0, sizeof(cl_mem), now_input_buff);
            //设置 kernel 参数
            clSetKernelArg(mat_slice_kernel, 1, sizeof(cl_mem), &mlp_data_ptr->weight_buff[i]);
            clSetKernelArg(mat_slice_kernel, 2, sizeof(cl_mem), &mlp_data_ptr->bias_buff[i]);
            clSetKernelArg(mat_slice_kernel, 3, sizeof(cl_mem), &output_buff);
            clSetKernelArg(mat_slice_kernel, 4, sizeof(int), &mlp_data_ptr->rows[i]);
            clSetKernelArg(mat_slice_kernel, 5, sizeof(int), &mlp_data_ptr->cols[i]);
            clSetKernelArg(mat_slice_kernel, 6, sizeof(int), &slice_position);
            err = (clEnqueueNDRangeKernel)(queue, mat_slice_kernel, 1, NULL,
                                   &global_size, NULL, 0, NULL, 
            &kernel_event);
            // 等待 kernel 完成
            clWaitForEvents(1, &kernel_event);
        }
        else
        {
            clSetKernelArg(mat_elu_kernel, 0, sizeof(cl_mem), now_input_buff);
            //设置 kernel 参数
            clSetKernelArg(mat_elu_kernel, 1, sizeof(cl_mem), &mlp_data_ptr->weight_buff[i]);
            clSetKernelArg(mat_elu_kernel, 2, sizeof(cl_mem), &mlp_data_ptr->bias_buff[i]);
            clSetKernelArg(mat_elu_kernel, 3, sizeof(cl_mem), &temp_output_buff[temp_out_id]);
            clSetKernelArg(mat_elu_kernel, 4, sizeof(int), &mlp_data_ptr->rows[i]);
            clSetKernelArg(mat_elu_kernel, 5, sizeof(int), &mlp_data_ptr->cols[i]);
            err = (clEnqueueNDRangeKernel)(queue, mat_elu_kernel, 1, NULL,
                                    &global_size, NULL, 0, NULL, 
            &kernel_event);

            // 等待 kernel 完成
            clWaitForEvents(1, &kernel_event);
            now_input_buff = &temp_output_buff[temp_out_id],temp_out_id = !temp_out_id;
        }
      
    }
}