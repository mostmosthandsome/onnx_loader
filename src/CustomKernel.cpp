#include "CustomKernel.h"
#ifdef MAC
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>

using namespace handsome;

class CustomKernel::CustomKernelPrivate
{
public:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context{NULL};
    cl_command_queue queue{NULL};
    cl_program program{NULL};
    cl_kernel mat_kernel{NULL},elu_kernel{NULL};
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int err;
    cl_mem input_buff,output_buff1,output_buff2;
    cl_mem weight_buff[4],bias_buff[4];
};

CustomKernel::CustomKernel():data_ptr(std::make_unique<CustomKernelPrivate>())
{
    /* 获取平台设备信息 */
    data_ptr->err = clGetPlatformIDs(1, &data_ptr->platform, &data_ptr->ret_num_platforms);
    data_ptr->err = clGetDeviceIDs(data_ptr->platform, CL_DEVICE_TYPE_GPU, 1, &data_ptr->device, &data_ptr->ret_num_devices);

    /* 创建 OpenCL 上下文 */
    data_ptr->context = clCreateContext( NULL, 1, &data_ptr->device, NULL, NULL, &data_ptr->err);

    /* 创建命令队列 */
    data_ptr->queue = clCreateCommandQueue(data_ptr->context, data_ptr->device, 0, &data_ptr->err);

    model_ptr = std::make_shared<OnnxLoader>();

    data_ptr->output_buff1 = clCreateBuffer(
        data_ptr->context,
        CL_MEM_READ_WRITE,
        sizeof(float) * 512,
        NULL,
        &data_ptr->err
    );
    data_ptr->output_buff2 = clCreateBuffer(
        data_ptr->context,
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
  data_ptr->err = clFlush(data_ptr->queue);
  data_ptr->err = clFinish(data_ptr->queue);
  data_ptr->err = clReleaseKernel(data_ptr->mat_kernel);
  data_ptr->err = clReleaseProgram(data_ptr->program);

  data_ptr->err = clReleaseCommandQueue(data_ptr->queue);
  data_ptr->err = clReleaseContext(data_ptr->context);

  for(int i = 0; i < 4; ++i)  data_ptr->err = clReleaseMemObject(data_ptr->weight_buff[i]);
  for(int i = 0; i < 4; ++i)  data_ptr->err = clReleaseMemObject(data_ptr->bias_buff[i]);
  data_ptr->err = clReleaseMemObject(data_ptr->input_buff);
  data_ptr->err = clReleaseMemObject(data_ptr->output_buff1),data_ptr->err = clReleaseMemObject(data_ptr->output_buff2);
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
    data_ptr->program = clCreateProgramWithSource(data_ptr->context, 1, 
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
    data_ptr->mat_kernel = clCreateKernel(data_ptr->program, "mat_kernel", &data_ptr->err);
    
    if(data_ptr->err < 0) {
        perror("Couldn't create the mat kernel");
        exit(1);   
    }

    data_ptr->elu_kernel = clCreateKernel(data_ptr->program, "elu_kernel", &data_ptr->err);
    
    if(data_ptr->err < 0) {
        perror("Couldn't create the mat kernel");
        exit(1);   
    }

    return;
}

void CustomKernel::load_onnx_model(std::string file_name)
{
    model_ptr->load_model(file_name);
    
    // net.0.weight
    data_ptr->weight_buff[0] = clCreateBuffer(
        data_ptr->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 512 * 265,
        model_ptr->weight0,
        &data_ptr->err
    );
    if (data_ptr->err < 0) { perror("Couldn't create weight_buff[0]"); exit(1); }

    // net.0.bias
    data_ptr->bias_buff[0] = clCreateBuffer(
        data_ptr->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 512,
        model_ptr->bias0,
        &data_ptr->err
    );
    if (data_ptr->err < 0) { perror("Couldn't create bias_buff[0]"); exit(1); }

    // net.2.weight
    data_ptr->weight_buff[1] = clCreateBuffer(
        data_ptr->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 256 * 512,
        model_ptr->weight1,
        &data_ptr->err
    );
    if (data_ptr->err < 0) { perror("Couldn't create weight_buff[1]"); exit(1); }

    // net.2.bias
    data_ptr->bias_buff[1] = clCreateBuffer(
        data_ptr->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 256,
        model_ptr->bias1,
        &data_ptr->err
    );
    if (data_ptr->err < 0) { perror("Couldn't create bias_buff[1]"); exit(1); }

    // net.4.weight
    data_ptr->weight_buff[2] = clCreateBuffer(
        data_ptr->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 128 * 256,
        model_ptr->weight2,
        &data_ptr->err
    );
    if (data_ptr->err < 0) { perror("Couldn't create weight_buff[2]"); exit(1); }

    // net.4.bias
    data_ptr->bias_buff[2] = clCreateBuffer(
        data_ptr->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 128,
        model_ptr->bias2,
        &data_ptr->err
    );
    if (data_ptr->err < 0) { perror("Couldn't create bias_buff[2]"); exit(1); }

    // net.6.weight
    data_ptr->weight_buff[3] = clCreateBuffer(
        data_ptr->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 12 * 128,
        model_ptr->weight3,
        &data_ptr->err
    );
    if (data_ptr->err < 0) { perror("Couldn't create weight_buff[3]"); exit(1); }

    // net.6.bias
    data_ptr->bias_buff[3] = clCreateBuffer(
        data_ptr->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 12,
        model_ptr->bias3,
        &data_ptr->err
    );
    if (data_ptr->err < 0) { perror("Couldn't create bias_buff[3]"); exit(1); }

}


void CustomKernel::inference(float input[],float output[])
{
    // 创建输入的向量 buffer，并拷贝数据
    data_ptr->input_buff = clCreateBuffer(
        data_ptr->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * 265,
        input,
        &data_ptr->err
    );
    if (data_ptr->err < 0) { perror("Couldn't create vec buffer"); exit(1); }
    cl_event kernel_event;
    size_t global_size;
    
    clock_t start, end;
    double cpu_time_used;

    start = clock();   // ==== 开始计时 ====
    for(int i = 0; i < 4; ++i)
    {
        
        global_size = model_ptr->rows[i];   // 每个 work-item 负责一行
        //设置 kernel 参数
        clSetKernelArg(data_ptr->mat_kernel, 0, sizeof(cl_mem), &data_ptr->weight_buff[i]);
        if(i == 0)  clSetKernelArg(data_ptr->mat_kernel, 1, sizeof(cl_mem), &data_ptr->input_buff);
        else        clSetKernelArg(data_ptr->mat_kernel, 1, sizeof(cl_mem), &data_ptr->output_buff1);
        clSetKernelArg(data_ptr->mat_kernel, 2, sizeof(cl_mem), &data_ptr->bias_buff[i]);
        clSetKernelArg(data_ptr->mat_kernel, 3, sizeof(cl_mem), &data_ptr->output_buff2);
        clSetKernelArg(data_ptr->mat_kernel, 4, sizeof(int), &model_ptr->rows[i]);
        clSetKernelArg(data_ptr->mat_kernel, 5, sizeof(int), &model_ptr->cols[i]);
        data_ptr->err = clEnqueueNDRangeKernel(data_ptr->queue, data_ptr->mat_kernel, 1, NULL,
                                    &global_size, NULL, 0, NULL, 
            &kernel_event);
            
        // 等待 kernel 完成
        clWaitForEvents(1, &kernel_event);

        // --- 矩阵乘法结果 ---
        std::vector<float> host_out(model_ptr->rows[i]);

        // 从 device -> host 读回 output_buff2
        clEnqueueReadBuffer(data_ptr->queue, data_ptr->output_buff2,
                            CL_TRUE, 0,
                            sizeof(float) * model_ptr->rows[i],
                            host_out.data(), 0, NULL, NULL);

        if(i == 3) break;

        clSetKernelArg(data_ptr->elu_kernel, 0, sizeof(cl_mem), &data_ptr->output_buff2);
        clSetKernelArg(data_ptr->elu_kernel, 1, sizeof(cl_mem), &data_ptr->output_buff1);
        clSetKernelArg(data_ptr->elu_kernel, 2, sizeof(int), &model_ptr->rows[i]);
        data_ptr->err = clEnqueueNDRangeKernel(data_ptr->queue, data_ptr->elu_kernel, 1, NULL,
                                    &global_size, NULL, 0, NULL, 
            &kernel_event);


        // 等待 kernel 完成
        clWaitForEvents(1, &kernel_event);

        clEnqueueReadBuffer(data_ptr->queue, data_ptr->output_buff1,
                    CL_TRUE, 0,
                    sizeof(float) * model_ptr->rows[i],
                    host_out.data(), 0, NULL, NULL);

      
    }

    end = clock();   // ==== 结束计时 ====

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0; // 毫秒
    printf("Total execution time: %f ms\n", cpu_time_used);
    // 6. 读回结果
    data_ptr->err = clEnqueueReadBuffer(
      data_ptr->queue,
      data_ptr->output_buff2,
      CL_TRUE,
      0,
      sizeof(float) * 12,
      output,
      0,
      NULL,
      NULL
  );

}
