#include <iostream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

cl_device_id device = NULL;
cl_context context = NULL;
cl_command_queue queue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_platform_id platform = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret;


float output[12];


void build_platform()
{
    cl_int err;

    err = clGetPlatformIDs(1, &platform, &ret_num_platforms);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &ret_num_devices);

    /* 创建 OpenCL 上下文 */
    context = clCreateContext( NULL, 1, &device, NULL, NULL, &err);

    /* 创建命令队列 */
    queue = clCreateCommandQueue(context, device, 0, &err);
}

void load_openCL_code(std::string func_name)
{
    cl_int err;

    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size; 


    /* Read program file and place content into buffer */
    program_handle = fopen("./src/mat_op.cl", "r");
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
    program = clCreateProgramWithSource(context, 1, 
        (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        perror("Couldn't create the program");
        exit(1);   
    }
    free(program_buffer);

    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    /* Create kernel for the kernel function */
    kernel = clCreateKernel(program, func_name.c_str(), &err);
    
    if(err < 0) {
        perror("Couldn't create the mat kernel");
        exit(1);   
    }
}

void test()
{
    float input[12];
    float mul[12],add[12];
    for(int i = 0; i < 12; ++i)    input[i] = i * 0.64;
    for(int i = 0; i < 12; ++i)    mul[i] = 1.0,add[i] = 0.0;
    cl_mem input_buff,output_buff, mul_buff,add_buff;

    int output_dim = 12;

    //create buff
    cl_int err;
    input_buff = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * output_dim,
        input,
        &err
    );

    output_buff = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * output_dim,
        NULL, &err
    );


    mul_buff = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * output_dim,
        mul,
        &err
    );

    add_buff = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * output_dim,
        add,
        &err
    );



    //set arg
    size_t global_size = output_dim;
    cl_event kernel_event;

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buff);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &mul_buff);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &add_buff);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_buff);
    ret = clSetKernelArg(kernel, 4, sizeof(int), &output_dim);


    err = (clEnqueueNDRangeKernel)(queue, kernel, 1, NULL,
                                   &global_size, NULL, 0, NULL, 
            &kernel_event);
            // 等待 kernel 完成
    clWaitForEvents(1, &kernel_event);

    //read_output

    err = clEnqueueReadBuffer(
        queue,
        output_buff,
        CL_TRUE,
        0,
        sizeof(float) * output_dim,
        output,
        0,
        NULL,
        NULL
    );
    if (err < 0)
    {
        std::cout << "err = " << err << std::endl;
        perror("Couldn't read output buffer"); exit(1);
    }

    
    ret = clReleaseMemObject(input_buff);
    ret = clReleaseMemObject(output_buff);
}

int main()
{
   
    std::string fuc_name = "test_kernel";
    build_platform();
    load_openCL_code(fuc_name);

    
    test();
    std::cout << "output:" << std::endl;
    for(int i = 0; i < 12; ++i) std::cout << output[i] << ',';
    putchar(10);
    /* 終了処理 */
    ret = clFlush(queue);//将命令队列中的命令发往设备
    ret = clFinish(queue);//等待命令执行完成。
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);

    ret = clReleaseCommandQueue(queue);
    ret = clReleaseContext(context);


  return 0;
}