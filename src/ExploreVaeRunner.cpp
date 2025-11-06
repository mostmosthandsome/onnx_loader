#include "ExploreVaeRunner.h"

using namespace handsome;

ExploreVaeRunner::ExploreVaeRunner()
{
    kernel.load_openCL_code("src/mat_op.cl");
}

ExploreVaeRunner::~ExploreVaeRunner()
{
    cl_int err;
    stop_flag = true;
    encode_ready_flag = true;
    v.notify_one();
    encode_thread.join();

    err = clReleaseMemObject(input1_buff),err = clReleaseMemObject(input2_buff);
    err = clReleaseMemObject(encoder_out_buff);
    err = clReleaseMemObject(body_vel_out_buff);
    err = clReleaseMemObject(fc_mu_out_buff);
    err = clReleaseMemObject(fc_mu_out_buff);
    err = clReleaseMemObject(final_out_buff);
}


void ExploreVaeRunner::load_onnx_model(std::string file_name)
{
    cl_int err;
    //create model
    model_ptr = std::make_shared<OnnxLoader>(file_name);
    encoder_ptr  = std::make_shared<MlpDataMemory>();
    body_vel_ptr = std::make_shared<MlpDataMemory>();
    fc_mu_ptr    = std::make_shared<MlpDataMemory>();
    actor_ptr    = std::make_shared<MlpDataMemory>();

    
    //load the params of mlp 
    kernel.load_mlp_params(model_ptr, actor_ptr, "actor");
    kernel.load_mlp_params(model_ptr, encoder_ptr, "encoder");
    kernel.load_mlp_params(model_ptr, body_vel_ptr, "est_explicit_layers.body_vel_buf");
    kernel.load_mlp_params(model_ptr, fc_mu_ptr, "fc_mu");
    std::cout << "load params finished\n";
    //create intermediate temp buff
    
    actor_in_buff = clCreateBuffer(kernel.context, CL_MEM_READ_WRITE,
        sizeof(float) * actor_ptr->input_dim,
        NULL, &err
    );

    final_out_buff = clCreateBuffer(kernel.context, CL_MEM_READ_WRITE,
        sizeof(float) * actor_ptr->output_dim,
        NULL, &err
    );

    input1_dim = 54;//考虑到写起来太麻烦了，这里直接用数字写死
    input2_dim = encoder_ptr->input_dim;
    output_dim = actor_ptr->output_dim;

    encoder_out_buff = clCreateBuffer(kernel.context, CL_MEM_READ_WRITE,
        sizeof(float) * encoder_ptr->output_dim,
        NULL, &err
    );
    body_vel_out_buff = clCreateBuffer(kernel.context, CL_MEM_READ_WRITE,
        sizeof(float) * body_vel_ptr->output_dim,
        NULL, &err
    );
    fc_mu_out_buff = clCreateBuffer(kernel.context, CL_MEM_READ_WRITE,
        sizeof(float) * fc_mu_ptr->output_dim,
        NULL, &err
    );

    stop_flag = false;
    history_input.resize(input2_dim,0.0f);
    encode_ready_flag = true, inference_ready_flag = false;
    encode_thread = std::thread(&handsome::ExploreVaeRunner::encode, this); 
    printf("ONNX model loaded successfully.\n");   
}


void ExploreVaeRunner::inference(float input1[], float input2[], float output[])
{
    cl_int err;
    std::unique_lock<std::mutex> lock(mtx);
    if(!inference_ready_flag)   v.wait(lock);
    // ==== 阶段 1: 创建输入 buffer ====
    input1_buff = clCreateBuffer(
        kernel.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * input1_dim,
        input1,
        &err
    );
    if (err < 0) { perror("Couldn't create input1 buffer"); exit(1); }

    // ==== 阶段 4: 拼接 + Actor MLP ====
    kernel.concat(
        input1_buff,
        input1_dim,
        body_vel_out_buff,
        body_vel_ptr->output_dim,
        fc_mu_out_buff,
        fc_mu_ptr->output_dim,
        actor_in_buff
    );

    kernel.inference_mlp(actor_in_buff, final_out_buff, actor_ptr);

    // ==== 阶段 5: 读回结果 ====
    err = clEnqueueReadBuffer(
        kernel.queue,
        final_out_buff,
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
    for(int i = input1_dim; i < input2_dim; ++i)    history_input[i] = input2[i];
    for(int i = 0; i < input1_dim; ++i)  history_input[i] = input1[i - input1_dim];
    //fresh history

    inference_ready_flag = false, encode_ready_flag = true;
    v.notify_one();
}



void ExploreVaeRunner::encode()
{
    cl_int err;
    while(!stop_flag)
    {
        std::unique_lock<std::mutex> lock(mtx);
        if(!encode_ready_flag)
        {
            v.wait(lock);
        }
        input2_buff = clCreateBuffer(
        kernel.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * input2_dim,
        history_input.data(),
        &err
        );
        if (err < 0) { perror("Couldn't create input2 buffer"); exit(1); }
        // ==== 阶段 1: Encoder MLP ====
        kernel.inference_mlp(input2_buff, encoder_out_buff, encoder_ptr);

        // ==== 阶段 2: Body Velocity MLP ====
        kernel.inference_single_end_clip(encoder_out_buff, body_vel_out_buff, body_vel_ptr,10.0);

        // ==== 阶段 3: FC_mu MLP ====
        kernel.inference_mlp(encoder_out_buff, fc_mu_out_buff, fc_mu_ptr);
        encode_ready_flag = false, inference_ready_flag = true;
        v.notify_one();
    }

}


