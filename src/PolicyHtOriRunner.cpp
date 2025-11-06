#include "PolicyHtOriRunner.h"

using namespace handsome;

PolicyHtOriRunner::PolicyHtOriRunner()
{
    kernel.load_openCL_code("src/mat_op.cl");
}

PolicyHtOriRunner::~PolicyHtOriRunner()
{
    cl_int err;
    stop_flag = true;
    encode_ready_flag = true;
    v.notify_one();
    encode_thread.join();

    err = clReleaseMemObject(input1_buff), err = clReleaseMemObject(input2_buff);
    err = clReleaseMemObject(cenet_encoder_out_buff);
    err = clReleaseMemObject(latent_mu_out_buff);
    err = clReleaseMemObject(final_out_buff);
    err = clReleaseMemObject(input1_buff);
    err = clReleaseMemObject(actor_in_buff);
}


void PolicyHtOriRunner::load_onnx_model(std::string file_name)
{
    cl_int err;
    //create model
    model_ptr = std::make_shared<OnnxLoader>(file_name);
    cenet_encoder_ptr  = std::make_shared<MlpDataMemory>();
    latent_mu_ptr = std::make_shared<MlpDataMemory>();
    actor_body_ptr    = std::make_shared<MlpDataMemory>();

    
    //load the params of mlp 
    kernel.load_mlp_params(model_ptr, cenet_encoder_ptr, "cenet_encoder");
    kernel.load_mlp_params(model_ptr, latent_mu_ptr, "latent_mu");
    kernel.load_mlp_params(model_ptr, actor_body_ptr, "actor_body");
    std::cout << "load params finished\n";
    //create intermediate temp buff
    
    cenet_encoder_out_buff = clCreateBuffer(kernel.context, CL_MEM_READ_WRITE,
        sizeof(float) * cenet_encoder_ptr->output_dim,
        NULL, &err
    );    
    latent_mu_out_buff = clCreateBuffer(kernel.context, CL_MEM_READ_WRITE,
        sizeof(float) * latent_mu_ptr->output_dim,
        NULL, &err
    );

    final_out_buff = clCreateBuffer(kernel.context, CL_MEM_READ_WRITE,
        sizeof(float) * actor_body_ptr->output_dim,
        NULL, &err
    );

    actor_in_buff = clCreateBuffer(kernel.context, CL_MEM_READ_WRITE,
        sizeof(float) * actor_body_ptr->input_dim,
        NULL, &err
    );

    input1_dim = 48, input2_dim = cenet_encoder_ptr->input_dim;//考虑到写起来太麻烦了，这里直接用数字写死
    output_dim = actor_body_ptr->output_dim;

    stop_flag = false;
    history_input.resize(input2_dim,0.0f);
    encode_ready_flag = true, inference_ready_flag = false;
    encode_thread = std::thread(&handsome::PolicyHtOriRunner::encode, this); 
    printf("ONNX model loaded successfully.\n");   
}


void PolicyHtOriRunner::inference(float input[], float output[])
{
    cl_int err;
    std::unique_lock<std::mutex> lock(mtx);
    if(!inference_ready_flag)   v.wait(lock);
    float obs[input1_dim];
    for(int i = 0; i < input1_dim; ++i) obs[i] = input[i];
    // ==== 阶段 1: 创建输入 buffer ====
    input1_buff = clCreateBuffer(
        kernel.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * input1_dim,
        obs,
        &err
    );
    if (err < 0) { perror("Couldn't create input1 buffer"); exit(1); }

    // ==== 阶段 4: 拼接 + Actor MLP ====
    kernel.concat(
        input1_buff,
        input1_dim,
        latent_mu_out_buff,
        latent_mu_ptr->output_dim,
        actor_in_buff
    );
    // //debug
    // {
    //     float temp[67];
    //     err = clEnqueueReadBuffer(
    //         kernel.queue,
    //         actor_in_buff,
    //         CL_TRUE,
    //         0,
    //         sizeof(float) * 67,
    //         temp,
    //         0,
    //         NULL,
    //         NULL
    //     );
    //     if (err < 0)
    //     {
    //         std::cout << "err = " << err << std::endl;
    //         perror("Couldn't read output buffer"); exit(1);
    //     }
    //     std::cout << "concat result:\n "; 
    //     for(int i = 0; i < 67; ++i)     std::cout << temp[i] << ", ";
    //     std::cout << "\n\n";

    // }

    kernel.inference_ori_actor(actor_in_buff, final_out_buff, actor_body_ptr);
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
    for(int i = 0; i < input2_dim; ++i)   history_input[i] = input[i];
    inference_ready_flag = false, encode_ready_flag = true;
    v.notify_one();
}



void PolicyHtOriRunner::encode()
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
        kernel.inference_mlp(input2_buff, cenet_encoder_out_buff, cenet_encoder_ptr);
        // ==== 阶段 2: Body Velocity MLP ====
        kernel.inference_with_slice_changed(cenet_encoder_out_buff, latent_mu_out_buff, latent_mu_ptr, 3);
        encode_ready_flag = false, inference_ready_flag = true;
        v.notify_one();
    }

}


