#pragma once
#include "CustomKernel.h"
#include <thread>
#include <condition_variable>
#ifdef MAC
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace handsome
{

class PolicyHtOriRunner
{
public:
    PolicyHtOriRunner();
    ~PolicyHtOriRunner();

    /**
     * @brief inference the model 
     * @param input the input vector, whose dimension should be 265
    */
    void inference(float input[], float output[]);

        /**
     * @brief encode the history_input and update the encoder_out_buff, body_vel_out_buff, fc_mu_out_buff
     */
    void encode();        
    
    /**
     * @brief load onnx model from specified path（the structure should match the model defined in OnnxLoader.h, and load the params into cl_mem
    */
    void load_onnx_model(std::string file_name);


private:
    //create a CustomKernel
    CustomKernel kernel;

    //used for thread optimization
    bool stop_flag;
    std::mutex mtx;
    std::condition_variable v;
    bool encode_ready_flag, inference_ready_flag;
    std::thread encode_thread;
    std::vector< float > history_input;

    //cl_mem preset
    cl_mem input1_buff,input2_buff;
    cl_mem cenet_encoder_out_buff, latent_mu_out_buff, final_out_buff, actor_in_buff;


    //model
    std::shared_ptr<OnnxLoader> model_ptr;
    //mlp_param
    std::shared_ptr<MlpDataMemory> cenet_encoder_ptr, latent_mu_ptr,actor_body_ptr;
    //dims
    int input1_dim,input2_dim,output_dim;    

};
}
