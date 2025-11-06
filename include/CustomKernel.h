#pragma once

#include <memory>
#include <string>
#ifdef MAC
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "OnnxLoader.h"
namespace handsome
{

class MlpDataMemory
{
public:
    ~MlpDataMemory();
    int num_layers;
    std::vector<cl_mem> weight_buff,bias_buff,mul_buff,add_buff;
    std::vector<int> rows,cols;
    int input_dim,output_dim;
};

class CustomKernel
{
public:
    CustomKernel();
    ~CustomKernel();
    /**
     * @brief compile the func_name in file_name
    */
    void load_openCL_code(std::string file_name);

    //for mlp load and inference
    /**
     * @brief load the mlp_name weights and params to mlp_ptr from model_ptr
    */
    void load_mlp_params(std::shared_ptr<OnnxLoader> model_ptr, std::shared_ptr<MlpDataMemory> mlp_ptr,std::string mlp_name);

    void concat(cl_mem &src1, int size1, cl_mem &src2, int size2, cl_mem &src3, int size3, cl_mem &dst);

    void concat(cl_mem &src1, int size1, cl_mem &src2, int size2, cl_mem &dst);

    /**
     * @brief do a inference of mlp_data_ptr, output will be put in output_buff
     # TODO 加入自动识别功能，自动识别最后一层是不是elu
    */

    void inference_mlp(cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr);

    void inference_single_end_clip(cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr, float clip_limit);

    void inference_ori_actor(cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr);

    /**
     * @brief after a mlp, put slice (0, slice_position) to the last position
    */
    void inference_with_slice_changed(cl_mem &input_buff, cl_mem &output_buff, std::shared_ptr<MlpDataMemory> mlp_data_ptr, int slice_position);
public:
    class CustomKernelPrivate;

    cl_context context{NULL};
    cl_command_queue queue{NULL};

private:
    std::unique_ptr<CustomKernelPrivate> data_ptr;
};
}
