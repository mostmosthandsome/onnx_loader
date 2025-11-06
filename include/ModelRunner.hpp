#pragma once
#include <string>
#include "CustomKernel.h"

namespace handsome
{

class ModelRunner
{
public:
    ModelRunner()
    {
        kernel.load_openCL_code("src/mat_op.cl");
    }

    virtual ~ModelRunner() {}

    /**
     * @brief load onnx model from specified path（the structure should match the model defined in OnnxLoader.h, and load the params into cl_mem
    */
    virtual void load_onnx_model(std::string file_name) = 0;

    /**
     * inference input to output, if has multi input, concat them to form input array
    */
    virtual void inference(float input[], float output[]) = 0;

protected:
    CustomKernel kernel;
};

}
