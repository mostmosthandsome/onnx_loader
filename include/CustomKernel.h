#include <memory>
#include <string>
#include "OnnxLoader.h"
namespace handsome
{
    class CustomKernel
    {
    public:
        CustomKernel();
        ~CustomKernel();
        /**
         * @brief compile the func_name in file_name
        */
        void load_openCL_code(std::string file_name);

        /**
         * @brief load onnx model from specified path（the structure should match the model defined in OnnxLoader.h, and load the params into cl_mem
        */
        void load_onnx_model(std::string file_name);

        /**
         * @brief inference the model 
         * @param input the input vector, whose dimension should be 265
        */
        void inference(float input[],float output[]);

    public:
        class CustomKernelPrivate;

    private:
        std::unique_ptr<CustomKernelPrivate> data_ptr;
        std::shared_ptr<OnnxLoader> model_ptr;
    };
}
