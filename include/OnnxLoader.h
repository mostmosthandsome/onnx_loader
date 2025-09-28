#include <string>
#include <vector>
#include <memory>
#include "onnx.pb.h"

namespace handsome
{
    class MlpParam
    {
    public:
        std::vector<std::vector<std::vector<float>>> weights; // shape: [num_layers][out_dim][in_dim]
        std::vector<std::vector<float>> biases;               // shape: [num_layers][out_dim]
        std::vector<int> rows;  // 每层输出神经元数
        std::vector<int> cols;  // 每层输入神经元数
        int num_layers;
    };

    class OnnxLoader
    {
    public:
        OnnxLoader()=default;
        void load_model(std::string file_path);
        void load_mlp_param(std::shared_ptr<MlpParam> mlp_param,std::string mlp_name);

    private:
        onnx::GraphProto& graph;

    };
}
