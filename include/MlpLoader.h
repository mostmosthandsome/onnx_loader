#include <string>
#include <vector>

namespace handsome
{
    class MlpLoader
    {
    public:
        MlpLoader()=default;
        void load_model(std::string file_path);

    public:
        // 每一层的 weight[i] 对应一个二维矩阵，bias[i] 对应一个一维向量
        std::vector<std::vector<std::vector<float>>> weights; // shape: [num_layers][out_dim][in_dim]
        std::vector<std::vector<float>> biases;               // shape: [num_layers][out_dim]

        std::vector<int> rows;  // 每层输出神经元数
        std::vector<int> cols;  // 每层输入神经元数
        int num_layers;
    };
}
