#include "OnnxLoader.h"
#include "onnx.pb.h"
#include <fstream>
#include <iostream>

using namespace handsome;

void OnnxLoader::load_model(std::string filename)
{
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open model file" << std::endl;
        return;
    }

    onnx::ModelProto onnx_model;
    if (!onnx_model.ParseFromIstream(&fin)) {
        std::cerr << "Failed to parse model" << std::endl;
        return;
    }

     // 获取 graph
    const onnx::GraphProto& graph = onnx_model.graph();

    // 遍历 initializer（权重参数）
    for (const auto& tensor : graph.initializer()) {
        const std::string& name = tensor.name();
        const std::string& raw  = tensor.raw_data();
        const float* src = reinterpret_cast<const float*>(raw.data());
        size_t num_elem  = raw.size() / sizeof(float);

        // 判断是 weight 还是 bias
        bool is_weight = (name.find("weight") != std::string::npos);
        bool is_bias   = (name.find("bias")   != std::string::npos);

        if (!is_weight && !is_bias) 
        {
            std::cerr << "[ERROR] detect parameter " << name << " is not a mlp weight/bias, loading parameter may not be true!";
            continue;
        }

        int layer_id = -1;
        {
            auto pos1 = name.find('.');
            auto pos2 = name.find('.', pos1 + 1);
            if (pos1 != std::string::npos && pos2 != std::string::npos) {
                layer_id = std::stoi(name.substr(pos1 + 1, pos2 - pos1 - 1)) / 2;
            }
        }

        if (layer_id < 0) {
            std::cerr << "Unrecognized tensor name: " << name << "\n";
            continue;
        }
        if (is_weight) {
            if ((int)weights.size() <= layer_id)   weights.resize(layer_id + 1),rows.resize(layer_id + 1),cols.resize(layer_id + 1);

            // ONNX tensor 是按行存储 (out_dim × in_dim)
            rows[layer_id] =  tensor.dims(0),cols[layer_id] = tensor.dims(1);

            if (num_elem != (size_t)(rows[layer_id] * cols[layer_id])) {
                std::cerr << "Shape mismatch for " << name << "\n";
                continue;
            }

            weights[layer_id].assign(rows[layer_id], std::vector<float>(cols[layer_id]));
            for (int i = 0; i < rows[layer_id]; i++)
                for (int j = 0; j < cols[layer_id]; j++)
                    weights[layer_id][i][j] = src[i * cols[layer_id] + j];
        }
        else if (is_bias) {
            if ((int)biases.size() <= layer_id)
                biases.resize(layer_id + 1);

            int out_dim = tensor.dims(0);
            if (num_elem != (size_t)out_dim) {
                std::cerr << "Shape mismatch for " << name << "\n";
                continue;
            }

            biases[layer_id].assign(src, src + out_dim);
        }
    }
    num_layers = weights.size();
}