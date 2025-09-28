#include "OnnxLoader.h"
#include <fstream>
#include <iostream>

using namespace handsome;

OnnxLoader::OnnxLoader(std::string filename)
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
    graph_ptr = std::make_shared<onnx::GraphProto>(onnx_model.graph());
}

void OnnxLoader::load_mlp_param(std::shared_ptr<MlpParam> mlp_param,std::string mlp_name)
{
        // 遍历 initializer（权重参数）
    for (const auto& tensor : graph_ptr->initializer()) {
        const std::string& name = tensor.name();
        const std::string& raw  = tensor.raw_data();
        const float* src = reinterpret_cast<const float*>(raw.data());
        size_t num_elem  = raw.size() / sizeof(float);
        
        if (name.find(mlp_name + ".") != 0)   continue; // 必须以 mlp_name 开头

        // 判断是 weight 还是 bias
        bool is_weight = (name.find("weight") != std::string::npos);
        bool is_bias   = (name.find("bias")   != std::string::npos);
        
        if (!is_weight && !is_bias) continue;

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
            if ((int)mlp_param->weights.size() <= layer_id)
               mlp_param->weights.resize(layer_id + 1),mlp_param->rows.resize(layer_id + 1),mlp_param->cols.resize(layer_id + 1);

            // ONNX tensor 是按行存储 (out_dim × in_dim)
            mlp_param->rows[layer_id] =  tensor.dims(0),mlp_param->cols[layer_id] = tensor.dims(1);

            if (num_elem != (size_t)(mlp_param->rows[layer_id] * mlp_param->cols[layer_id])) {
                std::cerr << "Shape mismatch for " << name << "\n";
                continue;
            }

            mlp_param->weights[layer_id].assign(mlp_param->rows[layer_id], std::vector<float>(mlp_param->cols[layer_id]));
            for (int i = 0; i < mlp_param->rows[layer_id]; i++)
                for (int j = 0; j < mlp_param->cols[layer_id]; j++)
                    mlp_param->weights[layer_id][i][j] = src[i * mlp_param->cols[layer_id] + j];
        }
        else if (is_bias) {
            if ((int)mlp_param->biases.size() <= layer_id)
                mlp_param->biases.resize(layer_id + 1);

            int out_dim = tensor.dims(0);
            if (num_elem != (size_t)out_dim) {
                std::cerr << "Shape mismatch for " << name << "\n";
                continue;
            }

            mlp_param->biases[layer_id].assign(src, src + out_dim);
        }
    }
    mlp_param->num_layers = mlp_param->weights.size();
}