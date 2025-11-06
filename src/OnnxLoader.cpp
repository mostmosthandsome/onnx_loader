#include "OnnxLoader.h"
#include <fstream>
#include <iostream>
#include <iomanip>

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
    std::cout << "loading mlp " << mlp_name << std::endl;
    int cnt = -1;
        // 遍历 initializer（权重参数）
    
    // === 构建initializer索引 ===
    std::unordered_map<std::string, const onnx::TensorProto*> tensor_map;
    for (const auto& tensor : graph_ptr->initializer()) {
        tensor_map[tensor.name()] = &tensor;
    }

    // === 构建  initializer -> 节点名 的反向映射 ===
    std::unordered_map<std::string, std::vector<std::string>> tensor_to_node;
    for (const auto& node : graph_ptr->node()) {
        for (const auto& input_name : node.input()) {
            if (tensor_map.count(input_name)) {
                tensor_to_node[input_name].push_back(node.name().empty() ? node.op_type() : node.name());
            }
        }
    }

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
        if(tensor_to_node[name].size()  == 1 && tensor_to_node[name][0].find("Mul") != std::string::npos)   //如果参数是作为mul的一部分
        {
            int out_dim = tensor.dims(0);
            if (num_elem != (size_t)out_dim) {
                std::cerr << "Shape mismatch for " << name << "\n";
                continue;
            }
            
            mlp_param->mul_param.resize(cnt + 1),mlp_param->mul_param[cnt].assign(src, src + out_dim);
            continue;
        }

        if(tensor_to_node[name].size()  == 1 && tensor_to_node[name][0].find("Add") != std::string::npos)   //如果参数是作为mul的一部分
        {
            int out_dim = tensor.dims(0);
            if (num_elem != (size_t)out_dim) {
                std::cerr << "Shape mismatch for " << name << "\n";
                continue;
            }
            mlp_param->add_param.resize(cnt + 1),mlp_param->add_param[cnt].assign(src, src + out_dim);
            continue;
        }

        int layer_id = 0;
        if (is_weight) {
            layer_id = ++cnt;
            if ((int)(mlp_param->weights.size()) <= layer_id)
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
            layer_id = cnt;
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