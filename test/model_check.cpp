#include "onnx.pb.h"
#include <fstream>
int main()
{
    std::ifstream fin("config/policy_ht_ori.onnx", std::ios::in | std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open model file" << std::endl;
        return -1;
    }

    onnx::ModelProto onnx_model;
    if (!onnx_model.ParseFromIstream(&fin)) {
        std::cerr << "Failed to parse model" << std::endl;
        return -1;
    }

    const onnx::GraphProto& graph = onnx_model.graph();

    // === 构建initializer索引 ===
    std::unordered_map<std::string, const onnx::TensorProto*> tensor_map;
    for (const auto& tensor : graph.initializer()) {
        tensor_map[tensor.name()] = &tensor;
    }

    // === 构建  initializer -> 节点名 的反向映射 ===
    std::unordered_map<std::string, std::vector<std::string>> tensor_to_node;
    for (const auto& node : graph.node()) {
        std::cout << "node " << node.name() << std::endl;
        for (const auto& input_name : node.input()) {
            if (tensor_map.count(input_name)) {
                tensor_to_node[input_name].push_back(node.name().empty() ? node.op_type() : node.name());
            }
        }
    }

    // === 打印每个initializer的信息 ===
    for (const auto& tensor : graph.initializer()) {
        const std::string& name = tensor.name();
        std::cout << "Tensor name: " << name << std::endl;

        std::cout << "  shape: [";
        for (int i = 0; i < tensor.dims_size(); ++i) {
            std::cout << tensor.dims(i);
            if (i != tensor.dims_size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  data_type: " << tensor.data_type() << std::endl;

        if (tensor_to_node.count(name)) {
            std::cout << "  used by node(s): (size = " << tensor_to_node[name].size() << ")\n";
            for (const auto& node_name : tensor_to_node[name]) {
                std::cout << node_name << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "  (unused initializer)" << std::endl;
        }

        std::cout << std::endl;
    }

    return 0;
}