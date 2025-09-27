#include "onnx.pb.h"
#include <fstream>

int main()
{
    std::ifstream fin("config/explore_vae.onnx", std::ios::in | std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open model file" << std::endl;
        return;
    }

    onnx::ModelProto onnx_model;
    if (!onnx_model.ParseFromIstream(&fin)) {
        std::cerr << "Failed to parse model" << std::endl;
        return;
    }

    const onnx::GraphProto& graph = onnx_model.graph();

    std::cout << "Model has " << graph.input_size() << " inputs:" << std::endl;
    for (const auto& input : graph.input()) {
        std::cout << "  input name : " << input.name() << std::endl;
    }

    for (const auto& tensor : graph.initializer()) {
        const std::string& name = tensor.name();
        std::cout << "layer name : " << name << std::endl;

    }
}