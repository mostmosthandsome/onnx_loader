# Dependencies
protobuf ( I use v32.1)
Could be installed using the folling instruction
```Bash
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
cmake -S . -B build  -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17  -Dprotobuf_BUILD_TESTS=OFF
cmake --build build
```
can also be found at https://zhuanlan.zhihu.com/p/1949584039087441208

# How to Compile
ensure you has cmake
use the following instructions
```
cmake -B build
cmake --build build
```

# How to customize your own network

请确保你的网络里只有由mlp和elu构成的mlp网络（是因为我只对这种进行了封装），不同mlp网络之间的连接可以用concat，clip连接。
做到这些后，你要修改的地方如下
1.阅读src文件夹下的main.cpp，对使用方法有一个简单的了解
2.将你的onnx模型路径粘贴到load_onnx_model函数里
3.修改src/CustomKernel.cpp下的load_onnx_model函数，用其中data_ptr->load_mlp_params将mlp网络参数全部加载进来，记得顺便修改一下CustomKernel::CustomKernelPrivate里的mlp_param指针名字。
4.修改src/CustomKernel.cpp下的CustomKernel::inference函数，inference路径按照你自己的网络进行设置,;
如果你有多个input，记得搜索一下，把对应的input有关的地方都改一下,input_buff和input_dim应该都要改
5.注意一下你网络里最大的中间结果不应超过512，如果超过，请
寻找以下代码：
data_ptr->relu_output_buff = clCreateBuffer(
        data_ptr->context,
        CL_MEM_READ_WRITE,
        sizeof(float) * 512,
        NULL,
        &data_ptr->err
    );
将其中的512改为你的最大中间结果（同理还有gemm_output_buff）