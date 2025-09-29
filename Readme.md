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

# How to customize your own network Kernel

使用方法可以查看main.cpp, 很简单
将你的模型路径放进load_onnx_model的参数里即可
main分支里，请确保你的网络是一个mlp，然后查看它的名字（也可以通过test文件夹下的model_check查看）

将CustomKernel.cpp第183行的网络名字修改为你的mlp名字，重新编译即可
