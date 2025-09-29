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