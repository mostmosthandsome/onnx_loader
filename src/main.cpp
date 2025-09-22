#define _CRT_SECURE_NO_WARNINGS
#include "CustomKernel.h"
#include <iostream>
#include <iomanip>

using namespace std;
int main()
{
  float input[265],output[12];
  for(int i = 0; i < 265; ++i)  input[i] = 0.1;
  handsome::CustomKernel test_kernel;
  test_kernel.load_openCL_code("src/mat_op.cl");
  test_kernel.load_onnx_model("config/actor.onnx");
  test_kernel.inference(input,output);
  for(int i = 0; i < 12; ++i)  std::cout << fixed << setprecision(7) << output[i] << ' ';
  putchar(10);
  return 0;

}