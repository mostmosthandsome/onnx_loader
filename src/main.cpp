#define _CRT_SECURE_NO_WARNINGS
#include "ExploreVaeRunner.h"
#include "PolicyHtOriRunner.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <thread>

using namespace std;
int main()
{
  float input[288],output[12];
  for(int i = 0; i < 288; ++i)  input[i] = i * 0.1;
  handsome::PolicyHtOriRunner test_kernel;

  test_kernel.load_onnx_model("config/policy_ht_ori.onnx");


  clock_t start, end;
  double cpu_time_used = 0;
  int test_case_num = 10;
  for(int i = 0; i < test_case_num; ++i)
  {
    start = clock();   // ==== 开始计时 ====
    test_kernel.inference(input, output);
    end = clock();   // ==== 结束计时 ====

    this_thread::sleep_for(chrono::milliseconds(200));
    cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;
  }
    
  // float std_out[12] = {
  //     -0.8302960395812988, 0.39889413118362427, -1.033607006072998, 0.0012681186199188232,
  //     -0.26702144742012024, -0.25018367171287537, -0.5394785404205322, -0.1281868815422058,
  //     0.24171936511993408, 0.3140263557434082, -0.9361374378204346, -0.1058325320482254
  // };
  // float sum_abs_err = 0, sum_sq_err = 0;
  // for (int i = 0; i < 12; ++i) {
  //   float err = output[i] - std_out[i];
  //   sum_abs_err += std::fabs(err);
  // }
  // std::cout << "Mean abs error: " << sum_abs_err / 12 << std::endl;
  printf("Total execution time: %f ms\n", cpu_time_used / test_case_num);
  std::cout << "output : ";
  for(int i = 0; i < 12; ++i) std::cout << output[i] << ',';
  putchar(10);
  return 0;

}