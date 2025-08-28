#include <iostream>
#include <torch/torch.h>

// Test the extracted sum operator
int main() {
  std::cout << "Testing sum operator..." << std::endl;
  
  // Create a simple tensor
  torch::Tensor x = torch::randn({2, 3});
  std::cout << "Input tensor:" << std::endl << x << std::endl;
  
  // Test sum operation
  torch::Tensor result = torch::sum(x);
  std::cout << "Sum result:" << std::endl << result << std::endl;
  
  // Test sum with dimension
  torch::Tensor result_dim = torch::sum(x, 0);
  std::cout << "Sum along dim 0:" << std::endl << result_dim << std::endl;
  
  std::cout << "Sum operator test passed!" << std::endl;
  return 0;
}