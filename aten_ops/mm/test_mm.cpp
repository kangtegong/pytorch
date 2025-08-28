#include <iostream>
#include <torch/torch.h>

// Test the extracted mm operator
int main() {
  std::cout << "Testing mm operator..." << std::endl;
  
  // Create simple tensors for matrix multiplication
  torch::Tensor a = torch::randn({2, 3});
  torch::Tensor b = torch::randn({3, 4});
  std::cout << "Tensor A (2x3):" << std::endl << a << std::endl;
  std::cout << "Tensor B (3x4):" << std::endl << b << std::endl;
  
  // Test matrix multiplication
  torch::Tensor result = torch::mm(a, b);
  std::cout << "Result A @ B (2x4):" << std::endl << result << std::endl;
  
  std::cout << "MM operator test passed!" << std::endl;
  return 0;
}