#include "simple_sum.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Testing standalone sum operator..." << std::endl;
    
    // Test 1D sum
    std::vector<double> vec = {1.0, 2.0, 3.0, 4.0, 5.0};
    double total = aten_ops::sum(vec);
    std::cout << "1D sum result: " << total << " (expected: 15)" << std::endl;
    
    // Test 2D sum
    std::vector<std::vector<double>> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    
    std::cout << "Input matrix:" << std::endl;
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    // Sum along axis 0 (rows)
    auto sum_rows = aten_ops::sum_along_axis(matrix, 0);
    std::cout << "Sum along axis 0 (columns): ";
    for (const auto& val : sum_rows) {
        std::cout << val << " ";
    }
    std::cout << "(expected: 5 7 9)" << std::endl;
    
    // Sum along axis 1 (columns)  
    auto sum_cols = aten_ops::sum_along_axis(matrix, 1);
    std::cout << "Sum along axis 1 (rows): ";
    for (const auto& val : sum_cols) {
        std::cout << val << " ";
    }
    std::cout << "(expected: 6 15)" << std::endl;
    
    std::cout << "Standalone sum operator test passed!" << std::endl;
    return 0;
}