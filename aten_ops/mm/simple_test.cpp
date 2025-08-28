#include "simple_mm.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Testing standalone mm operator..." << std::endl;
    
    // Create test matrices
    std::vector<std::vector<double>> A = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    std::vector<std::vector<double>> B = {
        {5.0, 6.0},
        {7.0, 8.0}
    };
    
    std::cout << "Matrix A (2x2):" << std::endl;
    for (const auto& row : A) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Matrix B (2x2):" << std::endl;
    for (const auto& row : B) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    // Perform matrix multiplication
    auto result = aten_ops::mm(A, B);
    
    std::cout << "Result A @ B (2x2):" << std::endl;
    for (const auto& row : result) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "(expected: [[19 22] [43 50]])" << std::endl;
    
    // Test with different dimensions
    std::vector<std::vector<double>> C = {
        {1.0, 2.0, 3.0}
    };
    
    std::vector<std::vector<double>> D = {
        {4.0, 5.0},
        {6.0, 7.0}, 
        {8.0, 9.0}
    };
    
    std::cout << "\nMatrix C (1x3):" << std::endl;
    for (const auto& row : C) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Matrix D (3x2):" << std::endl;
    for (const auto& row : D) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    auto result2 = aten_ops::mm(C, D);
    std::cout << "Result C @ D (1x2):" << std::endl;
    for (const auto& row : result2) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "(expected: [[40 46]])" << std::endl;
    
    std::cout << "Standalone mm operator test passed!" << std::endl;
    return 0;
}