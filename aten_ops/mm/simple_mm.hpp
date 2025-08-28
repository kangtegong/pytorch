// Simple standalone matrix multiplication implementation without PyTorch dependencies
#include <vector>
#include <stdexcept>
#include <iostream>

namespace aten_ops {

template<typename T>
std::vector<std::vector<T>> mm(
    const std::vector<std::vector<T>>& A,
    const std::vector<std::vector<T>>& B) {
    
    if (A.empty() || B.empty() || A[0].empty()) {
        throw std::invalid_argument("Input matrices cannot be empty");
    }
    
    size_t m = A.size();      // rows of A
    size_t k = A[0].size();   // cols of A / rows of B
    size_t n = B[0].size();   // cols of B
    
    if (B.size() != k) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    // Verify all rows have consistent dimensions
    for (const auto& row : A) {
        if (row.size() != k) {
            throw std::invalid_argument("Matrix A has inconsistent row sizes");
        }
    }
    for (const auto& row : B) {
        if (row.size() != n) {
            throw std::invalid_argument("Matrix B has inconsistent row sizes");
        }
    }
    
    // Initialize result matrix
    std::vector<std::vector<T>> result(m, std::vector<T>(n, T(0)));
    
    // Perform matrix multiplication
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t ki = 0; ki < k; ++ki) {
                result[i][j] += A[i][ki] * B[ki][j];
            }
        }
    }
    
    return result;
}

} // namespace aten_ops