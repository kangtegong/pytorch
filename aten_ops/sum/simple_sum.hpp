// Simple standalone sum implementation without PyTorch dependencies
#include <vector>
#include <numeric>
#include <iostream>

namespace aten_ops {

template<typename T>
T sum(const std::vector<T>& data) {
    return std::accumulate(data.begin(), data.end(), T(0));
}

template<typename T>
std::vector<T> sum_along_axis(const std::vector<std::vector<T>>& matrix, int axis) {
    if (axis == 0) {
        // Sum along rows (result is a row vector)
        if (matrix.empty()) return {};
        
        std::vector<T> result(matrix[0].size(), T(0));
        for (const auto& row : matrix) {
            for (size_t i = 0; i < row.size(); ++i) {
                result[i] += row[i];
            }
        }
        return result;
    } else if (axis == 1) {
        // Sum along columns (result is a column vector)
        std::vector<T> result;
        for (const auto& row : matrix) {
            result.push_back(sum(row));
        }
        return result;
    }
    return {};
}

} // namespace aten_ops