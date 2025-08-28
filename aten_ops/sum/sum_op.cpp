#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sum_native.h>
#include <ATen/ops/nansum_native.h>
#endif

namespace at {
namespace native {

// Declare stubs defined elsewhere
DECLARE_DISPATCH(void, sum_stub);
DECLARE_DISPATCH(void, nansum_stub);

// Helper function to check if accumulation buffer should be used
static inline bool should_use_acc_buffer(const TensorIterator& iter) {
  const auto ndim = iter.ndim();
  // need acc_buffer when the input dtype is not the same as output dtype
  if (iter.dtype(1) != iter.dtype(0)) {
    return true;
  }
  // the typical case is that input and output dtypes are the same,
  // and input is fp32 / fp64, then we should not use acc_buffer
  if (iter.dtype(1) != at::kBFloat16 && iter.dtype(1) != at::kHalf) {
    return false;
  }
  // The condition for using acc_buffer for BFloat16/Half is that: for permuted-layout-input,
  // the output stride is [0, 0, x, x, ...] with x >= 0,
  // which means we are reducing two adjacent reduced dims.
  if (ndim < 2) {
    return false;
  }
  auto out_strides = iter.strides(0);
  for (const auto dim : c10::irange(0, 2)) {
      if (out_strides[dim] != 0) {
        return false;
      }
  }
  return true;
}

TORCH_IMPL_FUNC(sum_out)
(const Tensor& self,
 OptionalIntArrayRef opt_dim,
 bool keepdim,
 std::optional<ScalarType> opt_dtype,
 const Tensor& result) {
  auto iter = meta::make_reduction_from_out_ty(self, result, opt_dim, keepdim, result.scalar_type());
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    // Here is a limitation of TensorIterator reductions for permuted input with lower precision on CPU.
    // Consider the case: TensorIterator coalesces such input and output to >= 2 dims tensors,
    // and the output stride is [0, 0, x, x, ...] with x >= 0 (two reduced dimensions and non-reduced dims).
    // Since the reduction loop only operates on two dimensions at a time,
    // the intermediate sums is forced to do accumulation in the second reduced dim with lower precision.
    // See https://github.com/pytorch/pytorch/issues/83149
    if (should_use_acc_buffer(iter)) {
      auto tmp_output = at::empty(result.sizes(), result.options().dtype(kFloat));
      at::sum_outf(self.to(ScalarType::Float), opt_dim, keepdim, /*dtype=*/std::nullopt, tmp_output);
      result.copy_(tmp_output);
    } else{
      sum_stub(iter.device_type(), iter);
    }
  }
}

Tensor sum(const Tensor &self, std::optional<ScalarType> dtype) {
  return at::sum(self, IntArrayRef{}, false, dtype);
}

Tensor sum(const Tensor& self, DimnameList dim, bool keepdim, std::optional<ScalarType> dtype) {
  return at::sum(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& sum_out(const Tensor& self, DimnameList dim,
                bool keepdim, std::optional<ScalarType> opt_dtype, Tensor& result) {
  return at::sum_out(result, self, dimnames_to_positions(self, dim), keepdim, opt_dtype);
}

Tensor& nansum_out(const Tensor& self, at::OptionalIntArrayRef dim,
                       bool keepdim, std::optional<ScalarType> opt_dtype, Tensor& result) {
  if (self.device().is_cpu()) {
    TORCH_CHECK(!c10::isComplexType(self.scalar_type()), "nansum on CPU does not support complex inputs");
  }

  // For integral types, use existing sum as
  // integral types don't have `Nan`.
  if (c10::isIntegralType(self.scalar_type(), true)){
    return at::sum_out(result, self, dim, keepdim, opt_dtype);
  }

  ScalarType dtype = get_dtype_from_result(result, opt_dtype);
  auto iter = make_reduction("nansum", result, self, dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result = result.zero_();
  } else {
    nansum_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor nansum(const Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<ScalarType> opt_dtype) {
  ScalarType dtype = get_dtype_from_self(self, opt_dtype, true);
  Tensor result = create_reduction_result(self, dim, keepdim, dtype);
  return at::native::nansum_out(self, dim, keepdim, dtype, result);
}

} // namespace native
} // namespace at