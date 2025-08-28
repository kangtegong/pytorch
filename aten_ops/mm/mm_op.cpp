#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mkldnn/Matmul.h>
#include <ATen/native/mkldnn/Utils.h>
#include <c10/util/accumulate.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/mm_native.h>
#endif

namespace at {
namespace native {

// Helper function to check if MKL-DNN matrix multiplication should be used
static inline bool apply_mkldnn_matmul_heur(int64_t m, int64_t n, int64_t k) {
#if defined(__aarch64__) && AT_MKLDNN_ACL_ENABLED()
  // On AArch64 apply heuristics to use MKL-DNN GEMM
  static constexpr int64_t min_dim = 128;
  static constexpr int64_t min_size = 128 * 128 * 128;
  return at::globalContext().userEnabledMkldnn() && m > min_dim && k > min_dim && n > min_dim && m * k * n > min_size;
#else
  return false;
#endif
}

static void addmm_impl_cpu_(
    Tensor &result, const Tensor &self, Tensor m1, Tensor m2, const Scalar& beta, const Scalar& alpha) {
  TORCH_INTERNAL_ASSERT(self.dim() == 2 && m1.dim() == 2 && m2.dim() == 2);

  TORCH_CHECK(
    m1.dtype() == m2.dtype(),
    "expected m1 and m2 to have the same dtype, but got: ", m1.dtype(), " != ", m2.dtype()
  )
  // Array access is faster than .size(n) and .stride(n)
  const auto self_sizes = self.sizes();
  auto m1_strides = m1.strides();
  auto m1_sizes = m1.sizes();
  auto m2_strides = m2.strides();
  auto m2_sizes = m2.sizes();

  TORCH_CHECK(
      self_sizes[0] == m1_sizes[0] && self_sizes[1] == m2_sizes[1],
      "input shape is incompatible with matrix multiplication (",
      m1_sizes[0], "x", m1_sizes[1], " @ ", m2_sizes[0], "x", m2_sizes[1], " != ",
      self_sizes[0], "x", self_sizes[1], ")");

  at::native::resize_output(result, self_sizes);
  const auto result_strides = result.strides();
  const auto result_sizes = result.sizes();

  if (result.numel() == 0) {
    return;
  }

  // Some paths in the code below do not handle multiplications of the form [a, 0] x [0, b]
  if (m1_sizes[1] == 0) {
    if (beta.toComplexDouble() == 0.0) {
      result.zero_();
    } else {
      if (!self.is_same(result)) {
        result.copy_(self);
      }
      result.mul_(beta);
    }
    return;
  }

  if (beta.toComplexDouble() != 0.0 && !self.is_same(result)) {
    result.copy_(self);
  }

  bool transpose_c = false;
  Tensor c;

  // Cast result as matrix a
  if (result_strides[0] == 1 &&
      (result_sizes[1] == 1 || result_strides[1] >= std::max(int64_t{1}, result_sizes[0]))) {
    transpose_c = false;
    c = result.resolve_conj();
  } else if (result_strides[1] == 1 &&
             (result_sizes[0] == 1 || result_strides[0] >= std::max(int64_t{1}, result_sizes[1]))) {
    std::swap(m1, m2);
    std::swap(m1_sizes, m2_sizes);
    std::swap(m1_strides, m2_strides);
    transpose_c = true;
    c = result.resolve_conj();
  } else {
    transpose_c = false;
    // make c FORTRAN contiguous
    c = result.resolve_conj().transpose(0, 1).contiguous().transpose_(0, 1);
  }

  const int64_t m = result_sizes[transpose_c ? 1 : 0];
  const int64_t n = result_sizes[transpose_c ? 0 : 1];
  const int64_t k = m1_sizes[transpose_c ? 0 : 1];

  // Cast m1 as matrix a
  bool transpose_a = false;
  Tensor a;
  /* Need lda >= max(1, (transpose_a ? k : m)) */
  if (m1_strides[transpose_c ? 1 : 0] == 1 &&
      m1_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, m)) {
    transpose_a = false;
    a = m1.resolve_conj();
  } else if (m1_strides[transpose_c ? 0 : 1] == 1 &&
             m1_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, k)) {
    transpose_a = true;
    a = m1;
  } else {
    transpose_a = !transpose_c;
    a = m1.clone(at::MemoryFormat::Contiguous);
  }

  // Cast m2 as matrix b
  bool transpose_b = false;
  Tensor b;
  /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
  if (m2_strides[transpose_c ? 1 : 0] == 1 &&
      m2_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, k)) {
    transpose_b = false;
    b = m2.resolve_conj();
  } else if (m2_strides[transpose_c ? 0 : 1] == 1 &&
             m2_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, n)) {
    transpose_b = true;
    b = m2;
  } else {
    transpose_b = !transpose_c;
    b = m2.clone(at::MemoryFormat::Contiguous);
  }

  const int64_t lda = a.strides()[(transpose_a == transpose_c) ? 1 : 0];
  const int64_t ldb = b.strides()[(transpose_b == transpose_c) ? 1 : 0];
  const int64_t ldc = c.strides()[transpose_c ? 0 : 1];

  // Always ensure the conjugation for c is resolved since there's no way to specify c's conjugation in the gemm call
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!c.is_conj());

  bool dispatched = false;
#if defined(__aarch64__) && AT_MKLDNN_ACL_ENABLED()
  // On AArch64 if LHS matrix in BLAS routine is transposed but RHS is not then
  // it is faster to call oneDNN matrix multiplication primitive with RHS*LHS
  // that will call then into Arm® Compute Library (ACL) GEMM kernel and also
  // additionally have support for running kernel with BF16 instructions
  if (transpose_c) {
    bool apply_heur = apply_mkldnn_matmul_heur(b.sizes()[0], b.sizes()[1], a.sizes()[1]);
    if (apply_heur && transpose_a && !transpose_b && result.scalar_type() == at::ScalarType::Float) {
      try {
        mkldnn_matmul(b, a, c, beta.to<float>(), alpha.to<float>());
        // We have dispatched to ACL GEMM for single precision float
        // so do not need to dispatch to BLAS GEMM below
        dispatched = true;
      } catch (const std::exception& e) {
        TORCH_WARN("mkldnn_matmul failed, switching to BLAS gemm:", e.what());
        at::globalContext().setUserEnabledMkldnn(false);
      }
    }
  }
#endif

  if(!dispatched) {
    // Apply BLAS routine
    _AT_DISPATCH_ADDMM_TYPES(result.scalar_type(), "addmm_impl_cpu_", [&]{
          using opmath_t = at::opmath_type<scalar_t>;
          at::native::cpublas::gemm(
              transpose_a ? a.is_conj() ? TransposeType::ConjTranspose : TransposeType::Transpose : TransposeType::NoTranspose,
              transpose_b ? b.is_conj() ? TransposeType::ConjTranspose : TransposeType::Transpose : TransposeType::NoTranspose,
              m, n, k,
              alpha.to<opmath_t>(),
              a.const_data_ptr<scalar_t>(), lda,
              b.const_data_ptr<scalar_t>(), ldb,
              beta.to<opmath_t>(),
              c.mutable_data_ptr<scalar_t>(), ldc);
        });
  }

  if (!c.is_same(result)) {
    result.copy_(c);
  }
}

TORCH_IMPL_FUNC(mm_out_cpu)(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  {
    at::NoNamesGuard guard;
    addmm_impl_cpu_(const_cast<Tensor&>(result), result, self, mat2, 0, 1);
  }
}

} // namespace native
} // namespace at
