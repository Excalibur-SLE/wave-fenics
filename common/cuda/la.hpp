#include "CUDA/utils.hpp"
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <type_traits>

template <class T>
struct dependent_false : std::false_type {};

namespace {
template <typename C>
void assert_cuda(C e) {
  if (e != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("CUDA ERROR: " + std::to_string(e));
}
} // namespace

namespace linalg {
/*******************************************/
/* Vector operations using CUDA and CUBLAS */
/*******************************************/
//-------------------------------------------------------------------------//
/// Performs the ceiled division operation.
template <class T>
inline T divceil(T numerator, T denominator) {
  return (numerator / denominator + (numerator % denominator > 0));
}
//-------------------------------------------------------------------------//
template <typename Vector>
void copy(cublasHandle_t handle, const Vector& x, Vector& y) {

  using T = typename Vector::value_type;
  assert(x.array().size() == y.array().size());

  const T* _x = x.array().data();
  T* _y = y.mutable_array().data();
  std::size_t n = x.map()->size_local();

  cublasStatus_t status;

  if constexpr (std::is_same<T, double>())
    status = cublasDcopy(handle, n, _x, 1, _y, 1);
  else if constexpr (std::is_same<T, float>())
    status = cublasScopy(handle, n, _x, 1, _y, 1);
  else
    static_assert(dependent_false<T>::value);

  assert_cuda(status);
}
//-------------------------------------------------------------------------//
/// Provides hints to the runtime library that data should be made available
/// on a device earlier than Unified Shared Memory would normally require it
/// to be available.
template <typename Vector>
void prefetch(int device, Vector& x) {
  using T = typename Vector::value_type;
  const T* _x = x.mutable_array().data();
  const std::size_t count = x.mutable_array().size() * sizeof(T);
  cudaMemPrefetchAsync(_x, count, device);
  cudaDeviceSynchronize();
}
//-------------------------------------------------------------------------//
template <typename Scalar, typename Vector>
void axpy(cublasHandle_t handle, Scalar alpha, const Vector& x, Vector& y) {
  using T = typename Vector::value_type;

  assert(x.array().size() == y.array().size());
  const T* _x = x.array().data();
  T* _y = y.mutable_array().data();
  std::size_t n = x.map()->size_local();

  cublasStatus_t status;
  T _alpha = static_cast<T>(alpha);

  if constexpr (std::is_same<T, double>())
    status = cublasDaxpy(handle, n, &_alpha, _x, 1, _y, 1);
  else if constexpr (std::is_same<T, float>())
    status = cublasSaxpy(handle, n, &alpha, _x, 1, _y, 1);
  else
    static_assert(dependent_false<T>::value);

  assert_cuda(status);
}
//-------------------------------------------------------------------------//
template <typename Vector>
auto inner_product(cublasHandle_t handle, const Vector& x, const Vector& y) {
  using T = typename Vector::value_type;
  assert(x.array().size() == y.array().size());
  const T* _x = x.array().data();
  const T* _y = y.array().data();
  std::size_t n = x.map()->size_local();
  cublasStatus_t status;
  T result = 0;
  if constexpr (std::is_same<T, double>())
    cublasDdot(handle, n, _x, 1, _y, 1, &result);
  else if constexpr (std::is_same<T, float>())
    cublasSdot(handle, n, _x, 1, _y, 1, &result);
  else
    static_assert(dependent_false<T>::value);
  assert_cuda(status);
  return result;
}
//-------------------------------------------------------------------------//
template <typename Vector>
auto squared_norm(cublasHandle_t handle, const Vector& x) {
  using T = typename Vector::value_type;
  const T* _x = x.array().data();
  std::size_t n = x.map()->size_local();
  cublasStatus_t status;
  T result = 0;
  if constexpr (std::is_same<T, double>())
    cublasDnrm2(handle, n, _x, 1, &result);
  else if constexpr (std::is_same<T, float>())
    cublasSnrm2(handle, n, _x, 1, &result);
  else
    static_assert(dependent_false<T>::value);
  assert_cuda(status);
  return result;
}
//-------------------------------------------------------------------------//
template <typename Scalar, typename Vector>
void scale(cublasHandle_t handle, Scalar alpha, Vector& x) {
  using T = typename Vector::value_type;
  const T* _x = x.mutable_array().data();
  std::size_t n = x.map()->size_local();

  T _alpha = static_cast<T>(alpha);
  cublasStatus_t status;

  if constexpr (std::is_same<T, double>())
    cublasDscal(handle, n, &_alpha, _x, 1);
  else if constexpr (std::is_same<T, float>())
    cublasSscal(handle, n, &_alpha, _x, 1);
  else
    static_assert(dependent_false<T>::value);
  assert_cuda(status);
}
} // namespace linalg