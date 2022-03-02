#pragma once

#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>

namespace cuda {
template <class T>
class array {
public:
  explicit array(std::size_t size) : _size(size) {
    cudaError_t e = cudaMalloc((void**)&_data, _size * sizeof(T));
    if (e != cudaSuccess) {
      throw std::runtime_error("Failed to allocate device memory.");
    }
  }

  ~array() { cudaFree(_data); }

  size_t size() const { return _size; }

  const T* data() const { return _data; }
  T* data() { return _data; }

  // set
  template <class Container>
  void set(Container& source) {
    assert(source.size() == _size);
    cudaError_t e
        = cudaMemcpy(_data, source.data(), _size * sizeof(T), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
      throw std::runtime_error("Failed to copy data from host to device");
    }
  }

  std::vector<T> copy_to_host() {
    std::vector<T> host_vec(_size);
    T* h_data = host_vec.data();
    cudaError_t e = cudaMemcpy(h_data, _data, _size * sizeof(T), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
      throw std::runtime_error("failed to copy to host memory");
    }
    return host_vec;
  }
  // private functions
private:
  T* _data;
  std::size_t _size;
};

} // namespace cuda