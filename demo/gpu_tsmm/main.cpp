#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>
#include <stdexcept>

void assert_cublas(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(" Unable to allocate memoy - cublas error");
}
int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int ndofs = 125;
  int ncells = 100000;

  cublasHandle_t handle;
  cublasCreate(&handle);

  double* xe;
  unsigned int size_xe = ncells * ndofs;
  unsigned int mem_size_xe = sizeof(double) * size_xe;
  assert_cublas(cudaMalloc((void**)&xe, mem_size_xe));
  cudaMemset(&xe, 0.5, mem_size_xe);

  double* xq;
  unsigned int size_xq = ncells * ndofs;
  unsigned int mem_size_xq = sizeof(double) * size_xq;
  assert_cublas(cudaMalloc((void**)&xq, mem_size_xq));
  cudaMemset(&xq, 0, ncells * ndofs * sizeof(double));

  double* ue;
  unsigned int size_ue = ncells * ndofs;
  unsigned int mem_size_ue = sizeof(double) * size_ue;
  assert_cublas(cudaMalloc((void**)&ue, mem_size_ue));
  cudaMemset(&ue, 0, ncells * ndofs * sizeof(double));

  double* phi;
  unsigned int size_phi = ncells * ndofs;
  unsigned int mem_size_phi = sizeof(double) * size_phi;
  assert_cublas(cudaMalloc((void**)&phi, mem_size_phi));

  double alpha = 1;
  double beta = 0;

  double t = MPI_Wtime();
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncells, ndofs, ndofs, &alpha, xe, ncells,
              phi, ndofs, &beta, xq, ncells);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ncells, ndofs, ndofs, &alpha, xq, ncells,
              phi, ndofs, &beta, ue, ncells);
  cudaDeviceSynchronize();
  t = MPI_Wtime() - t;

  std::cout << "Number of cells: " << ncells;
  std::cout << "\nNumber of dofs: " << ndofs;
  std::cout << "\n#GFLOPs: " << (4 * ncells * ndofs * ndofs) / t / 1e9;

  cudaFree(xe);
  cudaFree(xq);
  cudaFree(ue);
  cudaFree(phi);

  MPI_Finalize();

  return 0;
}
