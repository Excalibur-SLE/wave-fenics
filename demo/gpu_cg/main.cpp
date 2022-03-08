#include "bp1.h"

#include "CUDA/allocator.hpp"
#include "CUDA/cg.hpp"
#include "mesh.hpp"
#include "operators.hpp"

#include "cublas_v2.h"
#include "utils.hpp"
#include <dolfinx.h>

using namespace dolfinx;
namespace po = boost::program_options;

template <typename T, typename Alloc = std::allocator<T>>
using apply_fn = std::function<void(const la::Vector<T, Alloc>&, la::Vector<T, Alloc>&)>;

int main(int argc, char* argv[]) {
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);

  auto [s, type, p, format, queue_type] = read_inputs(argc, argv);

  MPI_Comm mpi_comm{MPI_COMM_WORLD};
  std::vector function_space
      = {functionspace_form_bp1_a1, functionspace_form_bp1_a2, functionspace_form_bp1_a3,
         functionspace_form_bp1_a4, functionspace_form_bp1_a5};

  std::vector form_a = {form_bp1_a1, form_bp1_a2, form_bp1_a3, form_bp1_a4, form_bp1_a5};
  std::vector form_L = {form_bp1_L1, form_bp1_L2, form_bp1_L3, form_bp1_L4, form_bp1_L5};
  std::map<std::string, std::shared_ptr<const fem::Constant<double>>> constants;
  {
    int bs = 1;

    // Create Hex mesh -  E = 2^s
    auto mesh = std::make_shared<mesh::Mesh>(benchmark::create_hex_mesh(mpi_comm, s));

    // Create Function Space
    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(*function_space.at(p - 1), "v_0", mesh));

    // Create RHS arbitrary coefficient f
    auto f = std::make_shared<fem::Function<double>>(V);
    f->interpolate([](auto& x) { return xt::row(x, 0) + 4; });

    // Define variational forms (a = L)
    auto a = std::make_shared<fem::Form<double>>(
        fem::create_form<double>(*form_a.at(p - 1), {V, V}, {}, constants, {}));
    auto L = std::make_shared<fem::Form<double>>(
        fem::create_form<double>(*form_L.at(p - 1), {V}, {{"w0", f}}, {}, {}));

    // Assemble RHS vector
    CUDA::allocator<double> allocator{};
    la::Vector<double, decltype(allocator)> bvec(V->dofmap()->index_map, bs, allocator);
    la::Vector<double, decltype(allocator)> uvec(V->dofmap()->index_map, bs, allocator);
    fem::assemble_vector(bvec.mutable_array(), *L);
    bvec.scatter_rev(common::IndexMap::Mode::add);

    
    int mpi_size = dolfinx::MPI::size(mpi_comm);
    int rank = dolfinx::MPI::rank(mpi_comm);

    int numGpus = 0;
    cudaGetDeviceCount(&numGpus);

    if (numGpus != mpi_size && mpi_size != 1) {
      throw std::runtime_error("The number of MPI processes should be less or equal the "
                               "number of available devices.");
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaSetDevice(rank);

    std::function<void(const la::Vector<double, CUDA::allocator<double>>&,
		       la::Vector<double, CUDA::allocator<double>>&)> matvec = [&](auto a, auto b){
			 copy<la::Vector<double, CUDA::allocator<double>>>(handle, a, b);};

    
    int number_it = device::cg(handle, uvec, bvec, matvec, 50, 1e-4);
    std::cout << "its = " << number_it << "\n";
    
    cublasDestroy(handle);
  }

  common::subsystem::finalize_mpi();
  return 0;
}