#include <boost/program_options.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/timing.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>

using namespace dolfinx;
namespace po = boost::program_options;

auto read_inputs(int argc, char* argv[]) {
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "q", po::value<std::string>()->default_value("cpu"), "Queue type")(
      "s", po::value<int>()->default_value(10),
      "Exponent to compute the number of elements - E = "
      "2^s")("type", po::value<int>()->default_value(0),
             "Assembly type - 0 Full Assembly, 1 "
             "Partial")("p", po::value<int>()->default_value(1), "Polynomial degree")(
      "format", po::value<std::string>()->default_value("table"),
      "Format to print the results.");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(),
            vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    exit(0);
  }

  const int s = vm["s"].as<int>();
  const int type = vm["type"].as<int>();
  const int p = vm["p"].as<int>();
  const std::string queue_type = vm["q"].as<std::string>();
  const std::string format = vm["format"].as<std::string>();

  struct return_values {
    int s, type, p;
    std::string format, queue_type;
  };

  return return_values{s, type, p, format, queue_type};
}

void output_table(std::shared_ptr<fem::FunctionSpace> V, int degree, double t,
                  int number_it, int op_type, std::string format) {

  // Compute output values
  auto mesh = V->mesh();
  MPI_Comm mpi_comm = mesh->comm();
  int rank = dolfinx::MPI::rank(mpi_comm);
  int mpi_size = dolfinx::MPI::size(mpi_comm);
  std::int64_t ncells_global = mesh->topology().index_map(3)->size_global();
  std::int64_t ndofs_global = V->dofmap()->index_map->size_global();
  double metric = ndofs_global / (t / number_it);
  auto [count, wall, user, total] = dolfinx::timing("~setup phase");

  if (format == "table") {
    if (rank == 0) {
      std::cout << "Operator type \t\t\t" << op_type << std::endl;
      std::cout << "Dofs*iteration/second \t\t" << metric << std::endl;
      std::cout << "Polynomial degree \t\t" << degree << std::endl;
      std::cout << "Number of processes \t\t" << mpi_size << std::endl;
      std::cout << "Number of cells \t\t" << ncells_global << std::endl;
      std::cout << "Number of dofs \t\t\t" << ndofs_global << std::endl;
      std::cout << "Number of iterations \t\t" << number_it << std::endl;
      std::cout << "Solve time per iteration \t" << t << std::endl;
      std::cout << std::endl;
    }
    dolfinx::list_timings(mpi_comm, {dolfinx::TimingType::wall});
  } else {
    if (rank == 0) {
      std::cout << op_type << ", ";
      std::cout << metric << ", ";
      std::cout << degree << ", ";
      std::cout << mpi_size << ", ";
      std::cout << ncells_global << ", ";
      std::cout << ndofs_global << ", ";
      std::cout << number_it << ", ";
      std::cout << t << ", ";
      std::cout << wall << " \n";
    }
  }
}
