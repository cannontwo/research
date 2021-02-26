#define TINYCOLORMAP_WITH_EIGEN
#include <tinycolormap.hpp>

#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl_stability/transition_map.hpp>
#include <cannon/research/parl_stability/lyapunov_finding.hpp>

using namespace cannon::research::parl;

int main(int argc, char** argv) {
  if (argc != 2) {
    log_info("Pass a PWA HDF5 file to load");
  }
   
  PWAFunc pwa = load_pwa(std::string(argv[1]));

  // TODO
}
