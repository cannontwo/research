#define TINYCOLORMAP_WITH_EIGEN
#include <tinycolormap.hpp>

#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl_stability/transition_map.hpp>
#include <cannon/research/parl_stability/lyapunov_finding.hpp>

using namespace cannon::research::parl;

int main(int argc, char** argv) {
  if (argc != 3) {
    log_info("Pass a PWA HDF5 file to load and a radius to contrict it to");
  }
   
  PWAFunc pwa = load_pwa(std::string(argv[1]));
  double radius = std::stod(argv[2]);

  // This can be adjusted to get a fuller picture at the expense of more
  // computation time.
  pwa = restrict_pwa(pwa, radius);

  auto transition_map_pair = compute_transition_map(pwa);
  
  std::vector<LyapunovComponent> lyap;
  PWAFunc refined_pwa;
  double alpha_1, alpha_3, theta;
  std::tie(lyap, refined_pwa, alpha_1, alpha_3, theta) = find_lyapunov(pwa,
      transition_map_pair.first, transition_map_pair.second, 100);

  check_lyap(lyap, refined_pwa, alpha_1, alpha_3, theta);

  // TODO In addition to checking Lyapunov properties, check that the found
  // Lyapunov function's domain is in fact a PI set.
}
