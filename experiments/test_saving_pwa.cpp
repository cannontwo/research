#include <cannon/research/parl/runner.hpp>
#include <cannon/research/parl/envs/inverted_pendulum.hpp>
#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl_stability/transition_map.hpp>

using namespace cannon::research::parl;

int main() {
  Hyperparams params;

  auto env = std::make_shared<InvertedPendulumEnvironment>();

  Runner r(env,
      "/home/cannon/Documents/cannon/cannon/research/experiments/parl_configs/r10c10_short.yaml",
      false);
  r.run();

  auto controlled_system = r.get_agent()->get_controlled_system();
  log_info(controlled_system[0].A_);

  // Compute polytopal representation of Voronoi diagram of PARL refs
  auto diagram = compute_voronoi_diagram(r.get_agent()->get_refs());

  // Create transition map using CGAL polygon affine mapping
  auto current_pwa = compute_parl_pwa_func(r.get_agent(), diagram);

  log_info("Plotting before saving");
  auto transition_map_pair = compute_transition_map(current_pwa);
  plot_transition_map(transition_map_pair, current_pwa);

  save_pwa(current_pwa, "parl_pwa.h5");
  auto loaded_pwa = load_pwa("parl_pwa.h5");

  log_info("Plotting after loading");
  auto loaded_transition_map_pair = compute_transition_map(loaded_pwa);
  plot_transition_map(loaded_transition_map_pair, loaded_pwa);
}
