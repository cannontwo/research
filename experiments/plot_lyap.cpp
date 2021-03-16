#define TINYCOLORMAP_WITH_EIGEN
#include <tinycolormap.hpp>

#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl_stability/transition_map.hpp>
#include <cannon/research/parl_stability/lyapunov_finding.hpp>

using namespace cannon::research::parl;

int main(int argc, char** argv) {
  if (argc != 2) {
    log_info("Pass a Lyapunov HDF5 file to load");
  }

  std::vector<LyapunovComponent> lyap;
  double theta;
   
  std::tie(lyap, theta) = load_lyap(std::string(argv[1]));

  Plotter p;
  double eps = 1e-6;

  const unsigned int GRID_SIZE = 200;
  for (unsigned int i = 0; i < GRID_SIZE; i++) {
    for (unsigned int j = 0; j < GRID_SIZE; j++) {
      Vector2d test_state = Vector2d::Zero();

      // TODO Automatically derive bounds
      test_state[0] = -4.0 + (8.0 / GRID_SIZE) * i;
      test_state[1] = -4.0 + (8.0 / GRID_SIZE) * j;

      for (unsigned int k = 0; k < lyap.size(); k++) {
        if (is_inside(test_state, lyap[k].poly_)) {
          double lyap_val = evaluate_lyap(lyap, test_state);
          if (lyap_val < theta - eps) {
            float t = lyap_val / theta;

            Vector3d rgb = tinycolormap::GetColor(t, tinycolormap::ColormapType::Viridis).ConvertToEigen();
            Vector4f point_value_color = Vector4f::Ones();
            point_value_color[0] = rgb[0];
            point_value_color[1] = rgb[1];
            point_value_color[2] = rgb[2];

            RowVector2f plot_point;
            plot_point[0] = test_state[0];
            plot_point[1] = test_state[1];
            p.plot_points(plot_point, point_value_color);
          }
        }
      }
    }
  }

  p.render();
}
