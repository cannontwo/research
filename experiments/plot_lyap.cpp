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

  Plotter p(true);
  double eps = 1e-6;

  unsigned int processed = 0;
  for (auto& component : lyap) {
    MatrixX4f colors = MatrixX4f::Zero(component.poly_.size(), 4);

    // TODO Only plot polygons for which all vertices are < theta
    
    bool all_above = true;
    
    unsigned int i = 0;
    for (auto it = component.poly_.vertices_begin(); it < component.poly_.vertices_end(); it++) {
      Vector2d test_state = Vector2d::Zero();
      test_state[0] = CGAL::to_double(it->x());
      test_state[1] = CGAL::to_double(it->y());

      double lyap_val = evaluate_lyap(lyap, test_state);
      if (lyap_val < theta - eps)
        all_above = false;

      float t = lyap_val / theta;
      //Vector3d rgb = tinycolormap::GetColor(t, tinycolormap::ColormapType::Viridis).ConvertToEigen();
      Vector4f point_value_color = Vector4f::Ones();
      //point_value_color[0] = rgb[0];
      //point_value_color[1] = rgb[1];
      //point_value_color[2] = rgb[2];
      
      // TODO This is a pretty big hack to get correct Viridis interpolation on
      // these polygons.
      point_value_color[0] = t;

      colors.row(i) = point_value_color.transpose();

      i++;
    }

    if (!all_above) 
      p.plot_polygon(component.poly_, colors);

    processed++;

    log_info("Processed", processed, "/", lyap.size(), "Lyapunov components");
  }
  

  //const unsigned int GRID_SIZE = 200;
  //for (unsigned int i = 0; i < GRID_SIZE; i++) {
  //  for (unsigned int j = 0; j < GRID_SIZE; j++) {
  //    Vector2d test_state = Vector2d::Zero();

  //    // TODO Automatically derive bounds
  //    test_state[0] = -1.5 + (3.0 / GRID_SIZE) * i;
  //    test_state[1] = -5.5 + (11.0 / GRID_SIZE) * j;

  //    for (unsigned int k = 0; k < lyap.size(); k++) {
  //      if (is_inside(test_state, lyap[k].poly_)) {
  //        double lyap_val = evaluate_lyap(lyap, test_state);
  //        if (lyap_val < theta - eps) {
  //          float t = lyap_val / theta;

  //          Vector3d rgb = tinycolormap::GetColor(t, tinycolormap::ColormapType::Viridis).ConvertToEigen();
  //          Vector4f point_value_color = Vector4f::Ones();
  //          point_value_color[0] = rgb[0];
  //          point_value_color[1] = rgb[1];
  //          point_value_color[2] = rgb[2];

  //          RowVector2f plot_point;
  //          plot_point[0] = test_state[0];
  //          plot_point[1] = test_state[1];
  //          p.plot_points(plot_point, point_value_color);
  //        }
  //      }
  //    }
  //  }
  //}

  p.render();
}
