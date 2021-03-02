#define TINYCOLORMAP_WITH_EIGEN
#include <tinycolormap.hpp>

#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl_stability/transition_map.hpp>
#include <cannon/research/parl_stability/lyapunov_finding.hpp>

#include <cannon/plot/plotter.hpp>

using namespace cannon::plot;
using namespace cannon::research::parl;

struct TrajPlotter{
  TrajPlotter(const PWAFunc& p) : pwa(p) {}

  Plotter plotter;
  PWAFunc pwa;
  double mouse_xpos = 0.0;
  double mouse_ypos = 0.0;
};

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
  TrajPlotter *p = (TrajPlotter*)glfwGetWindowUserPointer(window);

  int width, height;
  glfwGetWindowSize(window, &width, &height);

  float normalized_x = (2 * p->mouse_xpos / (float)width) - 1.0;
  float normalized_y = (2 * (height - p->mouse_ypos) / (float)height) - 1.0;

  Vector4f screen_coords = Vector4f::Ones();
  screen_coords[0] = normalized_x;
  screen_coords[1] = normalized_y;

  Vector4f plot_coords = p->plotter.axes_.make_scaling_matrix().inverse() * screen_coords;

  double max_val = std::pow(p->plotter.axes_.x_max_, 2.0);
  max_val += std::pow(p->plotter.axes_.y_max_, 2.0);
  max_val = std::sqrt(max_val);

  double t = plot_coords.head(2).norm() / max_val;

  Vector3d rgb = tinycolormap::GetColor(t, tinycolormap::ColormapType::Viridis).ConvertToEigen();
  Vector4f point_value_color = Vector4f::Ones();
  point_value_color[0] = rgb[0];
  point_value_color[1] = rgb[1];
  point_value_color[2] = rgb[2];

  Vector2d state;
  state[0] = plot_coords[0];
  state[1] = plot_coords[1];
  for (unsigned int i = 0; i < 200; i++) {
    try {
    Vector2d next_state = evaluate_pwa(p->pwa, state);
    p->plotter.plot_points(state.cast<float>().transpose(), point_value_color);
    state = next_state;
    } catch (...) {
      break;
    }
  }
}

void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
  TrajPlotter *p = (TrajPlotter*)glfwGetWindowUserPointer(window);
  p->mouse_xpos = xpos;
  p->mouse_ypos = ypos;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    log_info("Pass a PWA HDF5 file to load");
  }

  auto pwa = load_pwa(std::string(argv[1]));

  TrajPlotter p(pwa);

  glfwSetWindowUserPointer(p.plotter.w_.get_gl_window(), &p);
  glfwSetMouseButtonCallback(p.plotter.w_.get_gl_window(), mouse_button_callback);
  glfwSetCursorPosCallback(p.plotter.w_.get_gl_window(), cursor_pos_callback);

  p.plotter.render();
}
