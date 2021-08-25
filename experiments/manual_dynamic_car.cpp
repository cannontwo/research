#include <cannon/research/parl/envs/dynamic_car.hpp>

#include <GLFW/glfw3.h>

using namespace cannon::research::parl;
using namespace cannon::graphics;

static VectorXd env_action = VectorXd::Zero(2);

void env_key_callback(GLFWwindow* w, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_W && action == GLFW_PRESS)
    env_action[0] = 2.0;

  if (key == GLFW_KEY_W && action == GLFW_RELEASE)
    env_action[0] = 0.0;

  if (key == GLFW_KEY_S && action == GLFW_PRESS)
    env_action[0] = -2.0;

  if (key == GLFW_KEY_S && action == GLFW_RELEASE)
    env_action[0] = 0.0;

  if (key == GLFW_KEY_A && action == GLFW_PRESS)
    env_action[1] = -M_PI;

  if (key == GLFW_KEY_A && action == GLFW_RELEASE)
    env_action[1] = 0.0;

  if (key == GLFW_KEY_D && action == GLFW_PRESS)
    env_action[1] = M_PI;

  if (key == GLFW_KEY_D && action == GLFW_RELEASE)
    env_action[1] = 0.0;
}

std::pair<std::shared_ptr<DeferredRenderer>, geometry::ModelPtr> start_rendering() {
  std::shared_ptr<DeferredRenderer> renderer(new DeferredRenderer);

  auto car_model = renderer->viewer->spawn_model("assets/car/lowpoly_car_revised.obj");
  car_model->set_pos({0.0, 0.0, 0.0});
  car_model->set_scale(0.01);

  auto goal_sphere_model = renderer->viewer->spawn_model("assets/sphere/sphere.obj");
  float goal_x = 1.0;
  float goal_y = 1.0;
  goal_sphere_model->set_pos({goal_x, 0.0, goal_y});
  goal_sphere_model->set_scale(0.05);

  // Wall geometry
  auto north_wall = renderer->viewer->spawn_plane();
  north_wall->set_pos({0.0, 0.0, 2.0});
  AngleAxisf north_rot(M_PI, Vector3f::UnitY());
  north_wall->set_rot(north_rot);
  north_wall->set_scale(5.0);

  auto south_wall = renderer->viewer->spawn_plane();
  south_wall->set_pos({0.0, 0.0, -2.0});
  AngleAxisf south_rot(0, Vector3f::UnitY());
  south_wall->set_rot(south_rot);
  south_wall->set_scale(5.0);

  auto east_wall = renderer->viewer->spawn_plane();
  east_wall->set_pos({2.0, 0.0, 0.0});
  AngleAxisf east_rot(-M_PI / 2, Vector3f::UnitY());
  east_wall->set_rot(east_rot);
  east_wall->set_scale(5.0);

  auto west_wall = renderer->viewer->spawn_plane();
  west_wall->set_pos({-2.0, 0.0, 0.0});
  AngleAxisf west_rot(M_PI / 2, Vector3f::UnitY());
  west_wall->set_rot(west_rot);
  west_wall->set_scale(5.0);

  auto north_back_wall = renderer->viewer->spawn_plane();
  north_back_wall->set_pos({0.0, 0.0, 2.0});
  AngleAxisf north_back_rot(0.0, Vector3f::UnitY());
  north_back_wall->set_rot(north_back_rot);
  north_back_wall->set_scale(5.0);

  auto south_back_wall = renderer->viewer->spawn_plane();
  south_back_wall->set_pos({0.0, 0.0, -2.0});
  AngleAxisf south_back_rot(M_PI, Vector3f::UnitY());
  south_back_wall->set_rot(south_back_rot);
  south_back_wall->set_scale(5.0);

  auto east_back_wall = renderer->viewer->spawn_plane();
  east_back_wall->set_pos({2.0, 0.0, 0.0});
  AngleAxisf east_back_rot(M_PI / 2, Vector3f::UnitY());
  east_back_wall->set_rot(east_back_rot);
  east_back_wall->set_scale(5.0);

  auto west_back_wall = renderer->viewer->spawn_plane();
  west_back_wall->set_pos({-2.0, 0.0, 0.0});
  AngleAxisf west_back_rot(-M_PI / 2, Vector3f::UnitY());
  west_back_wall->set_rot(west_back_rot);
  west_back_wall->set_scale(5.0);

  // Set camera position
  renderer->viewer->c.set_pos({-3.0, 7.0, -3.0});
  Vector3f dir = renderer->viewer->c.get_pos() - Vector3f::Zero();
  dir.normalize();
  renderer->viewer->c.set_direction(dir);

  return std::make_pair(renderer, car_model);
}

int main() {
  std::shared_ptr<DeferredRenderer> renderer;
  geometry::ModelPtr car_model;
  std::tie(renderer, car_model) = start_rendering();

  auto env = std::make_shared<DynamicCarEnvironment>(VectorXd::Zero(5), VectorXd::Ones(5));

  glfwSetKeyCallback(renderer->viewer->w->get_gl_window(), env_key_callback);
  VectorXd start(5);
  start << 0.0,
           0.0,
           0.0,
           0.0,
           0.0;

  renderer->render_loop([&] {
      static VectorXd state = env->reset(start);

      double x = state[0];
      double y = state[1];
      double th = state[2];
      Vector3f pos;
      pos[0] = x;
      pos[1] = 0.0;
      pos[2] = y;
      car_model->set_pos(pos);

      AngleAxisf rot(-th + M_PI/2.0, Vector3f::UnitY());
      car_model->set_rot(rot);

      std::this_thread::sleep_for(
          std::chrono::duration<double, std::ratio<1>>(env->get_time_step()));


      double reward;
      bool done;
      std::tie(state, reward, done) = env->step(env_action);

      log_info("Reward is", reward);
      });

}
