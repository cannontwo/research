#include <cannon/research/parl/envs/dynamic_car.hpp>
#include <cannon/control/pid.hpp>
#include <cannon/math/interp.hpp>
#include <cannon/geom/graph.hpp>
#include <cannon/geom/trajectory.hpp>
#include <cannon/plot/plotter.hpp>
#include <cannon/research/parl/parl.hpp>

using namespace cannon::research::parl;
using namespace cannon::control;
using namespace cannon::math;
using namespace cannon::geom;
using namespace cannon::plot;

int main() {
  ControlledTrajectory traj;
  traj.load("logs/sst_dynamic_car_plan.h5");

  // TODO Make time an IMGUI control
  Plotter plotter;
  plotter.render([&]() {
    static float time = 0.0;

    bool changed = false;
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("Reward Plotting")) {
        changed = changed || ImGui::SliderFloat("Traj time", &time, 0.0, traj.length());

        ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }

    if (changed) {
      plotter.clear();
      plotter.plot([&](const Vector2d &p) {
        auto plan_state = traj(time).first;
        double tracking_reward = -((p - plan_state.head(2)).norm());
        return tracking_reward;
      }, 15, -2.0, 2.0, -2.0, 2.0);
      plotter.plot([&](double t){
          return traj(t).first;
          }, 200, 0.0, traj.length());
    }
  });
}
