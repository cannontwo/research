#include <cannon/research/parl/envs/dynamic_car.hpp>
#include <cannon/control/pid.hpp>
#include <cannon/math/interp.hpp>
#include <cannon/geom/graph.hpp>
#include <cannon/geom/trajectory.hpp>
#include <cannon/plot/plotter.hpp>

using namespace cannon::research::parl;
using namespace cannon::control;
using namespace cannon::math;
using namespace cannon::geom;
using namespace cannon::plot;

unsigned int node_id(unsigned int cols, unsigned int i, unsigned int j) {
  return j * cols + i;
}

Graph construct_grid(unsigned int rows, unsigned int cols) {
  Graph g;

  for (unsigned int i = 0; i < cols; ++i) {
    for (unsigned int j = 0; j < rows; ++j) {
      unsigned int current = node_id(cols, i, j);

      if (i > 0)
        g.add_edge(current, node_id(cols, i - 1, j), 1.0);

      if (i < cols - 1)
        g.add_edge(current, node_id(cols, i + 1, j), 1.0);

      if (j > 0)
        g.add_edge(current, node_id(cols, i, j - 1), 1.0);

      if (j < rows - 1)
        g.add_edge(current, node_id(cols, i, j + 1), 1.0);
    }
  }

  return g;
}

Trajectory plan_astar_traj() {
  unsigned int rows = 10;
  unsigned int cols = 10;
  Graph g = construct_grid(rows, cols);

  std::vector<std::pair<double, double>> locs(rows * cols);

  for (unsigned int i = 0; i < cols; ++i) {
    for (unsigned int j = 0; j < rows; ++j) {
      locs[node_id(cols, i, j)] = std::make_pair(static_cast<double>(i) * 0.1, static_cast<double>(j) * 0.1);
    }
  }

  unsigned int goal_idx = cols*rows - 1;
  
  auto path = g.astar(0, goal_idx, [&](unsigned int v) {
    double dx = locs[goal_idx].first - locs[v].first;
    double dy = locs[goal_idx].second - locs[v].second;

    return std::sqrt(dx * dx + dy * dy);
  });

  assert(path[0] == 0);
  assert(path[path.size()-1] == goal_idx);

  Trajectory traj;
  traj.push_back(VectorXd::Ones(2), 0.0);
  traj.push_back(VectorXd::Ones(2), 10.0);
  //for (unsigned int i = 0; i < path.size(); ++i) {
  //  VectorXd spt(2);
  //  spt << locs[path[i]].first,
  //         locs[path[i]].second;
  //  traj.push_back(spt, i);
  //}

  return traj;
}

std::pair<std::vector<Vector2d>, std::vector<Vector2d>>
plot_pid_traj(const MultiSpline &plan, const Ref<const MatrixXd> &kp,
              const Ref<const MatrixXd> &ki, const Ref<const MatrixXd> &kd,
              double length) {
  auto env = std::make_shared<DynamicCarEnvironment>(VectorXd::Zero(5), VectorXd::Ones(5));
  PidController controller(2, 2, env->get_time_step());

  controller.proportional_gain() = kp;
  controller.integral_gain() = ki;
  controller.derivative_gain() = kd;

  double time = 0.0;
  VectorXd state = env->reset(VectorXd::Zero(5));
  std::vector<Vector2d> executed, controls;
  for (unsigned int i = 0; i < 100 * length; ++i) {
    executed.push_back(state.head(2));

    controller.ref() = plan(time);
    VectorXd pid_action = controller.get_control(state.head(2));
    controls.push_back(env->get_constrained_control(pid_action));

    double reward;
    bool done;
    std::tie(state, reward, done) = env->step(pid_action);

    time += env->get_time_step();
  }

  executed.push_back(state.head(2));
  return std::make_pair(executed, controls);
}

int main() {
  auto traj = plan_astar_traj();
  auto plan = traj.interp();

  MatrixXf Kp(2, 2), Ki(2, 2), Kd(2, 2);

  Kp << 0.0, 0.0,
        0.0, 0.0;

  Ki << 0.0, 0.0,
        0.0, 0.0;

  Kd << 0.0, 0.0,
        0.0, 0.0;

  Plotter plotter;
  plotter.render([&]() {
    bool changed = false;
    static int current_mode = 0;
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("PID")) {
        changed = changed || ImGui::InputFloat("Kp(0, 0)", &Kp(0, 0), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Kp(0, 1)", &Kp(0, 1), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Kp(1, 0)", &Kp(1, 0), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Kp(1, 1)", &Kp(1, 1), -100.0, 100.0);

        changed = changed || ImGui::InputFloat("Ki(0, 0)", &Ki(0, 0), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Ki(0, 1)", &Ki(0, 1), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Ki(1, 0)", &Ki(1, 0), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Ki(1, 1)", &Ki(1, 1), -100.0, 100.0);

        changed = changed || ImGui::InputFloat("Kd(0, 0)", &Kd(0, 0), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Kd(0, 1)", &Kd(0, 1), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Kd(1, 0)", &Kd(1, 0), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Kd(1, 1)", &Kd(1, 1), -100.0, 100.0);

        static const char* modes[] = {"Full", "X", "Y", "Controls"};
        changed = changed || ImGui::Combo("Display Mode", &current_mode, modes, IM_ARRAYSIZE(modes));
        
        ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }

    if (changed) {
      plotter.clear();
      auto [pts, controls] = plot_pid_traj(plan, Kp.cast<double>(), Ki.cast<double>(),
                                    Kd.cast<double>(), traj.length());

      
      double time = 0.0;
      std::vector<Vector2d> times, pts_oned;
      switch (current_mode) {
        case 0:
          plotter.plot(traj, 200, 0.0, traj.length());
          plotter.plot(plan, 200, 0.0, traj.length());
          plotter.plot(pts);
          break;
        case 1:
          plotter.plot([&](double t) { return plan(t)[0]; }, 200, 0.0,
                       traj.length());
          for (auto& p : pts) {
            Vector2d tmp;
            tmp << time, p[0];
            pts_oned.push_back(tmp);
            time += 0.01;
          }
          plotter.plot(pts_oned);
          break;
        case 2:
          plotter.plot([&](double t) { return plan(t)[1]; }, 200, 0.0,
                       traj.length());
          for (auto& p : pts) {
            Vector2d tmp;
            tmp << time, p[1];
            pts_oned.push_back(tmp);
            time += 0.01;
          }
          plotter.plot(pts_oned);
          break;
        case 3:
          // TODO Plot controls
          for (auto& p : controls) {
            Vector2d tmp;
            tmp << time, p[0];
            pts_oned.push_back(tmp);
            time += 0.01;
          }
          plotter.plot(pts_oned);
          pts_oned.clear();
          time = 0.0;
          for (auto& p : controls) {
            Vector2d tmp;
            tmp << time, p[1];
            pts_oned.push_back(tmp);
            time += 0.01;
          }
          plotter.plot(pts_oned);
          break;
      }
    }


  });
}
