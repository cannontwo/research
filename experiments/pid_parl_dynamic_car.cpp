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
  unsigned int rows = 2;
  unsigned int cols = 2;
  Graph g = construct_grid(rows, cols);

  double horiz_diff = 1.0 / (cols-1);
  double vert_diff = 1.0 / (rows-1);

  std::vector<std::pair<double, double>> locs(rows * cols);

  for (unsigned int i = 0; i < cols; ++i) {
    for (unsigned int j = 0; j < rows; ++j) {
      locs[node_id(cols, i, j)] = std::make_pair(static_cast<double>(i) * horiz_diff, static_cast<double>(j) * vert_diff);
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
  //traj.push_back(VectorXd::Ones(2), 0.0);
  //traj.push_back(VectorXd::Ones(2), 10.0);
  for (unsigned int i = 0; i < path.size(); ++i) {
    VectorXd spt(2);
    spt << locs[path[i]].first,
           locs[path[i]].second;
    traj.push_back(spt, i*10);
  }

  return traj;
}


double compute_traj_error(const MultiSpline &plan, const std::vector<Vector2d>& executed) {
  double path_error = 0.0;
  for (unsigned int i = 0; i < executed.size(); ++i) {
    auto plan_pt = plan(i * 0.01);
    path_error += (plan_pt - executed[i]).norm();
  }

  return path_error;
}

std::vector<double> get_ref_point_times(double length, unsigned int num_refs) {
  // Placing reference points halfway between each waypoint on the path
  std::vector<double> points;

  double duration_delta = length / static_cast<double>(num_refs);

  double accumulated_dur = 0.0;
  while (accumulated_dur < length) {
    points.push_back(accumulated_dur);
    accumulated_dur += duration_delta;
  }

  return points;
}

MatrixXd make_error_system_refs(double length, unsigned int num_refs=100) {
  auto times = get_ref_point_times(length, num_refs);

  // Since this is the error system, all dimensions except for time should be
  // zero
  MatrixXd refs = MatrixXd::Zero(6, times.size());
  for (unsigned int i = 0; i < times.size(); i++) {
    refs(5, i) = times[i];
  }

  return refs;
}

std::shared_ptr<ob::StateSpace>
make_error_state_space(std::shared_ptr<Environment> env, double duration) {
  auto time_space = std::make_shared<ob::RealVectorStateSpace>(1);
  ob::RealVectorBounds b(1);
  b.setLow(0.0);
  b.setHigh(duration + 1.0);
  time_space->setBounds(b);
  time_space->setup();

  auto cspace = std::make_shared<ob::CompoundStateSpace>();
  cspace->addSubspace(env->get_state_space(), 1.0);
  cspace->addSubspace(time_space, 1.0);
  cspace->setup();

  return cspace;
}

VectorXd compute_error_state(const VectorXd& ref, const VectorXd& actual, double time) {
  assert(ref.size() == actual.size());

  VectorXd diff = ref - actual;
  VectorXd ret = VectorXd::Zero(ref.size() + 1);
  ret.head(ref.size()) = diff;
  ret[ref.size()] = time;

  return ret;
}

std::vector<Vector2d>
execute_pid_traj(const MultiSpline &plan, PidController &controller,
                 std::shared_ptr<DynamicCarEnvironment> env, double length) {

  controller.reset();

  double time = 0.0;
  VectorXd state = env->reset(VectorXd::Zero(5));
  //env->render();
  std::vector<Vector2d> executed;
  for (unsigned int i = 0; i < 100 * length; ++i) {
    std::cout << "\r" << "On step " << i << std::flush;
    executed.push_back(state.head(2));

    controller.ref() = plan(time);
    VectorXd pid_action = controller.get_control(state.head(2));

    double reward;
    bool done;
    std::tie(state, reward, done) = env->step(pid_action);
    //env->render();
    
    log_info("State is", state);
    
    time += env->get_time_step();
  }

  executed.push_back(state.head(2));
  return executed;
}

std::pair<std::vector<Vector2d>, double>
execute_parl_pid_traj(const MultiSpline &plan,
                      const ControlledTrajectory &traj,
                      std::shared_ptr<Parl> parl,
                      std::shared_ptr<DynamicCarEnvironment> env, double length,
                      bool do_controller_update) {

  double time = 0.0;
  VectorXd state = env->reset(VectorXd::Zero(5));
  //env->render();
  std::vector<Vector2d> executed;
  std::vector<VectorXd> states;
  double total_reward = 0.0;
  for (unsigned int i = 0; i < 100 * length; ++i) {
    std::cout << "\r" << "On step " << i << std::flush;
    executed.push_back(state.head(2));

    VectorXd pid_action = traj(time).second;

    // Compute PARL action
    auto plan_state = traj(time).first;
    auto error_state = compute_error_state(plan_state, state, time);
    auto parl_action = parl->get_action(error_state);

    auto combined_action = parl_action + pid_action;

    double reward;
    bool done;
    std::tie(state, reward, done) = env->step(combined_action);
    //env->render();

    time += env->get_time_step();

    auto traj_state = plan(time);

    double tracking_reward = -((state.head(2) - plan_state.head(2)).norm());

    total_reward += tracking_reward;

    // Train PARL
    auto new_plan_state = traj(time).first;
    auto new_error_state = compute_error_state(new_plan_state, state, time);
    parl->process_datum(error_state, parl_action, tracking_reward, new_error_state);
    states.push_back(error_state);
  }

  if (do_controller_update) {
    for (auto state : states) {
      parl->value_grad_update_controller(state);
    }
  }

  executed.push_back(state.head(2));
  return std::make_pair(executed, total_reward);
}

int main() {
  auto traj = plan_astar_traj();
  auto plan = traj.interp();

  auto env = std::make_shared<DynamicCarEnvironment>(VectorXd::Zero(5),
                                                     VectorXd::Ones(5));

  MatrixXd Kp(2, 2), Ki(2, 2), Kd(2, 2);

  Kp << 10.0, 10.0,
        20.0, 20.0;

  Ki << 1.0, 1.0,
        1.0, 1.0;

  Kd << 0.0, 0.0,
        0.0, 0.0;

  PidController controller(2, 2, env->get_time_step());

  controller.proportional_gain() = Kp;
  controller.integral_gain() = Ki;
  controller.derivative_gain() = Kd;

  Hyperparams params;
  params.load_config("/home/cannon/Documents/cannon/cannon/research/"
                     "experiments/parl_configs/r10c10_dc.yaml");

  auto parl = std::make_shared<Parl>(
      make_error_state_space(env, traj.length()), env->get_action_space(),
      make_error_system_refs(traj.length(), 20), params);

  auto pid_pts = execute_pid_traj(plan, controller, env, traj.length());
  auto path_error = compute_traj_error(plan, pid_pts);

  // Compute error between plan and controlled path
  log_info("Without PARL, point-to-point path execution error was", path_error);
  log_info("Without PARL, average path execution error was", path_error / pid_pts.size());

  DynamicCarEnvironment model_env;
  VectorXd plan_start(5);
  plan_start.head(2) = traj(0.0);
  model_env.reset(plan_start);
  auto controlled_traj = get_pid_controlled_trajectory(
      [&](const Ref<const VectorXd> &state,
          const Ref<const VectorXd> &control) {

        model_env.reset(state);

        auto [new_state, reward, done] = model_env.step(control);

        return new_state;
      },
      traj, controller, 2, 5);

  std::vector<Vector2d> parl_pts;
  int cached_traj_num = -1;
  std::thread plot_thread([&]() {
    Plotter plotter;

    plotter.render([&]() {
      static int traj_num = cached_traj_num;

      if (traj_num != cached_traj_num) {
        plotter.clear();
        plotter.plot(traj, 200, 0.0, traj.length());
        plotter.plot(plan, 200, 0.0, traj.length());
        plotter.plot([&](double t) { return controlled_traj(t).first; }, 200,
                     0.0, controlled_traj.length());

        plotter.plot(pid_pts);
        plotter.plot(parl_pts);

        traj_num = cached_traj_num;
      }
    });

  });

  // Doing initial learning
  for (unsigned int i = 0; i < 100; ++i) {
    auto [new_parl_pts, total_reward] = execute_parl_pid_traj(plan, controlled_traj, parl, env, controlled_traj.length(), false);
    parl_pts = new_parl_pts;
    cached_traj_num = i;
    auto parl_path_error = compute_traj_error(plan, parl_pts);
    log_info("Initial training (run", i, "), point-to-point path execution error was", parl_path_error);
    log_info("Initial training (run", i, "), average path execution error was", parl_path_error / parl_pts.size());
    log_info("Initial training (run", i, "), had trajectory tracking reward", total_reward);
  }

  // Use PARL to augment control, compute error in same way
  for (unsigned int i = 0; i < 1000; ++i) {
    auto [new_parl_pts, total_reward] = execute_parl_pid_traj(plan, controlled_traj, parl, env, controlled_traj.length(), true);
    parl_pts = new_parl_pts;
    cached_traj_num = i;
    auto parl_path_error = compute_traj_error(plan, parl_pts);
    log_info("With PARL (run", i, "), point-to-point path execution error was", parl_path_error);
    log_info("With PARL (run", i, "), average path execution error was", parl_path_error / parl_pts.size());
    log_info("With PARL (run", i, "), had trajectory tracking reward", total_reward);

  } 

  plot_thread.join();
}
