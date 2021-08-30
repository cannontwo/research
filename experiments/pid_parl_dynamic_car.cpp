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
  //traj.push_back(VectorXd::Ones(2), 0.0);
  //traj.push_back(VectorXd::Ones(2), 10.0);
  for (unsigned int i = 0; i < path.size(); ++i) {
    VectorXd spt(2);
    spt << locs[path[i]].first,
           locs[path[i]].second;
    traj.push_back(spt, i);
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
    executed.push_back(state.head(2));

    controller.ref() = plan(time);
    VectorXd pid_action = controller.get_control(state.head(2));

    double reward;
    bool done;
    std::tie(state, reward, done) = env->step(pid_action);
    //env->render();

    time += env->get_time_step();
  }

  executed.push_back(state.head(2));
  return executed;
}

std::vector<Vector2d>
execute_parl_pid_traj(const MultiSpline &plan, PidController &controller,
                      std::shared_ptr<Parl> parl,
                      std::shared_ptr<DynamicCarEnvironment> env,
                      double length) {

  controller.reset();

  double time = 0.0;
  VectorXd state = env->reset(VectorXd::Zero(5));
  //env->render();
  std::vector<Vector2d> executed;
  std::vector<VectorXd> states;
  for (unsigned int i = 0; i < 100 * length; ++i) {
    std::cout << "\r" << "On step " << i << std::flush;
    executed.push_back(state.head(2));

    controller.ref() = plan(time);
    VectorXd pid_action = controller.get_control(state.head(2));

    // Compute PARL action
    VectorXd plan_state(5);
    plan_state.head(2) = plan(time);
    auto error_state = compute_error_state(plan_state, state, time);
    auto parl_action = parl->get_action(error_state);

    auto combined_action = parl_action + pid_action;

    double reward;
    bool done;
    std::tie(state, reward, done) = env->step(combined_action);
    //env->render();

    time += env->get_time_step();

    double tracking_reward = -((state.head(2) - plan(time)).norm());

    // Train PARL
    VectorXd new_plan_state(5);
    new_plan_state.head(2) = plan(time);
    auto new_error_state = compute_error_state(new_plan_state, state, time);
    parl->process_datum(error_state, parl_action, tracking_reward, new_error_state);
    states.push_back(error_state);
  }

  for (auto state : states) {
    parl->value_grad_update_controller(state);
  }

  executed.push_back(state.head(2));
  return executed;
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

  auto pts = execute_pid_traj(plan, controller, env, traj.length());
  auto path_error = compute_traj_error(plan, pts);

  // Compute error between plan and controlled path
  log_info("Without PARL, point-to-point path execution error was", path_error);
  log_info("Without PARL, average path execution error was", path_error / pts.size());

  // Use PARL to augment control, compute error in same way
  for (unsigned int i = 0; i < 100; ++i) {
    auto parl_pts = execute_parl_pid_traj(plan, controller, parl, env, traj.length());
    auto parl_path_error = compute_traj_error(plan, parl_pts);
    log_info("With PARL (run", i, "), point-to-point path execution error was", parl_path_error);
    log_info("With PARL (run", i, "), average path execution error was", parl_path_error / parl_pts.size());

    Plotter plotter;
    plotter.plot(traj, 200, 0.0, traj.length());
    plotter.plot(plan, 200, 0.0, traj.length());
    plotter.plot(parl_pts);
    plotter.render();
  } 
}
