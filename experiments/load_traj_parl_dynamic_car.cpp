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

double compute_traj_error(const ControlledTrajectory &plan, const std::vector<Vector2d>& executed) {
  double path_error = 0.0;
  for (unsigned int i = 0; i < executed.size(); ++i) {
    auto plan_pt = plan(i * 0.01).first;
    path_error += (plan_pt.head(2) - executed[i]).norm();
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

MatrixXd make_error_system_value_refs(const ControlledTrajectory& traj, unsigned int num_refs=100) {
  auto times = get_ref_point_times(traj.length(), num_refs);

  auto spline = traj.state_traj().interp();

  //Trajectory traj_2d;
  //for (unsigned int i = 0; i < times.size(); ++i) {
  //  VectorXd s = VectorXd::Zero(2);
  //  s = spline(times[i]).head(2);
  //  traj_2d.push_back(s, times[i]);
  //}
  //auto spline_2d = traj_2d.interp();

  // Since this is the error system, all dimensions except for time should be
  // zero
  //
  // TODO Right now this is just hard coded along normal vector -- is there a better way to decide how to place refs around trajectory?
  MatrixXd refs = MatrixXd::Zero(6, 3*times.size());
  for (unsigned int i = 0; i < times.size(); i++) {
    refs(5, 3*i) = times[i];

    VectorXd value_ref_pos = VectorXd::Zero(6);
    value_ref_pos.head(5) = spline.normal(times[i]);
    value_ref_pos[5] = times[i];
    refs.col(3*i + 1) = value_ref_pos;

    VectorXd value_ref_neg = VectorXd::Zero(6);
    value_ref_neg.head(5) = -spline.normal(times[i]);
    value_ref_neg[5] = times[i];
    refs.col(3*i + 2) = value_ref_neg;
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

std::pair<std::vector<Vector2d>, double>
execute_parl_pid_traj(const ControlledTrajectory &traj,
                      std::shared_ptr<Parl> parl,
                      std::shared_ptr<DynamicCarEnvironment> env,
                      bool do_controller_update) {

  double time = 0.0;
  VectorXd state = env->reset(VectorXd::Zero(5));
  //env->render();
  std::vector<Vector2d> executed;
  std::vector<VectorXd> states;
  double total_reward = 0.0;
  for (unsigned int i = 0; i < (1.0/env->get_time_step()) * traj.length(); ++i) {
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

    auto traj_state = traj(time).first;

    double tracking_reward = -((state.head(2) - plan_state.head(2)).norm()) - parl_action.norm();

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
  auto env = std::make_shared<DynamicCarEnvironment>(VectorXd::Zero(5),
                                                     VectorXd::Ones(5));

  ControlledTrajectory traj;
  traj.load("logs/sst_dynamic_car_plan.h5");

  Hyperparams params;
  params.load_config("/home/cannon/Documents/cannon/cannon/research/"
                     "experiments/parl_configs/r10c10_dc.yaml");

  MatrixXd refs = make_error_system_refs(traj.length(), 20);
  MatrixX2f ref_points = MatrixX2f::Zero(refs.cols(), 2);
  for (unsigned int i = 0; i < refs.cols(); ++i) {
    VectorXd traj_pt = traj(refs(5, i)).first;
    ref_points.row(i) = traj_pt.head(2).cast<float>();
  }

  MatrixXd value_refs = make_error_system_value_refs(traj, 20);
  MatrixX2f value_ref_points = MatrixX2f::Zero(value_refs.cols(), 2);
  for (unsigned int i = 0; i < value_refs.cols(); ++i) {
    VectorXd traj_pt = traj(value_refs(5, i)).first;
    VectorXd ref_pt = traj_pt + value_refs.col(i).head(5);
    value_ref_points.row(i) = ref_pt.head(2).cast<float>();
  }

  auto parl =
      std::make_shared<Parl>(make_error_state_space(env, traj.length()),
                             env->get_action_space(), refs, value_refs, params);

  DynamicCarEnvironment model_env;
  VectorXd plan_start(5);
  plan_start.head(2) = traj(0.0).first.head(2);
  model_env.reset(plan_start);

  std::vector<Vector2d> parl_pts;
  int cached_traj_num = -1;
  //std::thread plot_thread([&]() {
  //  Plotter plotter;

  //  plotter.render([&]() {
  //    static int traj_num = cached_traj_num;

  //    if (traj_num != cached_traj_num) {
  //      plotter.clear();
  //      plotter.plot([&](double t){return traj(t).first;}, 200, 0.0, traj.length());
  //      plotter.plot(parl_pts);

  //      traj_num = cached_traj_num;
  //    }
  //  });

  //});

  // Doing initial learning
  for (unsigned int i = 0; i < 100; ++i) {
    auto [new_parl_pts, total_reward] = execute_parl_pid_traj(traj, parl, env, false);
    parl_pts = new_parl_pts;
    cached_traj_num = i;
    auto parl_path_error = compute_traj_error(traj, parl_pts);
    log_info("Initial training (run", i, "), point-to-point path execution error was", parl_path_error);
    log_info("Initial training (run", i, "), average path execution error was", parl_path_error / parl_pts.size());
    log_info("Initial training (run", i, "), had trajectory tracking reward", total_reward);
  }

  // Use PARL to augment control, compute error in same way
  for (unsigned int i = 0; i < 100; ++i) {
    auto [new_parl_pts, total_reward] = execute_parl_pid_traj(traj, parl, env, true);
    parl_pts = new_parl_pts;
    cached_traj_num = i;
    auto parl_path_error = compute_traj_error(traj, parl_pts);
    log_info("With PARL (run", i, "), point-to-point path execution error was", parl_path_error);
    log_info("With PARL (run", i, "), average path execution error was", parl_path_error / parl_pts.size());
    log_info("With PARL (run", i, "), had trajectory tracking reward", total_reward);

  } 

  //plot_thread.join();
  
  // Plotting learned value function
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
        VectorXd full_p = VectorXd::Zero(5);
        full_p.head(2) = p;
        auto plan_state = traj(time).first;
        return parl->predict_value(compute_error_state(plan_state, full_p, time));
      }, 15, -2.0, 2.0, -2.0, 2.0);
      plotter.plot([&](double t){
          return traj(t).first;
          }, 200, 0.0, traj.length());
      plotter.plot_points(value_ref_points, {1.0, 0.0, 0.0, 1.0});
    }
  });
  
}
