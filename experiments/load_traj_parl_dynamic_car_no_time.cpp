#include <cannon/research/parl/envs/dynamic_car.hpp>
#include <cannon/control/pid.hpp>
#include <cannon/math/interp.hpp>
#include <cannon/geom/graph.hpp>
#include <cannon/geom/trajectory.hpp>
#include <cannon/plot/plotter.hpp>
#include <cannon/research/parl/parl.hpp>
#include <cannon/utils/experiment_runner.hpp>
#include <cannon/utils/experiment_writer.hpp>

#include <thirdparty/HighFive/include/highfive/H5Easy.hpp>

using namespace cannon::research::parl;
using namespace cannon::control;
using namespace cannon::math;
using namespace cannon::geom;
using namespace cannon::plot;
using namespace cannon::utils;

/*!
 * \brief Struct containing parameters to be varied in experiments
 */
struct ExpParams {
  double l; //!< Environment axle length param
  double ref_radius; //!< PARL ref radius
  std::vector<int> ref_grid_sizes; //!< PARL ref grid sizes
};

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

VectorXd compute_error_state(const VectorXd& ref, const VectorXd& actual) {
  assert(ref.size() == actual.size());

  VectorXd diff = ref - actual;
  VectorXd ret = VectorXd::Zero(ref.size());
  ret.head(ref.size()) = diff;

  return ret;
}

std::vector<Vector2d> execute_traj(const ControlledTrajectory &traj,
                                   std::shared_ptr<DynamicCarEnvironment> env) {

  double time = 0.0;
  VectorXd state = env->reset(VectorXd::Zero(5));
  //env->render();
  std::vector<Vector2d> executed;

  for (unsigned int i = 0; i < (1.0/env->get_time_step()) * traj.length(); ++i) {
    std::cout << "\r" << "On step " << i << std::flush;
    executed.push_back(state.head(2));

    VectorXd pid_action = traj(time).second;

    // Compute PARL action
    auto plan_state = traj(time).first;

    double reward;
    bool done;
    std::tie(state, reward, done) = env->step(pid_action);
    //env->render();
    
    time += env->get_time_step();
  }

  executed.push_back(state.head(2));
  return executed;
}

std::pair<std::vector<Vector2d>, double>
execute_parl_pid_traj(const ControlledTrajectory &traj,
                      std::shared_ptr<Parl> parl,
                      std::shared_ptr<DynamicCarEnvironment> env,
                      bool do_controller_update,
                      bool testing=false) {

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
    auto error_state = compute_error_state(plan_state, state);
    auto parl_action = parl->get_action(error_state, testing);

    auto combined_action = parl_action + pid_action;

    double reward;
    bool done;
    std::tie(state, reward, done) = env->step(combined_action);
    //env->render();
    
    auto constrained_combined_action = env->get_constrained_control(combined_action);
    auto effective_parl_action = constrained_combined_action - pid_action;

    time += env->get_time_step();

    auto traj_state = traj(time).first;

    double tracking_reward = -((state.head(2) - plan_state.head(2)).norm()) -
                             0.2 * effective_parl_action.norm();

    total_reward += tracking_reward;

    // Train PARL
    auto new_plan_state = traj(time).first;
    auto new_error_state = compute_error_state(new_plan_state, state);
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

void run_exp(ExperimentWriter &w, int seed, const ExpParams& experiment_params) {
  assert(!(render_interactive && render_value));

  w.start_log("train_rewards");
  w.write_line("train_rewards", "episode,reward");

  w.start_log("train_distances");
  w.write_line("train_distances", "episode,distance");

  w.start_log("test_reward");
  w.write_line("test_reward", "reward");

  w.start_log("test_distance");
  w.write_line("test_distance", "distance");

  auto env = std::make_shared<DynamicCarEnvironment>(VectorXd::Zero(5),
                                                     VectorXd::Ones(5), experiment_params.l);

  ControlledTrajectory traj;
  traj.load("logs/sst_dynamic_car_plan.h5");

  Hyperparams params;
  params.load_config("/home/cannon/Documents/cannon/cannon/research/"
                     "experiments/parl_configs/r10c10_dc.yaml");

  MatrixXd refs = env->sample_grid_refs(experiment_params.ref_grid_sizes) * experiment_params.ref_radius;

  auto parl = std::make_shared<Parl>(env->get_state_space(),
                                     env->get_action_space(), refs, params);

  auto real_pts = execute_traj(traj, env);
  H5Easy::File no_parl_traj_file(w.get_dir() + "/no_parl_traj.h5", H5Easy::File::Overwrite);
  std::string states_path("/states/");
  for (unsigned int i = 0; i < real_pts.size(); ++i) {
    H5Easy::dump(no_parl_traj_file, states_path + std::to_string(i), real_pts[i]);
  }
  log_info("On real system, point-to-point path execution error is", compute_traj_error(traj, real_pts));

  //std::vector<Vector2d> parl_pts;
  //int cached_traj_num = -1;
  //std::thread plot_thread;
  //if (render_interactive) {
  //  plot_thread = std::thread([&]() {
  //    Plotter plotter;

  //    plotter.render([&]() {
  //      static int traj_num = cached_traj_num;

  //      if (traj_num != cached_traj_num) {
  //        plotter.clear();
  //        plotter.plot([&](double t){return traj(t).first;}, 200, 0.0, traj.length());
  //        plotter.plot(real_pts);
  //        plotter.plot(parl_pts);

  //        traj_num = cached_traj_num;
  //      }
  //    });

  //  });
  //}


  // Doing initial learning
  for (unsigned int i = 0; i < 100; ++i) {
    auto [parl_pts, total_reward] = execute_parl_pid_traj(traj, parl, env, false);
    //parl_pts = new_parl_pts;
    //cached_traj_num = i;
    auto parl_path_error = compute_traj_error(traj, parl_pts);
    log_info("Initial training (run", i, "), point-to-point path execution error was", parl_path_error);
    log_info("Initial training (run", i, "), average path execution error was", parl_path_error / parl_pts.size());
    log_info("Initial training (run", i, "), had trajectory tracking reward", total_reward);
  }

  // Use PARL to augment control, compute error in same way
  for (unsigned int i = 0; i < 100; ++i) {
    auto [parl_pts, total_reward] = execute_parl_pid_traj(traj, parl, env, true);
    //parl_pts = new_parl_pts;
    //cached_traj_num = i;
    auto parl_path_error = compute_traj_error(traj, parl_pts);
    log_info("With PARL (run", i, "), point-to-point path execution error was", parl_path_error);
    log_info("With PARL (run", i, "), average path execution error was", parl_path_error / parl_pts.size());
    log_info("With PARL (run", i, "), had trajectory tracking reward", total_reward);

    // Write to file
    std::stringstream ss;
    ss << i << "," << total_reward;
    w.write_line("train_rewards", ss.str());

    ss.clear();
    ss.str("");
    ss << i << "," << parl_path_error;
    w.write_line("train_distances", ss.str());
  } 

  // Testing
  auto [parl_pts, total_reward] = execute_parl_pid_traj(traj, parl, env, false, true);
  H5Easy::File final_traj_file(w.get_dir() + "/final_traj.h5", H5Easy::File::Overwrite);
  for (unsigned int i = 0; i < parl_pts.size(); ++i) {
    H5Easy::dump(final_traj_file, states_path + std::to_string(i), parl_pts[i]);
  }

  //parl_pts = new_parl_pts;
  auto parl_path_error = compute_traj_error(traj, parl_pts);
  log_info("Testing mode point-to-point path execution error was", parl_path_error);
  log_info("Testing mode average path execution error was", parl_path_error / parl_pts.size());
  log_info("Testing mode had trajectory tracking reward", total_reward);

  std::stringstream ss;
  ss << total_reward;
  w.write_line("test_reward", ss.str());

  ss.clear();
  ss.str("");
  ss << parl_path_error;
  w.write_line("test_distance", ss.str());

  //if (render_interactive) {
  //  plot_thread.join();
  //}
  
  // Plotting learned value function
  //if (render_value) {
  //  Plotter plotter;
  //  plotter.render([&]() {
  //    static float time = 0.0;

  //    bool changed = false;
  //    if (ImGui::BeginMainMenuBar()) {
  //      if (ImGui::BeginMenu("Reward Plotting")) {
  //        changed = changed || ImGui::SliderFloat("Traj time", &time, 0.0, traj.length());

  //        ImGui::EndMenu();
  //      }
  //      ImGui::EndMainMenuBar();
  //    }

  //    if (changed) {
  //      plotter.clear();

  //      auto plan_state = traj(time).first;
  //      plotter.plot([&](const Vector2d &p) {
  //        VectorXd full_p = VectorXd::Zero(5);
  //        full_p.head(2) = p;
  //        full_p.tail(3) = plan_state.tail(3);
  //        return parl->predict_value(compute_error_state(plan_state, full_p));
  //      }, 15, plan_state[0] - 0.4, plan_state[0] + 0.4, plan_state[1] - 0.4, plan_state[1] + 0.4);
  //      plotter.plot([&](double t){
  //          return traj(t).first;
  //          }, 200, 0.0, traj.length());

  //      MatrixX2f ref_points = MatrixX2f::Zero(refs.cols(), 2);
  //      for (unsigned int i = 0; i < refs.cols(); ++i) {
  //        ref_points.row(i) = traj(time).first.head(2).cast<float>() + refs.col(i).head(2).cast<float>();
  //      }
  //      plotter.plot_points(ref_points, {1.0, 0.0, 0.0, 1.0});
  //    }
  //  });
  //}
  
}

int main() {
  ExperimentRunner runner("logs/parl_planning_nt_large_axle", 10,
                          [](ExperimentWriter &w, int seed) {
                            auto f = w.get_file("seed.txt");
                            f << seed;
                            f.close();

                            run_exp(w, seed, {1.3, 0.1, {5, 5, 1, 1, 1}});
                          });

  runner.run();
}
