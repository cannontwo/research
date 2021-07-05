#include <cannon/physics/systems/dynamic_car.hpp>
#include <cannon/research/parl/envs/dynamic_car.hpp>
#include <cannon/research/parl/hyperparams.hpp>
#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl_planning/lqr_executor.hpp>
#include <cannon/research/parl_planning/aggregate_model.hpp>
#include <cannon/utils/experiment_runner.hpp>
#include <cannon/utils/experiment_writer.hpp>

using namespace cannon::research::parl;
using namespace cannon::physics::systems;
using namespace cannon::utils;

static Vector2d s_goal = Vector2d::Ones();

// Whether to do learning
static bool learn = false;
static double tracking_threshold = 1.0;

std::vector<double> get_ref_point_times(oc::PathControl &path, unsigned int num_refs) {
  // Placing reference points halfway between each waypoint on the path
  double total_duration = path.length();
  std::vector<double> points;

  double duration_delta = total_duration / static_cast<double>(num_refs);

  double accumulated_dur = 0.0;
  while (accumulated_dur < total_duration) {
    points.push_back(accumulated_dur);
    accumulated_dur += duration_delta;
  }

  return points;
}

MatrixXd make_error_system_refs(oc::PathControl &path, unsigned int num_refs=100) {
  auto times = get_ref_point_times(path, num_refs);

  // Since this is the error system, all dimensions except for time should be
  // zero
  MatrixXd refs = MatrixXd::Zero(6, times.size());
  for (unsigned int i = 0; i < times.size(); i++) {
    refs(5, i) = times[i];
  }

  return refs;
}

oc::PathControl plan_to_goal() {
  // TODO Create fixed path
}

void run_exp(ExperimentWriter &w, int seed) {
  // Store seed for future reference
  auto seed_file = w.get_file("seed.txt");
  seed_file << seed;
  seed_file.close();

  ompl::RNG::setSeed(seed);

  auto env = std::make_shared<DynamicCarEnvironment>();
  env->reset();

  // Planning for a different length of car
  auto nominal_sys = std::make_shared<DynamicCarSystem>(1.0);

  VectorXd start = VectorXd::Zero(5);
  VectorXd goal = VectorXd::Zero(5);
  goal[0] = s_goal[0];
  goal[1] = s_goal[1];

  env->reset(start);

  MatrixX2d state_space_bounds = env->get_state_space_bounds();

  auto planning_sys = std::make_shared<AggregateModel>(
      nominal_sys, env->get_state_space()->getDimension(),
      env->get_action_space()->getDimension(), 10, state_space_bounds,
      env->get_time_step(), learn);

  w.start_log("distances");
  w.write_line("distances", "timestep,distance");

  w.start_log("learned_model_error");
  w.write_line("learned_model_error", "timestep,error");

  LQRExecutor executor(env, goal.head(2), tracking_threshold, learn);

  auto path = plan_to_goal();
  path.printAsMatrix(std::cout);

  while ((env->get_state().head(2) - goal.head(2)).norm() > 0.1 &&
         executor.get_overall_timestep() <
             executor.get_max_overall_timestep()) {

    w.start_log("planned_traj");
    w.write_line("planned_traj",
                 "timestep,statex,statey,stateth,statev,statedth,controla,controlth");

    w.start_log("executed_traj");
    w.write_line("executed_traj",
                 "timestep,statex,statey,stateth,statev,statedth,controla,controlth");

    auto parl = executor.execute_path(planning_sys, path, w, seed);

    // Update model used by planner to incorporate dynamics learned while
    // following this path.
    if (learn)
      planning_sys->process_path_parl_lqr(parl);

    // Compute and write AggregateModel overall linearization error
    double error = planning_sys->compute_model_error(env);
    w.write_line("learned_model_error", std::to_string(executor.get_overall_timestep()) + "," +
                                            std::to_string(error));

    // TODO Compute prediction error as well?
  }

  planning_sys->save(w.get_dir() + "/aggregate_model.h5");

  if ((env->get_state().head(2) - goal.head(2)).norm() < 0.1)
    log_info("Made it to goal!");
  else
    log_info("Maximum timesteps exceeded");
}

int main(int argc, char **argv) {
  std::string log_path;

  if (argc > 1) {
    if (argv[1][0] == '1')
      learn = true;
  }

  if (learn) {
    log_path = std::string("logs/parl_planning_exps/learning/");
  } else {
    log_path = std::string("logs/parl_planning_exps/no_learning/");
  }

  ExperimentRunner runner(log_path, 10, run_exp);

  runner.run();
}
