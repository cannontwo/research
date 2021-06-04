#include <ompl/base/goals/GoalState.h>
#include <ompl/config.h>
#include <ompl/control/ODESolver.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/util/RandomNumbers.h>

#include <cannon/physics/systems/dynamic_car.hpp>
#include <cannon/research/parl/envs/dynamic_car.hpp>
#include <cannon/research/parl/hyperparams.hpp>
#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl_planning/executor.hpp>
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

std::vector<double> get_ref_point_times(oc::PathControl &path) {
  // Placing reference points halfway between each waypoint on the path
  std::vector<double> durations = path.getControlDurations();
  std::vector<double> points;

  double accumulated_dur = 0.0;
  for (unsigned int i = 0; i < durations.size(); i++) {
    points.push_back(accumulated_dur + (0.5 * durations[i]));
    accumulated_dur += durations[i];
  }

  return points;
}


MatrixXd make_error_system_refs(oc::PathControl &path) {
  auto times = get_ref_point_times(path);

  // Since this is the error system, all dimensions except for time should be
  // zero
  MatrixXd refs = MatrixXd::Zero(6, times.size());
  for (unsigned int i = 0; i < times.size(); i++) {
    refs(5, i) = times[i];
  }

  return refs;
}


oc::PathControl plan_to_goal(std::shared_ptr<System> sys,
                             std::shared_ptr<DynamicCarEnvironment> env,
                             VectorXd &start_state, VectorXd &goal_state) {
  assert(start_state.size() == env->get_state_space()->getDimension());
  assert(goal_state.size() == env->get_action_space()->getDimension());

  ob::ScopedState<ob::CompoundStateSpace> start(env->get_state_space());
  start->as<ob::SE2StateSpace::StateType>(0)->setX(start_state[0]);
  start->as<ob::SE2StateSpace::StateType>(0)->setY(start_state[1]);
  start->as<ob::SE2StateSpace::StateType>(0)->setYaw(start_state[2]);
  start->as<ob::RealVectorStateSpace::StateType>(1)->values[0] = start_state[3];
  start->as<ob::RealVectorStateSpace::StateType>(1)->values[1] = start_state[4];

  ob::ScopedState<ob::CompoundStateSpace> goal(env->get_state_space());
  goal->as<ob::SE2StateSpace::StateType>(0)->setX(goal_state[0]);
  goal->as<ob::SE2StateSpace::StateType>(0)->setY(goal_state[1]);
  goal->as<ob::SE2StateSpace::StateType>(0)->setYaw(goal_state[2]);
  goal->as<ob::RealVectorStateSpace::StateType>(1)->values[0] = goal_state[3];
  goal->as<ob::RealVectorStateSpace::StateType>(1)->values[1] = goal_state[4];

  oc::SimpleSetup ss(env->get_action_space());
  auto si = ss.getSpaceInformation();

  si->setStateValidityChecker([si](const ob::State *state) {
    // TODO Make real validity checker with obstacle geometry
    return si->satisfiesBounds(state);
  });

  auto odeSolver(std::make_shared<oc::ODEBasicSolver<>>(
      si, [&](const oc::ODESolver::StateType &q, const oc::Control *control,
              oc::ODESolver::StateType &qdot) {
        sys->ompl_ode_adaptor(q, control, qdot);
      }));

  // Make it clear that KinCarSystem is discrete-time
  si->setStatePropagator(oc::ODESolver::getStatePropagator(
      odeSolver, DynamicCarSystem::ompl_post_integration));

  ss.setStartAndGoalStates(start, goal, 0.1);

  // Necessary for discrete-time model
  ss.getSpaceInformation()->setPropagationStepSize(env->get_time_step());
  log_info("Propagation step size is",
           ss.getSpaceInformation()->getPropagationStepSize());

  //auto planner = std::make_shared<oc::SST>(ss.getSpaceInformation());
  auto planner = std::make_shared<oc::RRT>(ss.getSpaceInformation());
  ss.setPlanner(planner);

  ss.setup();

  double plan_time = 1.0;
  bool have_plan = false;

  while (!have_plan && plan_time < 30.0) {

    log_info("Planning for", plan_time, "seconds");
    ob::PlannerStatus solved = ss.solve(plan_time);
    if (solved) {
      if (ss.haveExactSolutionPath()) {
        std::cout << "Found solution:" << std::endl;
        ss.getSolutionPath().printAsMatrix(std::cout);

        return ss.getSolutionPath();
      } else {
        plan_time *= 2.0;
      }
    } else {
      plan_time *= 2.0;
    }
  }

  // If we get here, planning has failed 
  return oc::PathControl(si);
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

void run_exp(ExperimentWriter &w, int seed) {
  // Store seed for future reference
  auto seed_file = w.get_file("seed.txt");
  seed_file << seed;
  seed_file.close();

  ompl::RNG::setSeed(seed);

  auto env = std::make_shared<DynamicCarEnvironment>();

  // Planning for a different length of car
  auto nominal_sys = std::make_shared<DynamicCarSystem>(0.1);

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

  // TODO Record closest approach to goal at each timestep up to max number of
  // timesteps
  // TODO Run planner without PARL as well
  Executor executor(env, goal, tracking_threshold, learn);

  while ((env->get_state() - goal).norm() > 0.1 && executor.get_overall_timestep() < 1e3) {
    start = env->get_state();
    // auto path = plan_to_goal(nominal_sys, env, start, goal);

    auto path = plan_to_goal(planning_sys, env, start, goal);
    if (path.getControlCount() == 0) {
      log_info("No plan found");
      break;
    }

    // TODO Do we want to adjust bounds on action space? State space?
    Hyperparams params;
    params.load_config("/home/cannon/Documents/cannon/cannon/research/"
                       "experiments/parl_configs/r10c10_dc.yaml");
    auto parl = std::make_shared<Parl>(
        make_error_state_space(env, path.length()), env->get_action_space(),
        make_error_system_refs(path), params, seed);

    w.start_log("planned_traj");
    w.write_line("planned_traj",
                 "timestep,statex,statey,stateth,statev,statedth,controla,controlth");

    w.start_log("executed_traj");
    w.write_line("executed_traj",
                 "timestep,statex,statey,stateth,statev,statedth,controla,controlth");

    executor.execute_path(path, parl, w);

    // Update model used by planner to incorporate dynamics learned while
    // following this path.
    if (learn)
      planning_sys->process_path_parl(env, parl, path);

    // Compute and write AggregateModel overall linearization error
    double error = planning_sys->compute_model_error(env);
    w.write_line("learned_model_error", std::to_string(executor.get_overall_timestep()) + "," +
                                            std::to_string(error));

    // TODO Compute prediction error as well?
  }

  planning_sys->save(w.get_dir() + "/aggregate_model.h5");

  if ((env->get_state() - goal).norm() < 0.1)
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
