#include <ompl/base/goals/GoalState.h>
#include <ompl/config.h>
#include <ompl/control/ODESolver.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/util/RandomNumbers.h>

#include <cannon/physics/systems/kinematic_car.hpp>
#include <cannon/research/parl/envs/kinematic_car.hpp>
#include <cannon/research/parl/hyperparams.hpp>
#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl_planning/aggregate_model.hpp>
#include <cannon/utils/experiment_runner.hpp>
#include <cannon/utils/experiment_writer.hpp>

using namespace cannon::research::parl;
using namespace cannon::physics::systems;
using namespace cannon::utils;

static Vector2d s_goal = Vector2d::Ones();
static int overall_timestep = 0;

// Whether to do learning
static bool learn = false;

Vector4d compute_error_state(const Vector3d &ref, const Vector3d &actual,
                             double time) {
  Vector3d diff = ref - actual;
  Vector4d ret;
  ret.head(3) = diff;
  ret[3] = time;

  return ret;
}

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
  MatrixXd refs = MatrixXd::Zero(4, times.size());
  for (unsigned int i = 0; i < times.size(); i++) {
    refs(3, i) = times[i];
  }

  return refs;
}

void execute_control_for_duration(std::shared_ptr<KinematicCarEnvironment> env,
                                  std::shared_ptr<Parl> parl, const Vector3d &s,
                                  const Vector3d &next_s, const Vector2d &c,
                                  double start_time,
                                  std::chrono::milliseconds control_dur,
                                  ExperimentWriter &w) {

  double cur_dur = 0.0;
  Vector4d error_state = compute_error_state(s, env->get_state(), start_time);

  VectorXd new_state;
  bool done;
  double total_seg_reward = 0.0;
  do {
    Vector2d parl_c = Vector2d::Zero();
    if (learn) {
      parl_c = parl->get_action(error_state);
    }

    // log_info("PARL action is", parl_c);

    std::stringstream ss;
    ss << overall_timestep << "," << env->get_state()[0] << ","
       << env->get_state()[1] << "," << env->get_state()[2] << ","
       << (c + parl_c)[0] << "," << (c + parl_c)[1];
    w.write_line("executed_traj", ss.str());

    double env_reward;
    std::tie(new_state, env_reward, done) = env->step(c + parl_c);
    // env->render();

    overall_timestep++;

    ss.str("");
    ss << overall_timestep << "," << (new_state.head(2) - s_goal).norm();
    w.write_line("distances", ss.str());

    // TODO Check for ICS? Or somehow else execute emergency stopping maneuver?

    // Compute error coordinates for new_state
    double t = cur_dur / (float)control_dur.count();
    Vector3d interp_ref = ((1.0 - t) * s) + (t * next_s);
    double time = cur_dur + start_time;
    Vector4d new_error_state = compute_error_state(interp_ref, new_state, time);

    ss.str("");
    ss << overall_timestep << "," << interp_ref[0] << "," << interp_ref[1]
       << "," << interp_ref[2] << "," << c[0] << "," << c[1];
    w.write_line("planned_traj", ss.str());

    // Reward for PARL is relative to path
    double reward = -(interp_ref - new_state).norm();
    total_seg_reward += reward;

    // Train PARL using error states
    if (learn) {
      parl->process_datum(error_state, parl_c, reward, new_error_state, done);

      // TODO Is this the correct way to do controller updates? Do we want to
      // cache learned PARL controllers as well?
      parl->value_grad_update_controller(error_state);
    }

    error_state = new_error_state;

    cur_dur += env->get_time_step();
  } while (control_dur.count() > cur_dur * 1000);

  // Plotting reward for an entire control segment
  env->register_ep_reward(total_seg_reward);
}

void execute_path(std::shared_ptr<KinematicCarEnvironment> env,
                  oc::PathControl &path, std::shared_ptr<Parl> parl,
                  ExperimentWriter &w, double tracking_threshold = 1.0) {
  std::vector<ob::State *> states = path.getStates();
  std::vector<oc::Control *> controls = path.getControls();
  std::vector<double> durations = path.getControlDurations();
  assert(durations.size() == controls.size());

  double accumulated_dur = 0.0;
  for (unsigned int i = 0; i < controls.size(); i++) {
    Vector3d s = Vector3d::Zero();
    s[0] = states[i]->as<ob::SE2StateSpace::StateType>()->getX();
    s[1] = states[i]->as<ob::SE2StateSpace::StateType>()->getY();
    s[2] = states[i]->as<ob::SE2StateSpace::StateType>()->getYaw();

    Vector3d next_s = s;
    if (i < controls.size() - 1) {
      next_s[0] = states[i + 1]->as<ob::SE2StateSpace::StateType>()->getX();
      next_s[1] = states[i + 1]->as<ob::SE2StateSpace::StateType>()->getY();
      next_s[2] = states[i + 1]->as<ob::SE2StateSpace::StateType>()->getYaw();
    }

    // Break path tracking if state error greater than threshold. This is a
    // check at each waypoint.
    double path_error = (s - env->get_state()).norm();
    log_info("At step", i, "xy path error is", path_error);
    if (path_error > tracking_threshold) {
      // TODO This implicitly assumes that there is no autonomous dynamical
      // behavior; instead, a stopping maneuver should be executed before
      // replanning (or, under different assumptions, replanning should be done
      // for the predicted state after planning time).

      log_info("Replanning because tracking threshold exceeded");
      return;
    }

    Vector2d c = Vector2d::Zero();
    c[0] =
        controls[i]->as<oc::RealVectorControlSpace::ControlType>()->values[0];
    c[1] =
        controls[i]->as<oc::RealVectorControlSpace::ControlType>()->values[1];

    unsigned int duration_millis = std::floor(durations[i] * 1000);
    std::chrono::milliseconds control_dur(duration_millis);

    // log_info("Executing control", c, "for", control_dur.count(),
    // "milliseconds");

    execute_control_for_duration(env, parl, s, next_s, c, accumulated_dur,
                                 control_dur, w);
    accumulated_dur += durations[i];
  }

  log_info("Goal error on real system is",
           (env->get_state().head(2) - s_goal).norm());
}

oc::PathControl plan_to_goal(std::shared_ptr<System> sys,
                             std::shared_ptr<KinematicCarEnvironment> env,
                             Vector3d &start_state, Vector3d &goal_state) {
  ob::ScopedState<ob::SE2StateSpace> start(env->get_state_space());
  start->setX(start_state[0]);
  start->setY(start_state[1]);
  start->setYaw(start_state[2]);

  ob::ScopedState<ob::SE2StateSpace> goal(env->get_state_space());
  goal->setX(goal_state[0]);
  goal->setY(goal_state[1]);
  goal->setYaw(goal_state[2]);

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
      odeSolver, KinCarSystem::ompl_post_integration));

  ss.setStartAndGoalStates(start, goal, 0.05);

  // Necessary for discrete-time model
  ss.getSpaceInformation()->setPropagationStepSize(env->get_time_step());
  log_info("Propagation step size is",
           ss.getSpaceInformation()->getPropagationStepSize());

  auto planner = std::make_shared<oc::SST>(ss.getSpaceInformation());
  ss.setPlanner(planner);

  ss.setup();

  ob::PlannerStatus solved = ss.solve(1.0);
  if (solved) {
    std::cout << "Found solution:" << std::endl;
    ss.getSolutionPath().printAsMatrix(std::cout);

    return ss.getSolutionPath();
  } else {
    return oc::PathControl(si);
  }
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

  overall_timestep = 0;

  auto env = std::make_shared<KinematicCarEnvironment>();

  // Planning for a different length of car
  auto nominal_sys = std::make_shared<KinCarSystem>(1.0);

  Vector3d start = Vector3d::Zero();
  Vector3d goal;
  goal[0] = s_goal[0];
  goal[1] = s_goal[1];
  goal[2] = 0.0;

  env->reset(start);

  MatrixX2d state_space_bounds = env->get_state_space_bounds();

  auto planning_sys = std::make_shared<AggregateModel>(
      nominal_sys, env->get_state_space()->getDimension(),
      env->get_action_space()->getDimension(), 10, state_space_bounds,
      env->get_time_step(), learn);

  w.start_log("distances");
  w.write_line("distances", "timestep,distance");

  // TODO Record closest approach to goal at each timestep up to max number of
  // timesteps
  // TODO Run planner without PARL as well

  while ((env->get_state() - goal).norm() > 0.1 && overall_timestep < 1e3) {
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
                       "experiments/parl_configs/r10c10_kc.yaml");
    auto parl = std::make_shared<Parl>(
        make_error_state_space(env, path.length()), env->get_action_space(),
        make_error_system_refs(path), params, seed);

    w.start_log("planned_traj");
    w.write_line("planned_traj",
                 "timestep,statex,statey,stateth,controlv,controlth");

    w.start_log("executed_traj");
    w.write_line("executed_traj",
                 "timestep,statex,statey,stateth,controlv,controlth");

    execute_path(env, path, parl, w);

    // Update model used by planner to incorporate dynamics learned while
    // following this path.
    if (learn)
      planning_sys->process_path_parl(env, parl, path);
  }

  if ((env->get_state() - goal).norm() < 0.1)
    log_info("Made it to goal!");
  else
    log_info("Maximum timesteps exceeded");
}

int main() {
  std::string log_path;

  if (learn) {
    log_path = std::string("logs/parl_planning_exps/learning/");
  } else {
    log_path = std::string("logs/parl_planning_exps/no_learning/");
  }

  ExperimentRunner runner(log_path, 10, run_exp);

  runner.run();
}
