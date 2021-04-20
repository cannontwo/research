#include <ompl/control/SpaceInformation.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/control/ODESolver.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/config.h>

#include <cannon/research/parl/envs/kinematic_car.hpp>
#include <cannon/physics/systems/kinematic_car.hpp>
#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl/hyperparams.hpp>
#include <cannon/research/parl_planning/aggregate_model.hpp>

using namespace cannon::research::parl;
using namespace cannon::physics::systems;

Vector4d compute_error_state(const Vector3d& ref, const Vector3d& actual, double time) {
  Vector3d diff = ref - actual;
  Vector4d ret;
  ret.head(3) = diff;
  ret[3] = time;

  return ret;
}

std::vector<double> get_ref_point_times(oc::PathControl& path) {
  // Placing reference points halfway between each waypoint on the path
  std::vector<double> durations = path.getControlDurations();
  std::vector<double> points;
  
  double accumulated_dur = 0.0;
  for (unsigned int i = 0; i < durations.size(); i++) {
    log_info("Found planned duration", durations[i]);
    points.push_back(accumulated_dur + (0.5 * durations[i])); 
    accumulated_dur += durations[i];
  }

  return points;
}

MatrixXd make_error_system_refs(oc::PathControl& path) {
  auto times = get_ref_point_times(path);

  // Since this is the error system, all dimensions except for time should be zero
  MatrixXd refs = MatrixXd::Zero(4, times.size());
  for (unsigned int i = 0; i < times.size(); i++) {
    refs(3, i) = times[i];
  }

  return refs;
}

void execute_control_for_duration(std::shared_ptr<KinematicCarEnvironment> env, std::shared_ptr<Parl> parl, const Vector3d& s,
    const Vector3d& next_s, const Vector2d& c, 
    double start_time, std::chrono::milliseconds control_dur, bool learn=true) {

  auto start = std::chrono::steady_clock::now();
  std::chrono::milliseconds cur_dur;

  Vector4d error_state = compute_error_state(s, env->get_state(), 0.0);

  VectorXd new_state;
  double reward;
  bool done;
  do {
    Vector2d parl_c = Vector2d::Zero();
    if (learn) {
      parl_c = parl->get_action(error_state);
    } 

    //log_info("PARL action is", parl_c);

    // TODO Fix reward computation---should be relative to path, not directed at goal
    std::tie(new_state, reward, done) = env->step(c + parl_c);
    env->render();

    // TODO Check for ICS? Or somehow else execute emergency stopping maneuver?

    // Compute error coordinates for new_state
    double t = (float)cur_dur.count() / (float)control_dur.count();
    Vector3d interp_ref = ((1.0 - t) * s) + (t * next_s);
    double time = ((double)cur_dur.count() / 1000.0) + start_time;
    Vector4d new_error_state = compute_error_state(interp_ref, new_state, time);

    // Train PARL using error states
    if (learn) {
      parl->process_datum(error_state, parl_c, reward, new_error_state, done);
    }

    error_state = new_error_state;

    cur_dur = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
  } while (control_dur > cur_dur);

}

void execute_path(std::shared_ptr<KinematicCarEnvironment> env, oc::PathControl& path, 
    std::shared_ptr<Parl> parl, double tracking_threshold=0.5) {
  std::vector<ob::State*> states = path.getStates();
  std::vector<oc::Control*> controls = path.getControls();
  std::vector<double> durations = path.getControlDurations();
  assert(durations.size() == controls.size());

  double accumulated_dur = 0.0;
  for (unsigned int i = 0; i < controls.size(); i++) {
    Vector3d s;
    s[0] = states[i]->as<ob::SE2StateSpace::StateType>()->getX();
    s[1] = states[i]->as<ob::SE2StateSpace::StateType>()->getY();
    s[2] = states[i]->as<ob::SE2StateSpace::StateType>()->getYaw();

    Vector3d next_s = s;
    if (i < controls.size() - 1) {
      next_s[0] = states[i+1]->as<ob::SE2StateSpace::StateType>()->getX();
      next_s[1] = states[i+1]->as<ob::SE2StateSpace::StateType>()->getY();
      next_s[2] = states[i+1]->as<ob::SE2StateSpace::StateType>()->getYaw();
    }

    // Break path tracking if state error greater than threshold
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

    Vector2d c;
    c[0] = controls[i]->as<oc::RealVectorControlSpace::ControlType>()->values[0];
    c[1] = controls[i]->as<oc::RealVectorControlSpace::ControlType>()->values[1];

    unsigned int duration_millis = std::floor(durations[i] * 1000);
    std::chrono::milliseconds control_dur(duration_millis);
    
    execute_control_for_duration(env, parl, s, next_s, c, accumulated_dur, control_dur);
    accumulated_dur += durations[i];
  }

  Vector2d goal;
  goal[0] = 1.0;
  goal[1] = 1.0;
  log_info("Goal error on real system is", (env->get_state().head(2) - goal).norm());
}

oc::PathControl plan_to_goal(std::shared_ptr<System> sys, std::shared_ptr<KinematicCarEnvironment> env,
    Vector3d& start_state, Vector3d& goal_state) {
  oc::SimpleSetup ss(env->get_action_space());
  auto si = ss.getSpaceInformation();

  si->setStateValidityChecker([&](const ob::State *state) {
        // TODO Make real validity checker with obstacle geometry
        return true;
      });

  // TODO Rather than creating an ODESolver for KinCarSystem directly, should
  // create a wrapper class that integrates PARL learned dynamics
  
  auto odeSolver(std::make_shared<oc::ODEBasicSolver<>>(si, 
        [&](const oc::ODESolver::StateType& q, 
            const oc::Control* control, oc::ODESolver::StateType& qdot){
          sys->ompl_ode_adaptor(q, control, qdot);
        }));

  // Make it clear that KinCarSystem is discrete-time
  si->setStatePropagator(oc::ODESolver::getStatePropagator(odeSolver,
        KinCarSystem::ompl_post_integration));

  ob::ScopedState<ob::SE2StateSpace> start(env->get_state_space());
  start->setX(start_state[0]);
  start->setY(start_state[1]);
  start->setYaw(start_state[2]);

  ob::ScopedState<ob::SE2StateSpace> goal(env->get_state_space());
  goal->setX(goal_state[0]);
  goal->setY(goal_state[1]);
  goal->setYaw(goal_state[2]);

  ss.setStartAndGoalStates(start, goal, 0.05);

  // Necessary for discrete-time model
  ss.getSpaceInformation()->setPropagationStepSize(env->get_time_step());
  log_info("Propagation step size is", ss.getSpaceInformation()->getPropagationStepSize());

  auto planner = std::make_shared<oc::SST>(ss.getSpaceInformation());
  ss.setPlanner(planner);

  ss.setup();

  ob::PlannerStatus solved = ss.solve(1.0);
  if (solved) {
    std::cout << "Found solution:" << std::endl;
    ss.getSolutionPath().printAsMatrix(std::cout);

    return ss.getSolutionPath();
  } else {
    throw std::runtime_error("No solution found" );
  }
}

std::shared_ptr<ob::StateSpace> make_error_state_space(std::shared_ptr<Environment> env, double duration) {
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

int main() {
  auto env = std::make_shared<KinematicCarEnvironment>();

  // Planning for a different length of car
  auto nominal_sys = std::make_shared<KinCarSystem>(0.5);

  Vector3d start = Vector3d::Zero();
  Vector3d goal;
  goal[0] = 1.0;
  goal[1] = 1.0;
  goal[2] = 0.0;

  MatrixX2d state_space_bounds = env->get_state_space_bounds();

  
  auto planning_sys = std::make_shared<AggregateModel>(nominal_sys,
      env->get_state_space()->getDimension(),
      env->get_action_space()->getDimension(), 10, state_space_bounds,
      env->get_time_step());

  while ((env->get_state() - goal).norm() > 0.1) {
    start = env->get_state();
    //auto path = plan_to_goal(nominal_sys, env, start, goal);
    auto path = plan_to_goal(planning_sys, env, start, goal);

    // TODO Do we want to adjust bounds on action space? State space?
    Hyperparams params;
    params.load_config("/home/cannon/Documents/cannon/cannon/research/experiments/parl_configs/r10c10_kc.yaml"); 
    auto parl = std::make_shared<Parl>(make_error_state_space(env,
          path.length()), env->get_action_space(),
        make_error_system_refs(path), params);

    execute_path(env, path, parl);

    // Update model used by planner to incorporate dynamics learned while
    // following this path.
    planning_sys->process_path_parl(env, parl, path);
  }

  log_info("Made it to goal!");
}
