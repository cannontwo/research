#include <ompl/base/goals/GoalState.h>
#include <ompl/base/goals/GoalRegion.h>
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
#include <cannon/research/parl_planning/lqr_executor.hpp>
#include <cannon/research/parl_planning/aggregate_model.hpp>
#include <cannon/research/parl_planning/lqr_control_space.hpp>
#include <cannon/research/parl_planning/lqr_state_propagator.hpp>
#include <cannon/utils/experiment_runner.hpp>
#include <cannon/utils/experiment_writer.hpp>

using namespace cannon::research::parl;
using namespace cannon::physics::systems;
using namespace cannon::utils;

static Vector2d s_goal = Vector2d::Ones();

// Whether to do learning
static bool learn = false;
static double tracking_threshold = 1.0;

class DynamicCarProjector : public ob::ProjectionEvaluator
{
public:
    DynamicCarProjector(const ob::StateSpace *space) : ob::ProjectionEvaluator(space)
    {
    }
    unsigned int getDimension() const override
    {
        return 3u;
    }
    void project(const ob::State *state, Eigen::Ref<Eigen::VectorXd> projection) const override
    {
        auto cstate = state->as<ob::CompoundStateSpace::StateType>();
        projection[0] = cstate->as<ob::SE2StateSpace::StateType>(0)->getX();
        projection[1] = cstate->as<ob::SE2StateSpace::StateType>(0)->getY();
        projection[2] = cstate->as<ob::SE2StateSpace::StateType>(0)->getYaw();
    }
};

class SubsetGoalRegion : public ob::GoalRegion {
  public:
    SubsetGoalRegion(const ob::SpaceInformationPtr &si,
                     const Ref<const Vector2d> &geom_goal)
        : ob::GoalRegion(si), geom_goal_(geom_goal) {
      setThreshold(0.1);
    }

    virtual double distanceGoal(const ob::State *st) const override {

      auto se2 =
        si_->getStateSpace()->as<ob::CompoundStateSpace>()->as<ob::SE2StateSpace>(0);
      auto r2 = se2->as<ob::RealVectorStateSpace>(0);

      //auto current_geom = st->as<ob::CompoundStateSpace::StateType>()
      //                        ->as<ob::SE2StateSpace::StateType>(0);
      //auto goal_geom = se2->allocState()->as<ob::SE2StateSpace::StateType>();
      //goal_geom->setX(geom_goal_[0]);
      //goal_geom->setY(geom_goal_[1]);
      //goal_geom->setYaw(geom_goal_[2]);
      //double distance = se2->distance(current_geom, goal_geom);
      //se2->freeState(goal_geom);

      auto current_geom = st->as<ob::CompoundStateSpace::StateType>()
                              ->as<ob::SE2StateSpace::StateType>(0)
                              ->as<ob::RealVectorStateSpace::StateType>(0);
      auto goal_geom = r2->allocState()->as<ob::RealVectorStateSpace::StateType>();
      goal_geom->values[0] = geom_goal_[0];
      goal_geom->values[1] = geom_goal_[1];
      double distance = r2->distance(current_geom, goal_geom);
      r2->freeState(goal_geom);

      return distance;
    }

  private:
    Vector2d geom_goal_;
};

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

oc::PathControl get_vector_control_path(std::shared_ptr<System> sys,
                                        std::shared_ptr<LQRControlSpace> lqr_space,
                                        oc::SpaceInformationPtr vector_si,
                                        oc::PathControl &lqr_path,
                                        double timestep) {
  oc::PathControl ret_path(vector_si);

  std::vector<ob::State *> states = lqr_path.getStates();
  std::vector<oc::Control *> controls = lqr_path.getControls();
  std::vector<double> durations = lqr_path.getControlDurations();
  assert(durations.size() == controls.size());
  assert(states.size() == controls.size() + 1);

  ret_path.append(states[0]);
  
  for (unsigned int i = 0; i < controls.size(); i++) {
    // Compute vector controls for each time step
    
    auto lqr_control = static_cast<LQRControlSpace::ControlType*>(controls[i]);
    
    std::vector<double> prev_waypoint(vector_si->getStateSpace()->getDimension());
    std::vector<double> next_waypoint(vector_si->getStateSpace()->getDimension());

    vector_si->getStateSpace()->copyToReals(prev_waypoint, states[i]);
    vector_si->getStateSpace()->copyToReals(next_waypoint, states[i+1]);

    std::vector<double> cur_interp_waypoint(prev_waypoint);
    
    double time = 0.0;
    while (time < durations[i]) {

      // Compute next linearly interpolated state
      double interp_param = (time + timestep) / durations[i];
      std::vector<double> next_interp_waypoint(vector_si->getStateSpace()->getDimension());
      for (unsigned int i = 0; i < vector_si->getStateSpace()->getDimension(); ++i) {
        next_interp_waypoint[i] = prev_waypoint[i] + interp_param * (next_waypoint[i] - prev_waypoint[i]);
      }

      // Compute vector control
      VectorXd u(2);
      lqr_space->compute_u_star(lqr_control, cur_interp_waypoint, u);

      // Copy to OMPL control
      oc::Control *computed_control = vector_si->getControlSpace()->allocControl();
      computed_control->as<oc::RealVectorControlSpace::ControlType>()
          ->values[0] = u[0];
      computed_control->as<oc::RealVectorControlSpace::ControlType>()
          ->values[1] = u[1];

      // Append to path to be returned
      ob::State* new_path_state = vector_si->allocState();
      vector_si->getStateSpace()->copyFromReals(new_path_state, next_interp_waypoint);
      ret_path.append(new_path_state, computed_control, timestep);

      cur_interp_waypoint = next_interp_waypoint;
      time += timestep;

      vector_si->getControlSpace()->freeControl(computed_control);
      vector_si->getStateSpace()->freeState(new_path_state);
    }
  }
  

  return ret_path;
}

oc::PathControl plan_to_goal(std::shared_ptr<System> sys,
                             std::shared_ptr<DynamicCarEnvironment> env,
                             VectorXd &start_state, VectorXd &goal_state) {
  assert(start_state.size() == env->get_state_space()->getDimension());
  assert(goal_state.size() == env->get_action_space()->getDimension());

  //oc::SimpleSetup ss(env->get_action_space());
  auto lqr_action_space = std::make_shared<LQRControlSpace>(env->get_state_space(), 2);
  ob::RealVectorBounds bounds(2);
  bounds.low[0] = -0.5;
  bounds.high[0] = 0.5;
  bounds.low[1] = -M_PI * 2.0 / 180.0;
  bounds.high[1] = M_PI * 2.0 / 180.0;
  lqr_action_space->setBounds(bounds);
  lqr_action_space->setup();

  oc::SimpleSetup ss(lqr_action_space);
  auto si = ss.getSpaceInformation();

  si->getStateSpace()->registerDefaultProjection(
      std::make_shared<DynamicCarProjector>(si->getStateSpace().get()));

  si->setStateValidityChecker([si](const ob::State *state) {
    // TODO Make real validity checker with obstacle geometry
    return si->satisfiesBounds(state);
  });

  auto odeFn = [&](const oc::ODESolver::StateType &q, const oc::Control *control,
                 oc::ODESolver::StateType &qdot) {
    VectorXd u(2);
    lqr_action_space->compute_u_star(control, q, u);

    oc::Control *computed_lqr_control = env->get_action_space()->allocControl();

    computed_lqr_control->as<oc::RealVectorControlSpace::ControlType>()
        ->values[0] = u[0];
    computed_lqr_control->as<oc::RealVectorControlSpace::ControlType>()
        ->values[1] = u[1];

    sys->ompl_ode_adaptor(q, computed_lqr_control, qdot);

    env->get_action_space()->freeControl(computed_lqr_control);
  };

  auto linearizationFn = [&](const oc::ODESolver::StateType &q,
                             const oc::ODESolver::StateType &, Ref<MatrixXd> A,
                             Ref<MatrixXd> B) {
    sys->get_continuous_time_linearization(q, A, B);
  };

  // Make it clear that KinCarSystem is discrete-time
  //si->setStatePropagator(oc::ODESolver::getStatePropagator(
  //    odeSolver, DynamicCarSystem::ompl_post_integration));
  si->setStatePropagator(std::make_shared<LQRStatePropagator>(
      si, odeFn, linearizationFn, DynamicCarSystem::ompl_post_integration));

  ob::ScopedState<ob::CompoundStateSpace> start(env->get_state_space());
  start->as<ob::SE2StateSpace::StateType>(0)->setX(start_state[0]);
  start->as<ob::SE2StateSpace::StateType>(0)->setY(start_state[1]);
  start->as<ob::SE2StateSpace::StateType>(0)->setYaw(start_state[2]);
  start->as<ob::RealVectorStateSpace::StateType>(1)->values[0] = start_state[3];
  start->as<ob::RealVectorStateSpace::StateType>(1)->values[1] = start_state[4];

  //ob::ScopedState<ob::CompoundStateSpace> goal(env->get_state_space());
  //goal->as<ob::SE2StateSpace::StateType>(0)->setX(goal_state[0]);
  //goal->as<ob::SE2StateSpace::StateType>(0)->setY(goal_state[1]);
  //goal->as<ob::SE2StateSpace::StateType>(0)->setYaw(goal_state[2]);
  //goal->as<ob::RealVectorStateSpace::StateType>(1)->values[0] = goal_state[3];
  //goal->as<ob::RealVectorStateSpace::StateType>(1)->values[1] = goal_state[4];

  //ss.setStartAndGoalStates(start, goal, 0.1);
  ss.setStartState(start);
  Vector3d geom_goal(2);
  geom_goal << goal_state[0], goal_state[1];
  ss.setGoal(std::make_shared<SubsetGoalRegion>(ss.getSpaceInformation(), geom_goal));

  // Necessary for discrete-time model
  ss.getSpaceInformation()->setPropagationStepSize(env->get_time_step() * 10);
  log_info("Propagation step size is",
           ss.getSpaceInformation()->getPropagationStepSize());

  //auto planner = std::make_shared<oc::SST>(ss.getSpaceInformation());
  auto planner = std::make_shared<oc::RRT>(ss.getSpaceInformation());
  ss.setPlanner(planner);
  ss.setup();
  //ss.print();

  double plan_time = 1.0;
  bool have_plan = false;

  while (!have_plan && plan_time < 60.0) {

    log_info("Planning for", plan_time, "seconds");
    ob::PlannerStatus solved = ss.solve(plan_time);
    if (solved) {
      if (ss.haveExactSolutionPath()) {
        std::cout << "Found solution:" << std::endl;
        //ss.getSolutionPath().printAsMatrix(std::cout);

        // TODO Handle planned LQR controls 
        //return ss.getSolutionPath();

        auto vector_si = std::make_shared<oc::SpaceInformation>(
            env->get_state_space(), env->get_action_space());

        log_info("Before converting to vector controls, path is");
        ss.getSolutionPath().printAsMatrix(std::cout);

        return get_vector_control_path(
            sys, lqr_action_space, vector_si, ss.getSolutionPath(),
            ss.getSpaceInformation()->getPropagationStepSize());
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

  // TODO Record closest approach to goal at each timestep up to max number of
  // timesteps
  LQRExecutor executor(env, goal.head(2), tracking_threshold, learn);

  while ((env->get_state().head(2) - goal.head(2)).norm() > 0.1 &&
         executor.get_overall_timestep() <
             executor.get_max_overall_timestep()) {
    start = env->get_state();
    // auto path = plan_to_goal(nominal_sys, env, start, goal);

    auto path = plan_to_goal(planning_sys, env, start, goal);
    path.printAsMatrix(std::cout);
    log_info("Path has length", path.length());
    if (path.getControlCount() == 0) {
      log_info("No plan found");
      break;
    }

    // TODO Adjust bounds on action space? State space?
    w.start_log("planned_traj");
    w.write_line("planned_traj",
                 "timestep,statex,statey,stateth,statev,statedth,controla,controlth");

    w.start_log("executed_traj");
    w.write_line("executed_traj",
                 "timestep,statex,statey,stateth,statev,statedth,controla,controlth");

    auto parl = executor.execute_path(nominal_sys, path, w, seed);

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
