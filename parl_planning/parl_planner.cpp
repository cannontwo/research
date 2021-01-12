#include <cannon/research/parl_planning/parl_planner.hpp>

using namespace cannon::research::parl;

oc::PathControl ParlPlanner::plan_to_goal(std::shared_ptr<Environment> env,
    post_int_func ompl_post_integration, const ob::ScopedState<ob::StateSpace>&
    start, const ob::ScopedState<ob::StateSpace>& goal) {

  // TODO Before anything else, process current PARL model into aggregate model

  oc::SimpleSetup ss(env->get_action_space());
  auto si = ss.getSpaceInformation();

  // TODO Actually handle invalid states
  si->setStateValidityChecker([&](const ob::State *state) {
        return true;
      });

  auto sys = env->get_ode_sys();
  auto odeSolver(std::make_shared<oc::ODEBasicSolver<>>(si, 
        [&](const oc::ODESolver::StateType& q, 
            const oc::Control* control, oc::ODESolver::StateType& qdot){
          sys->ompl_ode_adaptor(q, control, qdot);
        }));
  si->setStatePropagator(oc::ODESolver::getStatePropagator(odeSolver,
        ompl_post_integration));

  ss.setStartAndGoalStates(start, goal, 0.05);
  ss.setPlanner(planner_);

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

// Free Functions

std::shared_ptr<ob::StateSpace> cannon::research::parl::make_error_state_space(const
    std::shared_ptr<Environment> env, double duration) {
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

MatrixXd cannon::research::parl::make_error_system_refs(oc::PathControl& path, unsigned int state_dim) {
  auto times = get_ref_point_times(path);

  // Since this is the error system, all dimensions except for time should be zero
  MatrixXd refs = MatrixXd::Zero(state_dim + 1, times.size());
  for (unsigned int i = 0; i < times.size(); i++) {
    refs(state_dim, i) = times[i];
  }

  return refs;
}

std::vector<double> cannon::research::parl::get_ref_point_times(oc::PathControl& path) {
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

VectorXd cannon::research::parl::compute_error_state(const VectorXd& ref, const
    VectorXd& actual, double time) {
  if (ref.size() != actual.size())
    throw std::runtime_error("Ref state and actual state had different dimensions");

  VectorXd diff = ref - actual;
  VectorXd ret(diff.size() + 1);
  ret.head(diff.size()) = diff;
  ret[diff.size()] = time;

  return ret;
}
