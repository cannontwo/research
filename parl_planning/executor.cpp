#include <cannon/research/parl_planning/executor.hpp>

using namespace cannon::research::parl;

void Executor::execute(post_int_func ompl_post_integration) {
  // TODO This is the main function
  
  while ((env_->get_state() - goal_).norm() > 0.1) {
    VectorXd start = env_->get_state();
    // copy start and goal to OMPL ScopedState for use with planner
    ob::ScopedState<> ompl_start(env_->get_state_space());
    ob::ScopedState<> ompl_goal(env_->get_state_space());

    for (unsigned int i = 0; i < start.size(); i++) {
      ompl_start[i] = start[i];
      ompl_goal[i] = goal_[i];
    }

    auto path = planner_->plan_to_goal(env_, ompl_post_integration, ompl_start, ompl_goal);

    // TODO Do we want to adjust bounds on action space?
    Hyperparams params;
    params.load_config("/home/cannon/Documents/cannon/cannon/research/experiments/parl_configs/r10c10_kc.yaml"); 
    Parl parl(make_error_state_space(env_, path.length()), env_->get_action_space(),
        make_error_system_refs(path, start.size()), params);

    execute_path(path);

    // TODO Update model used by planner to incorporate learned dynamics
    // This could just be a decorator on the original model that looks up the
    // learned PARL model for a particular region, but the learned PARL model
    // needs to be processed after the path is executed.
  }
}

void Executor::execute_path(oc::PathControl& path) {
  std::vector<ob::State*> states = path.getStates();
  std::vector<oc::Control*> controls = path.getControls();
  std::vector<double> durations = path.getControlDurations();
  assert(durations.size() == controls.size());

  double accumulated_dur = 0.0;
  for (unsigned int i = 0; i < controls.size(); i++) {
    VectorXd s = get_coords_from_ompl_state(states[i]);
    VectorXd next_s = get_coords_from_ompl_state(states[i+1]);

    // Break path tracking if state error greater than threshold
    double path_error = (s - env_->get_state()).norm();
    log_info("At step", i, "xy path error is", path_error);
    if (path_error > tracking_threshold_) {
      // TODO This implicitly assumes that there is no autonomous dynamical
      // behavior; instead, a stopping maneuver should be executed before
      // replanning (or, under different assumptions, replanning should be done
      // for the predicted state after planning time). Ties into ICS discussion.
      
      log_info("Replanning because tracking threshold exceeded");
      return;
    }

    Vector2d c;
    c[0] = controls[i]->as<oc::RealVectorControlSpace::ControlType>()->values[0];
    c[1] = controls[i]->as<oc::RealVectorControlSpace::ControlType>()->values[1];

    unsigned int duration_millis = std::floor(durations[i] * 1000);
    std::chrono::milliseconds control_dur(duration_millis);
    
    execute_control_for_duration(s, next_s, c, accumulated_dur, control_dur);
    accumulated_dur += durations[i];
  }

  log_info("Goal error on real system is", (env_->get_state() - goal_).norm());
  
}

void Executor::execute_control_for_duration(const VectorXd& s, const VectorXd& next_s, 
    const VectorXd& c, double start_time, std::chrono::milliseconds control_dur) {

  assert(s.size() == next_s.size());
  assert(s.size() == env_->get_state_space()->getDimension());
  assert(c.size() == env_->get_action_space()->getDimension());

  auto start = std::chrono::steady_clock::now();
  std::chrono::milliseconds cur_dur;

  VectorXd error_state = compute_error_state(s, env_->get_state(), 0.0);

  VectorXd new_state;
  double reward;
  bool done;
  do {
    VectorXd parl_c = planner_->parl_->get_action(error_state);

    std::tie(new_state, reward, done) = env_->step(c + parl_c);
    env_->render();

    // TODO Check for ICS? Or somehow else execute emergency stopping maneuver?

    // Compute error coordinates for new_state
    double t = (float)cur_dur.count() / (float)control_dur.count();

    // TODO Linear interpolation between reference states may not be the best
    // way to do this for strongly nonlinear systems.
    VectorXd interp_ref = ((1.0 - t) * s) + (t * next_s);
    double time = ((double)cur_dur.count() / 1000.0) + start_time;
    VectorXd new_error_state = compute_error_state(interp_ref, new_state, time);

    // Train PARL using error states
    planner_->parl_->process_datum(error_state, parl_c, reward, new_error_state, done);

    error_state = new_error_state;

    cur_dur = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
  } while (control_dur > cur_dur);

}

VectorXd Executor::get_coords_from_ompl_state(ob::State* s) {
  std::vector<double> coord_vec;

  env_->get_state_space()->copyToReals(coord_vec, s);

  VectorXd ret_vec(coord_vec.size());
  ret_vec = Map<VectorXd, Unaligned>(coord_vec.data(), coord_vec.size());

  return ret_vec;
}

void cannon::research::parl::noop_post_integration(const ob::State*, const oc::Control*, 
    const double, ob::State*) {
  // This is a no-op
}
