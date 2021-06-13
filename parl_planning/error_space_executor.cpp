#include <cannon/research/parl_planning/error_space_executor.hpp>

#include <cassert>

#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl_planning/ompl_utils.hpp>
#include <cannon/research/parl/environment.hpp>
#include <cannon/utils/experiment_writer.hpp>
#include <cannon/log/registry.hpp>

using namespace cannon::research::parl;
using namespace cannon::utils;
using namespace cannon::log;

void ErrorSpaceExecutor::execute_path( oc::PathControl &path, std::shared_ptr<Parl> parl,
    ExperimentWriter &w) {
  std::vector<ob::State *> states = path.getStates();
  std::vector<oc::Control *> controls = path.getControls();
  std::vector<double> durations = path.getControlDurations();
  assert(durations.size() == controls.size());

  double accumulated_dur = 0.0;
  for (unsigned int i = 0; i < controls.size(); i++) {
    VectorXd s = get_coords_from_ompl_state(env_, states[i]);
    VectorXd next_s = s;

    if (i < controls.size() - 1)
      next_s = get_coords_from_ompl_state(env_, states[i+1]);

    // Break path tracking if state error greater than threshold. This is a
    // check at each waypoint.
    double path_error = (s - env_->get_state()).norm();
    log_info("At step", i, "xy path error is", path_error);
    if (path_error > tracking_threshold_) {
      // TODO This implicitly assumes that there is no autonomous dynamical
      // behavior; instead, a stopping maneuver should be executed before
      // replanning (or, under different assumptions, replanning should be done
      // for the predicted state after planning time).

      log_info("Replanning because tracking threshold exceeded");
      return;
    }

    if (overall_timestep_ > max_overall_timestep_) 
      return;

    Vector2d c = get_control_from_ompl_control(env_, controls[i]);

    // log_info("Executing control", c, "for", control_dur.count(),
    // "milliseconds");

    execute_control_for_duration(parl, s, next_s, c, accumulated_dur,
                                 durations[i], w);
    accumulated_dur += durations[i];
  }

  log_info("Goal error on real system is",
           (env_->get_state().head(goal_.size()) - goal_).norm());
}

void ErrorSpaceExecutor::execute_control_for_duration(ParlPtr parl, const VectorXd &s,
    const VectorXd &next_s, const VectorXd &c, double start_time,
    double control_dur, ExperimentWriter &w) {

  assert(s.size() == env_->get_state_space()->getDimension());
  assert(next_s.size() == env_->get_state_space()->getDimension());

  double cur_dur = 0.0;
  double total_seg_reward = 0.0;
  std::vector<VectorXd> states;
  do {
    //log_info("Executing control for duration", env_->get_time_step(), "at overall timestep", overall_timestep_); 

    // Compute interpolated reference and error state
    VectorXd old_interp_ref, error_state;
    std::tie(old_interp_ref, error_state) = compute_interpolated_error_state_(
        cur_dur / control_dur, s, next_s, start_time + cur_dur);

    // Compute PARL control for error state
    VectorXd parl_c = VectorXd::Zero(env_->get_action_space()->getDimension());
    if (learn_) {
      parl_c = parl->get_action(error_state);
    }

    // Step the environment
    VectorXd new_state = execute_timestep(parl_c + c, w);

    // TODO Check for ICS? Or somehow else execute emergency stopping maneuver?

    write_distances_line_(w, next_s);

    // Compute error coordinates for new_state
    VectorXd interp_ref, new_error_state;
    std::tie(interp_ref, new_error_state) = compute_interpolated_error_state_(
        (cur_dur + env_->get_time_step()) / control_dur, s, next_s,
        cur_dur + start_time + env_->get_time_step());

    write_planned_traj_line_(w, interp_ref, c);

    // Compute trajectory following reward
    // TODO Frechet distance?
    double reward = -(interp_ref - new_state).norm();

    // Train PARL using error states
    if (learn_) {
      parl->process_datum(error_state, parl_c, reward, new_error_state, false);
      states.push_back(error_state);
    }

    total_seg_reward += reward;

    cur_dur += env_->get_time_step();
    overall_timestep_++;

  } while (control_dur > cur_dur);

  if (learn_) {
    for (auto state : states) {
      parl->value_grad_update_controller(state);
    }
  }

  // Plotting reward for an entire control segment
  if (render_)
    env_->register_ep_reward(total_seg_reward);
}

VectorXd ErrorSpaceExecutor::execute_timestep(const VectorXd &c, ExperimentWriter &w) {
  write_executed_traj_line_(w, c);

  double env_reward;
  bool done;
  VectorXd new_state;
  std::tie(new_state, env_reward, done) = env_->step(c);
  if (render_)
    env_->render();

  return new_state;
}

int ErrorSpaceExecutor::get_overall_timestep() {
  return overall_timestep_;
}

int ErrorSpaceExecutor::get_max_overall_timestep() {
  return max_overall_timestep_;
}

VectorXd ErrorSpaceExecutor::compute_error_state_(const VectorXd &ref, const VectorXd
    &actual, double time) {
  unsigned int state_dim = env_->get_state_space()->getDimension();

  assert(ref.size() == state_dim);
  assert(actual.size() == state_dim);

  VectorXd diff = ref - actual;
  VectorXd ret = VectorXd::Zero(state_dim + 1);
  ret.head(state_dim) = diff;
  ret[state_dim] = time;

  return ret;
}

std::pair<VectorXd, VectorXd> ErrorSpaceExecutor::compute_interpolated_error_state_(
    double t, const VectorXd &s, const VectorXd &next_s, double time) {
  assert(0.0 <= t);
  assert(t <= 1.0);

  VectorXd interp_ref = ((1.0 - t) * s) + (t * next_s);
  VectorXd error_state = compute_error_state_(interp_ref, env_->get_state(), time);

  return std::make_pair(interp_ref, error_state);
}

void ErrorSpaceExecutor::write_executed_traj_line_(ExperimentWriter &w, const VectorXd& c) {
  assert(c.size() == env_->get_action_space()->getDimension());

  std::stringstream ss;
  ss << overall_timestep_ << ",";

  auto s = env_->get_state();
  for (unsigned int i = 0; i < env_->get_state_space()->getDimension(); i++) {
    ss << s[i] << ",";
  }

  for (unsigned int i = 0; i < c.size(); i++)  {
    ss << c[i] << ",";
  }

  w.write_line("executed_traj", ss.str());
}

void ErrorSpaceExecutor::write_planned_traj_line_(ExperimentWriter &w, const VectorXd&
    interp_ref, const VectorXd& c) {
  assert(s.size() == env_->get_state_space()->getDimension());
  assert(c.size() == env_->get_action_space()->getDimension());

  std::stringstream ss;
  ss << overall_timestep_ << ",";

  for (unsigned int i = 0; i < interp_ref.size(); i++) {
    ss << interp_ref[i] << ",";
  }

  for (unsigned int i = 0; i < c.size(); i++)  {
    ss << c[i] << ",";
  }

  w.write_line("planned_traj", ss.str());
}

void ErrorSpaceExecutor::write_distances_line_(ExperimentWriter &w, const VectorXd& s) {
  assert(s.size() == env_->get_state_space()->getDimension());

  std::stringstream ss;
  ss << overall_timestep_ << "," << (s.head(goal_.size()) - goal_).norm();
  w.write_line("distances", ss.str());
}
