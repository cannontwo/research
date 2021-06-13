#include <cannon/research/parl_planning/lqr_executor.hpp>

#include <cassert>

#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl_planning/ompl_utils.hpp>
#include <cannon/research/parl_planning/lqr_control_space.hpp>
#include <cannon/research/parl/environment.hpp>
#include <cannon/physics/systems/system.hpp>
#include <cannon/utils/experiment_writer.hpp>
#include <cannon/log/registry.hpp>

using namespace cannon::research::parl;
using namespace cannon::utils;
using namespace cannon::log;
using namespace cannon::physics::systems;

std::shared_ptr<Parl>
LQRExecutor::execute_path(std::shared_ptr<System> nominal_sys,
                          oc::PathControl &path, ExperimentWriter &w,
                          int seed) {
  auto parl = construct_parl_agent_for_path(nominal_sys, path, seed);

  // Execute PARL controller for max_overall_timesteps_ or until path diverges
  // enough from plan
  while (overall_timestep_ < max_overall_timestep_) {
    VectorXd state = env_->get_state();
    VectorXd u = parl->get_action(state);
    write_executed_traj_line_(w, u);

    double env_reward;
    bool done;
    VectorXd new_state;
    std::tie(new_state, env_reward, done) = env_->step(u);

    write_distances_line_(w, new_state);

    // TODO Calculate interpolated waypoint
    //write_planned_traj_line_(w, interp_waypoint, u);

    // TODO Calculate reward driving toward next waypoint (piecewise reward
    // function along path)
    double reward = 0.0;

    if (learn_) {
      parl->process_datum(state, u, reward, new_state, false);

      // TODO Do controller updates on some interval instead
      parl->value_grad_update_controller(state);
    }

    if (render_) {
      env_->render();
      env_->register_ep_reward(reward);
    }

    // TODO How to compute path distance here? Frechet distance? Check against
    // interpolated waypoints? When do we trigger replanning?

    overall_timestep_ += 1;
  }

  return parl;
}

int LQRExecutor::get_overall_timestep() {
  return overall_timestep_;
}

int LQRExecutor::get_max_overall_timestep() {
  return max_overall_timestep_;
}

VectorXd LQRExecutor::interp_waypoints(const Ref<const VectorXd> &w0,
                                       const Ref<const VectorXd> &w1) {
  assert(w0.size() == env_->get_state_space()->getDimension());
  assert(w0.size() == w1.size());
  return w0 + 0.5 * (w1 - w0);
}

MatrixXd LQRExecutor::make_path_refs(oc::PathControl& path) {
  std::vector<ob::State *> states = path.getStates();
  MatrixXd refs = MatrixXd::Zero(env_->get_state_space()->getDimension(),
                                 states.size() - 1);

  for (unsigned int i = 0; i < states.size() - 1; i++) {
    auto w0 = get_coords_from_ompl_state(env_, states[i]);
    auto w1 = get_coords_from_ompl_state(env_, states[i+1]);

    refs.row(i) = interp_waypoints(w0, w1);
  }

  return refs;
}

ParlPtr LQRExecutor::construct_parl_agent_for_path(
    physics::systems::SystemPtr nominal_sys, oc::PathControl &path, int seed) {
  std::vector<ob::State *> states = path.getStates();
  std::vector<oc::Control *> controls = path.getControls();
  std::vector<double> durations = path.getControlDurations();
  assert(durations.size() == controls.size());
  
  // Construct PARL with interpolated waypoints
  // TODO May want to make which config to load a parameter for the executor
  Hyperparams params;
  params.load_config("/home/cannon/Documents/cannon/cannon/research/"
                     "experiments/parl_configs/r10c10_dc.yaml");

  auto refs = make_path_refs(path);
  auto parl = std::make_shared<Parl>(
      env_->get_state_space(), env_->get_action_space(), refs, params, seed);

  for (unsigned int i = 0; i < controls.size(); i++) {
    // Initialize PARL controllers with path LQR controls
    auto lqr_control = static_cast<LQRControlSpace::ControlType*>(controls[i]);
    parl->set_K_matrix_idx_(i, -lqr_control->K);
    parl->set_k_matrix_idx_(i, lqr_control->K * lqr_control->q0);
    // TODO Also initialize number of datapoints contributing to controller?

    // Initialize PARL dynamics with environment linearization
    auto w0 = get_coords_from_ompl_state(env_, states[i]);
    auto w1 = get_coords_from_ompl_state(env_, states[i+1]);
    VectorXd x = interp_waypoints(w0, w1);

    MatrixXd A, B;
    VectorXd c;
    std::tie(A, B, c) = nominal_sys->get_linearization(x);

    // Assuming that mean of input data would be the interpolated waypoint and
    // zero control
    VectorXd in_mean = VectorXd::Zero(env_->get_state_space()->getDimension() +
                                      env_->get_action_space()->getDimension());
    in_mean.head(x.size()) = x;
    parl->set_dynamics_idx_(i, A, B, c, in_mean); 
    // TODO Also initialize number of datapoints contributing to dynamics?
  }

  return parl;
}

void LQRExecutor::write_executed_traj_line_(ExperimentWriter &w, const VectorXd& c) {
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

void LQRExecutor::write_planned_traj_line_(ExperimentWriter &w, const VectorXd&
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

void LQRExecutor::write_distances_line_(ExperimentWriter &w, const VectorXd& s) {
  assert(s.size() == env_->get_state_space()->getDimension());

  std::stringstream ss;
  ss << overall_timestep_ << "," << (s.head(goal_.size()) - goal_).norm();
  w.write_line("distances", ss.str());
}
