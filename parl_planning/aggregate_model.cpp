#include <cannon/research/parl_planning/aggregate_model.hpp>

using namespace cannon::research::parl;

void AggregateModel::add_local_model(const RLSFilter& model, const VectorXd&
    ref_state, const VectorXd& next_ref_state, const VectorXd& ref_control,
    double tau, double tau_delta) {
  
  assert(ref_state.size() == state_dim_);
  assert(next_ref_state.size() == state_dim_);
  assert(ref_control.size() == action_dim_);
  assert(tau >= 0.0);
  assert(tau <= 1.0);
  assert(tau_delta >= 0.0);
  assert(tau_delta <= 1.0);

  MatrixXd theta = model.get_identified_mats().first;
  assert(theta.rows() == state_dim_);
  assert(theta.cols() == state_dim_ + action_dim_);

  VectorXd c_hat = model.get_identified_mats().second;
  assert(c_hat.size() == state_dim_);

  MatrixXd A_hat = theta.leftCols(state_dim_);
  MatrixXd B_hat = theta.rightCols(action_dim_);

  // See https://www.overleaf.com/project/5fff4fe3176331cc9ab8472d
  VectorXd interp_state = ref_state + tau * (next_ref_state - ref_state);
  VectorXd next_interp_state = interp_state + tau_delta * (next_ref_state - ref_state);
  VectorXd c_est = next_interp_state - (A_hat * interp_state) - (B_hat * ref_control) - c_hat;

  LinearParams tmp_param(A_hat, -B_hat, c_est, model.get_num_data());

  VectorXu grid_coords = get_grid_coords(ref_state);
  auto iter = parameters_.find(grid_coords);
  if (iter != parameters_.end()) {
    // There is no existing LinearParams object for grid_coords
    parameters_.insert({grid_coords, tmp_param});
  } else {
    // There is already a LinearParams, so we merge
    iter->second.merge(tmp_param);
  }
}

void AggregateModel::process_path_parl(std::shared_ptr<Environment> env,
    std::shared_ptr<Parl> model, oc::PathControl& path) {

  std::vector<ob::State*> states = path.getStates();
  std::vector<oc::Control*> controls = path.getControls();
  std::vector<double> durations = path.getControlDurations();

  for (unsigned int i = 0; i < controls.size(); i++) {
    VectorXd c = get_control_from_ompl_control(env, controls[i]);
    
    // TODO Finding appropriate waypoints, locating local parl model, computing tau/tau_delta
    //add_local_model(local_model, waypoint, next_waypoint, c, tau, tau_delta);
  }

}

VectorXu AggregateModel::get_grid_coords(const VectorXd& query) const {
  assert(query.size() == state_dim_);

  VectorXu coords(state_dim_);

  for (unsigned int i = 0; i < state_dim_; i++) {
    coords[i] = (unsigned int)floor(query[i] / cell_extent_[i]);

    if (coords[i] >= grid_size_)
      throw std::runtime_error("Passed query point outside state bounds");
  }

  return coords; 
}

// Free Functions
bool cannon::research::parl::vector_comp(const VectorXu& v1, const VectorXu& v2) {
  assert(v1.size() == v2.size());

  for (unsigned int i = 0; i < v1.size(); i++) {
    if (v1[i] == v2[i])
      continue;
    else 
      return v1[i] < v2[i];
  }

  return false;
}
