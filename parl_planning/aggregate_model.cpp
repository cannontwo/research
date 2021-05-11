#include <cannon/research/parl_planning/aggregate_model.hpp>

using namespace cannon::research::parl;

void AggregateModel::operator()(const VectorXd& s, VectorXd& dsdt, const double t) {

  (*nominal_model_)(s, dsdt, t);

  assert(s.size() == state_dim_ + action_dim_);

  const VectorXd x = s.head(state_dim_);
  const VectorXd u = s.tail(action_dim_);

  // Get appropriate local learned model
  LinearParams learned_params = get_local_model_for_state(x);

  // Since we return a zero-initialized LinearParams if the model is not found
  // in parameters_, this works correctly when we have no data.
  
  // TODO May want to reweight this based on uncertainty in model
  VectorXd learned_part = (learned_params.A_ * x) + (learned_params.B_ * u) + learned_params.c_;

  // Adjust for the fact that the learned model is discrete-time (with
  // time step = time_delta_) so that integration works correctly
  //dsdt += learned_part / time_delta_;
  if (learn_) 
    dsdt.head(state_dim_) += learned_part;
}

void AggregateModel::ompl_ode_adaptor(const oc::ODESolver::StateType& q,
    const oc::Control* control, oc::ODESolver::StateType& qdot) {
  VectorXd s = VectorXd::Zero(state_dim_ + action_dim_);
  VectorXd dsdt = VectorXd::Zero(state_dim_ + action_dim_);

  assert(q.size() == state_dim_);

  // Fill current state into s from q and control
  for (unsigned int i = 0; i < state_dim_; i++) {
    s[i] = q[i];
  }

  // This will error if this object was initialized with the wrong action_dim,
  // but there's no way to check the input Control* here
  for (unsigned int i = 0; i < action_dim_; i++) {
    s[state_dim_ + i] = control->as<oc::RealVectorControlSpace::ControlType>()->values[i];
  }
  
  (*this)(s, dsdt, 0.0);

  qdot.resize(q.size(), 0);
  for (unsigned int i = 0; i < q.size(); i++) {
    qdot[i] = dsdt[i];
  }
}

std::tuple<MatrixXd, MatrixXd, VectorXd> AggregateModel::get_linearization(const VectorXd& x) {
  MatrixXd A = MatrixXd::Zero(state_dim_, state_dim_);
  MatrixXd B = MatrixXd::Zero(state_dim_, action_dim_);
  VectorXd c = VectorXd::Zero(action_dim_);

  MatrixXd nA, nB;
  VectorXd nc;
  std::tie(nA, nB, nc) = nominal_model_->get_linearization(x);

  LinearParams learned_params = get_local_model_for_state(x);

  A = learned_params.A_ + nA;
  B = learned_params.B_ + nB;
  c = learned_params.c_ + nc;

  return std::make_tuple(A, B, c);
}

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

  if (model.get_num_data() == 0) {
    return;
  }

  MatrixXd theta = model.get_identified_mats().first;
  assert(theta.rows() == state_dim_ + 1); // Account for time dimension
  assert(theta.cols() == state_dim_ + action_dim_ + 1);

  VectorXd c_hat = model.get_identified_mats().second.head(state_dim_);
  assert(c_hat.size() == state_dim_);

  MatrixXd A_hat = theta.leftCols(state_dim_).topRows(state_dim_);
  MatrixXd B_hat = theta.rightCols(action_dim_).topRows(state_dim_);

  // See https://www.overleaf.com/project/5fff4fe3176331cc9ab8472d
  VectorXd interp_state = ref_state + tau * (next_ref_state - ref_state);
  VectorXd next_interp_state = interp_state + tau_delta * (next_ref_state - ref_state);
  VectorXd c_est = next_interp_state - (A_hat * interp_state) - (B_hat * ref_control) - c_hat;
  
  //log_info("Adding", model.get_num_data(), "datum model with A_hat", A_hat, ", B_hat ", B_hat, ", c_est", c_est);
  
  MatrixXd A, B;
  VectorXd c;
  std::tie(A, B, c) = nominal_model_->get_linearization(interp_state);

  // Add params that is only delta between learned model and nominal model linearization
  LinearParams tmp_param(A_hat - A, -B_hat - B, c_est - c, model.get_num_data());

  VectorXu grid_coords = get_grid_coords(ref_state);
  auto iter = parameters_.find(grid_coords);
  if (iter != parameters_.end()) {
    // There is already a LinearParams, so we merge
    iter->second.merge(tmp_param);
  } else {
    // There is no existing LinearParams object for grid_coords
    parameters_.insert({grid_coords, tmp_param});
  }
}

void AggregateModel::process_path_parl(std::shared_ptr<Environment> env,
    std::shared_ptr<Parl> model, oc::PathControl& path) {

  std::vector<ob::State*> states = path.getStates();
  std::vector<oc::Control*> controls = path.getControls();
  std::vector<double> durations = path.getControlDurations();

  double accumulated_time = 0.0;

  for (unsigned int i = 0; i < controls.size(); i++) {
    VectorXd c = get_control_from_ompl_control(env, controls[i]);

    ob::ScopedState<> w_state (path.getSpaceInformation()->getStateSpace());
    w_state = states[i];
    VectorXd waypoint = VectorXd::Zero(state_dim_);
    for (unsigned int j = 0; j < state_dim_; j++) { 
      waypoint[j] = w_state.reals()[j];
    }

    VectorXd next_waypoint = waypoint;
    if (i < controls.size() - 1) {
      ob::ScopedState<> wn_state (path.getSpaceInformation()->getStateSpace());
      wn_state = states[i+1];
      for (unsigned int j = 0; j < state_dim_; j++) { 
        next_waypoint[j] = w_state.reals()[j];
      }
    }

    // Get local model in terms of path time
    accumulated_time += 0.5 * durations[i];
    VectorXd query = VectorXd::Zero(state_dim_ + 1);
    query[state_dim_] = accumulated_time;
    accumulated_time += 0.5 * durations[i];

    unsigned int ref_idx = model->get_nearest_ref_idx(query);
    auto local_model = model->dynamics_models_[ref_idx];
    
    // Assuming that our data have mean that is halfway along this path
    // segment. This may not be a good assumption.
    double tau = 0.5;
    double tau_delta = time_delta_ / durations[i];

    //log_info("Adding local model for ref", i, "/", controls.size());
    add_local_model(local_model, waypoint, next_waypoint, c, tau, tau_delta);
  }

}

VectorXu AggregateModel::get_grid_coords(const VectorXd& query) const {
  assert(query.size() == state_dim_);

  //log_info("Query is", query);

  VectorXu coords(state_dim_);

  for (unsigned int i = 0; i < state_dim_; i++) {
    coords[i] = (unsigned int)floor((query[i] - bounds_(i, 0)) / cell_extent_[i]);

    if (coords[i] >= grid_size_)
      throw std::runtime_error("Passed query point outside state bounds");
  }

  return coords; 
}

LinearParams AggregateModel::get_local_model_for_state(const VectorXd& state) {
  try {
    VectorXu grid_coords = get_grid_coords(state);

    auto iter = parameters_.find(grid_coords);
    if (iter != parameters_.end()) {
      return LinearParams(iter->second.A_, iter->second.B_, iter->second.c_,
          iter->second.num_data_);
    } else {
      return LinearParams(state_dim_, action_dim_);
    }
  } catch (...) {
    // We can throw an exception if the input state is out of bounds, but in
    // this case we just return zero dynamics
    return LinearParams(state_dim_, action_dim_);
  }
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
