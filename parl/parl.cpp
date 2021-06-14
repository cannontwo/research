#include <cannon/research/parl/parl.hpp>

#include <stdexcept>
#include <cassert>
#include <random>

#include <cannon/geom/kd_tree_indexed.hpp>
#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/math/multivariate_normal.hpp>
#include <cannon/research/parl/linear_params.hpp>

using namespace cannon::research::parl;
using namespace cannon::geom;
using namespace cannon::math;

// Public methods

Parl::Parl(const std::shared_ptr<ob::StateSpace> state_space,
           const std::shared_ptr<oc::RealVectorControlSpace> action_space,
           const MatrixXd &refs, Hyperparams params, int seed,
           bool stability)
    : state_dim_(state_space->getDimension()),
      action_dim_(action_space->getDimension()), state_space_(state_space),
      action_space_(action_space), seed_(seed), refs_(refs), params_(params),
      value_model_(state_dim_, refs.cols(), params.discount_factor),
      ref_tree_(new KDTreeIndexed(state_dim_)), stability_(stability) {

  if (refs_.rows() != state_dim_)
    throw std::runtime_error("PARL reference points have the wrong dimension");
  num_refs_ = refs_.cols();

  if (seed_ != 0) {
    // Set seed across all instances of multivariate normal distribution
    MultivariateNormal::set_seed(seed_);
  }

  if (params_.use_line_search && params_.controller_learning_rate != 1.0) {
    throw std::runtime_error("When using line search, learning rate should be 1");
  }
  
  
  ref_tree_->insert(refs_);

  for (int i = 0; i < num_refs_; i++) {
    // The dynamics models predict on (state, action) -> state
    dynamics_models_.emplace_back(state_dim_ + action_dim_,
        state_dim_, params_.alpha, params_.forgetting_factor);

    controllers_.emplace_back(state_dim_, action_dim_,
        params_.controller_learning_rate, params_.use_adam);

    // Checking KDT construction
    assert(ref_tree_->get_nearest_idx(refs_.col(i)) == i);
  }

  VectorXd zero_vec = VectorXd::Zero(state_dim_);

  // Find indices of regions containing zero
  if (stability_) {
    auto diagram = compute_voronoi_diagram(refs_);
    auto polys = create_bounded_voronoi_polygons(refs_, diagram);

    for (unsigned int i = 0; i < refs.size(); i++) {
      if (is_inside(Vector2d::Zero(), polys[i])) {
        zero_ref_idxs_.push_back(i);
      }
    }

    log_info("References whose Voronoi regions cover (0, 0):");
    for (auto idx : zero_ref_idxs_) {
      log_info("\t (", idx, "):", refs.col(idx));
    }
  }
}

void Parl::process_datum(const VectorXd& state, const VectorXd& action,
    double reward, const VectorXd& next_state, bool done,
    bool use_local) {
  check_state_dim_(state);
  check_state_dim_(next_state);
  check_action_dim_(action);

  int idx = ref_tree_->get_nearest_idx(state);
  int next_idx = ref_tree_->get_nearest_idx(next_state);

  if (use_local) {
    VectorXd local_state = make_local_state_(state);
    VectorXd local_comb_vec = make_combined_vec_(local_state, action);
    dynamics_models_[idx].process_datum(local_state, next_state);
  } else {
    VectorXd comb_vec = make_combined_vec_(state, action);
    dynamics_models_[idx].process_datum(comb_vec, next_state);
    value_model_.process_datum(state, next_state, idx, next_idx, reward);
  }
}

void Parl::value_grad_update_controller(const VectorXd& state) {
  check_state_dim_(state);

  int idx = ref_tree_->get_nearest_idx(state);

  VectorXd k_grad;
  MatrixXd K_grad;
  std::tie(k_grad, K_grad) = calculate_approx_value_gradient(state);

  if (params_.use_line_search) {
    VectorXd k;
    MatrixXd K;
    std::tie(k, K) = controllers_[idx].get_mats();

    VectorXd searched_k_grad;
    MatrixXd searched_K_grad;
    std::tie(searched_k_grad, searched_K_grad) = line_search(k_grad, K_grad, state, k, K);
    controllers_[idx].apply_gradient(searched_k_grad, searched_K_grad);
  } else {
    controllers_[idx].apply_gradient(k_grad, K_grad);
  }

  if (stability_ && 
      (std::find(zero_ref_idxs_.begin(), zero_ref_idxs_.end(), idx) != zero_ref_idxs_.end())) {
    // Set controllers_[idx].k_ to be analytic so that stability is enforced around zero
    auto zero_B = get_B_matrix_idx_(idx);
    auto zero_c = dynamics_models_[idx].get_identified_mats().second;
    auto analytic_k = -(zero_B.transpose() * zero_B).inverse() * zero_B.transpose() * zero_c;
    controllers_[idx].set_k(analytic_k);
  }
}

VectorXd Parl::predict_next_state(const VectorXd& state, const VectorXd& action,
    bool use_local) {
  check_state_dim_(state);
  check_action_dim_(action);

  int idx = ref_tree_->get_nearest_idx(state);
  if (use_local) {
    VectorXd local_state = make_local_state_(state);
    VectorXd c = make_combined_vec_(local_state, action);
    return dynamics_models_[idx].predict(c);
  } else {
    VectorXd c = make_combined_vec_(state, action);

    return dynamics_models_[idx].predict(c);
  }
}

double Parl::predict_value(const VectorXd& state, bool use_local) {
  check_state_dim_(state);

  int idx = ref_tree_->get_nearest_idx(state); 
  if (use_local)
    throw std::runtime_error("Not implemented");
  else
    return value_model_.predict(state, idx);
}

VectorXd Parl::compute_value_gradient(const VectorXd& state) {
  check_state_dim_(state);
  VectorXd v_x = get_V_matrix_(state);
  MatrixXd b = get_B_matrix_(state);

  return b.transpose() * v_x;
}

VectorXd Parl::get_unconstrained_action(const VectorXd& state) {
  check_state_dim_(state);

  int idx = ref_tree_->get_nearest_idx(state);
  return controllers_[idx].get_action(state);
}

VectorXd Parl::get_action(const VectorXd& state, bool testing) {
  check_state_dim_(state);

  VectorXd raw_action = get_unconstrained_action(state);
  VectorXd noise_part = VectorXd::Zero(action_dim_);

  if (!testing) {
    MultivariateNormal d(MatrixXd::Identity(action_dim_, action_dim_));

    noise_part = d.sample() * action_noise_scale_;
  }

  if (params_.use_clipping)
    return constrain_action_(raw_action + noise_part);
  else 
    return raw_action + noise_part;
}

VectorXd Parl::simulated_step(const VectorXd& state, const VectorXd& action) {
  check_state_dim_(state);
  check_action_dim_(action);

  MatrixXd covar = get_pred_covar_(state);
  VectorXd pred_next_state = predict_next_state(state, action);

  MultivariateNormal noise_dist(covar);
  return pred_next_state + noise_dist.sample();
}

std::pair<VectorXd, MatrixXd> Parl::calculate_approx_value_gradient(const VectorXd& state) {
  check_state_dim_(state);

  int idx = ref_tree_->get_nearest_idx(state);

  MatrixXd A = get_A_matrix_idx_(idx);
  MatrixXd B = get_B_matrix_idx_(idx);
  VectorXd c = dynamics_models_[idx].get_identified_mats().second;

  MatrixXd K;
  VectorXd k;
  std::tie(k, K) = controllers_[idx].get_mats();
  VectorXd action = (K * state) + k;

  VectorXd pred_next_state = predict_next_state(state, action);

  VectorXd v_x = get_V_matrix_(pred_next_state);
  MatrixXd K_grad = B.transpose() * v_x * state.transpose();
  VectorXd k_grad = B.transpose() * v_x;

  return std::make_pair(k_grad, K_grad);
}

std::pair<VectorXd, MatrixXd> Parl::line_search(const VectorXd& k_grad,
    const MatrixXd& K_grad, const VectorXd& state, const VectorXd& k,
    const MatrixXd& K, double tau, double c) {
  check_state_dim_(state);

  // TODO (May want to abstract this to ml folder)

  double coef = 1.0;
  int i = 0;
  
  double m = k_grad.norm() + K_grad.norm();
  double t = c * m;

  VectorXd current_action = get_action(state);
  double current_pred_value = predict_value(predict_next_state(state, current_action));

  VectorXd ret_k_grad = k_grad;
  MatrixXd ret_K_grad = K_grad;
  double new_pred_value = 0.0;
  while (i < 15) {
    MatrixXd new_K = K + ret_K_grad;
    VectorXd new_k = k + ret_k_grad;
    VectorXd new_action = (new_K * state) + new_k;
    
    if (params_.clip_line_search && params_.use_clipping)
      new_action = constrain_action_(new_action);

    VectorXd pred_new_state = predict_next_state(state, new_action);

    if (params_.clip_line_search)
      pred_new_state = constrain_state_(pred_new_state);

    new_pred_value = predict_value(pred_new_state);

    if ((new_pred_value - current_pred_value) >= coef * t) {
      break;
    }

    coef *= tau;
    ret_k_grad = k_grad * coef;
    ret_K_grad = K_grad * coef;
    i += 1;
  }

  return std::make_pair(ret_k_grad, ret_K_grad);
}

void Parl::reset_value_model() {
  value_model_.reset();
}

void Parl::save() {
  // TODO
}

MatrixXd Parl::get_refs() {
  return refs_;
}

unsigned int Parl::get_nearest_ref_idx(const VectorXd& query) {
  if (query.size() != state_dim_)
    throw std::runtime_error("Passed state had the wrong dimension");
  
  return ref_tree_->get_nearest_idx(query);
}

std::vector<AutonomousLinearParams> Parl::get_controlled_system() {
  std::vector<AutonomousLinearParams> ret_vec;

  for (unsigned int i = 0; i < num_refs_; i++) {
    MatrixXd K;
    VectorXd k;
    std::tie(k, K) = controllers_[i].get_mats();
    VectorXd c = dynamics_models_[i].get_identified_mats().second;

    MatrixXd controlled_A = get_A_matrix_idx_(i) + (get_B_matrix_idx_(i) * K);
    VectorXd controlled_c = (get_B_matrix_idx_(i) * k) + c;
    ret_vec.push_back(AutonomousLinearParams(controlled_A, controlled_c,
          dynamics_models_[i].get_num_data()));
  }

  return ret_vec;
}

std::vector<AutonomousLinearParams> Parl::get_min_sat_controlled_system() {
  std::vector<AutonomousLinearParams> ret_vec;

  auto bounds = action_space_->getBounds();
  VectorXd min_sat_control = VectorXd::Zero(bounds.low.size());
  for (unsigned int i = 0; i < bounds.low.size(); i++) {
    min_sat_control[i] = bounds.low[i];
  }

  for (unsigned int i = 0; i < num_refs_; i++) {
    VectorXd c = dynamics_models_[i].get_identified_mats().second;

    MatrixXd controlled_A = get_A_matrix_idx_(i);
    VectorXd controlled_c = (get_B_matrix_idx_(i) * min_sat_control) + c;
    ret_vec.push_back(AutonomousLinearParams(controlled_A, controlled_c,
          dynamics_models_[i].get_num_data()));
  }

  return ret_vec;
}

std::vector<AutonomousLinearParams> Parl::get_max_sat_controlled_system() {
  std::vector<AutonomousLinearParams> ret_vec;

  auto bounds = action_space_->getBounds();
  VectorXd max_sat_control = VectorXd::Zero(bounds.high.size());
  for (unsigned int i = 0; i < bounds.high.size(); i++) {
    max_sat_control[i] = bounds.high[i];
  }

  for (unsigned int i = 0; i < num_refs_; i++) {
    VectorXd c = dynamics_models_[i].get_identified_mats().second;

    MatrixXd controlled_A = get_A_matrix_idx_(i);
    VectorXd controlled_c = (get_B_matrix_idx_(i) * max_sat_control) + c;
    ret_vec.push_back(AutonomousLinearParams(controlled_A, controlled_c,
          dynamics_models_[i].get_num_data()));
  }

  return ret_vec;
}

// Private methods
VectorXd Parl::make_combined_vec_(const VectorXd& state, const VectorXd& action) {
  check_state_dim_(state);
  check_action_dim_(action);

  VectorXd ret_vec(state_dim_ + action_dim_);
  ret_vec.head(state_dim_) = state;
  ret_vec.tail(action_dim_) = action;

  return ret_vec;
}

VectorXd Parl::make_local_state_(const VectorXd& global_state) {
  check_state_dim_(global_state);

  VectorXd r = ref_tree_->get_nearest_neighbor(global_state).first;
  return global_state - r;
}

double Parl::calculate_td_target_(double reward, const VectorXd& next_state,
    bool done, bool use_local) {
  check_state_dim_(next_state);

  double next_value = predict_value(next_state, use_local);

  if (done)
    return reward;
  else
    return reward + params_.discount_factor * next_value;
}

VectorXd Parl::get_V_matrix_(const VectorXd& state) {
  check_state_dim_(state);

  int idx = ref_tree_->get_nearest_idx(state); 
  return value_model_.get_mat(idx);
}

MatrixXd Parl::get_B_matrix_(const VectorXd& state) {
  check_state_dim_(state);

  int idx = ref_tree_->get_nearest_idx(state);
  MatrixXd theta = dynamics_models_[idx].get_identified_mats().first;
  return theta.rightCols(action_dim_);
}

MatrixXd Parl::get_A_matrix_idx_(int idx) {
  if (idx >= num_refs_)
    throw std::runtime_error("Ref point index too large");

  MatrixXd theta = dynamics_models_[idx].get_identified_mats().first;
  return theta.leftCols(state_dim_);
}

MatrixXd Parl::get_B_matrix_idx_(int idx) {
  if (idx >= num_refs_)
    throw std::runtime_error("Ref point index too large");

  MatrixXd theta = dynamics_models_[idx].get_identified_mats().first;
  return theta.rightCols(action_dim_);
}

VectorXd Parl::get_c_vector_idx_(int idx) {
  if (idx >= num_refs_)
    throw std::runtime_error("Ref point index too large");

  return dynamics_models_[idx].get_identified_mats().second;
}

void Parl::set_dynamics_idx_(int idx, const Ref<const MatrixXd> &A,
                             const Ref<const MatrixXd> &B,
                             const Ref<const VectorXd> &c,
                             const Ref<const VectorXd>& in_mean) {
  if (idx >= num_refs_)
    throw std::runtime_error("Ref point index too large");

  if (A.rows() != state_dim_ || A.cols() != state_dim_) 
    throw std::runtime_error("A matrix had wrong dimensions");

  if (B.rows() != state_dim_ || B.cols() != action_dim_) 
    throw std::runtime_error("B matrix had wrong dimensions");

  if (c.size() != state_dim_) 
    throw std::runtime_error("c vector had wrong dimensions");

  if (in_mean.size() != state_dim_ + action_dim_)
    throw std::runtime_error("Feature mean had wrong dimensions");

  MatrixXd theta(state_dim_ + action_dim_, state_dim_);
  theta << A.transpose(), 
           B.transpose();

  dynamics_models_[idx].set_params(theta, c, in_mean);
}

MatrixXd Parl::get_K_matrix_idx_(int idx) {
  if (idx >= num_refs_)
    throw std::runtime_error("Ref point index too large");

  return controllers_[idx].get_mats().second;
}

void Parl::set_K_matrix_idx_(int idx, const Ref<const MatrixXd>& K) {
  if (idx >= num_refs_)
    throw std::runtime_error("Ref point index too large");

  if (K.rows() != action_dim_ || K.cols() != state_dim_)
    throw std::runtime_error("Control gain matrix was the wrong size");

  controllers_[idx].set_K(K);
}

VectorXd Parl::get_k_vector_idx_(int idx) {
  if (idx >= num_refs_)
    throw std::runtime_error("Ref point index too large");

  return controllers_[idx].get_mats().first;
}

void Parl::set_k_matrix_idx_(int idx, const Ref<const VectorXd>& k) {
  if (idx >= num_refs_)
    throw std::runtime_error("Ref point index too large");

  if (k.size() != action_dim_)
    throw std::runtime_error("Control offset matrix was the wrong size");

  controllers_[idx].set_k(k);
}

MatrixXd Parl::get_pred_covar_(const VectorXd& state) {
  check_state_dim_(state);

  int idx = ref_tree_->get_nearest_idx(state);
  return dynamics_models_[idx].get_pred_error_covar();
}

inline void Parl::check_state_dim_(const VectorXd& state) const {
  if (state.size() != state_dim_)
    throw std::runtime_error("Passed state had the wrong dimension");
}

inline void Parl::check_action_dim_(const VectorXd& action) const {
  if (action.size() != action_dim_)
    throw std::runtime_error("Passed action had the wrong dimension");
}

inline VectorXd Parl::constrain_state_(const VectorXd& state) const {
  std::vector<double> state_vec(state.data(), state.data() + state.size());
  ob::State *tmp = state_space_->allocState();

  // TODO Is there a more efficient way to do this?
  state_space_->copyFromReals(tmp, state_vec);
  state_space_->enforceBounds(tmp);
  state_space_->copyToReals(state_vec, tmp);

  VectorXd ret_vec = Map<VectorXd, Unaligned>(state_vec.data(), state_vec.size());
  return ret_vec;
}

inline VectorXd Parl::constrain_action_(const VectorXd& action) const {
  VectorXd ret_vec(action_dim_);
  ob::RealVectorBounds b = action_space_->getBounds();

  for (unsigned int i = 0; i < action_dim_; i++) {
    ret_vec[i] = std::max(b.low[i], std::min(b.high[i], action[i]));
  }
  return ret_vec;
}
