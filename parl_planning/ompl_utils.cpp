#include <cannon/research/parl_planning/ompl_utils.hpp>

using namespace cannon::research::parl;


VectorXd cannon::research::parl::get_coords_from_ompl_state(std::shared_ptr<Environment> env, 
    ob::State* s) {
  std::vector<double> coord_vec;

  env->get_state_space()->copyToReals(coord_vec, s);

  VectorXd ret_vec(coord_vec.size());
  ret_vec = Map<VectorXd, Unaligned>(coord_vec.data(), coord_vec.size());

  return ret_vec;
}

VectorXd cannon::research::parl::get_control_from_ompl_control(std::shared_ptr<Environment> env, 
    oc::Control* c) {

  VectorXd ret_vec = VectorXd::Zero(env->get_action_space()->getDimension());

  for (unsigned int i = 0; i < env->get_action_space()->getDimension(); i++) {
    ret_vec[i] = c->as<oc::RealVectorControlSpace::ControlType>()->values[i];
  }

  return ret_vec;
}
