#ifndef CANNON_RESEARCH_PARL_PLANNING_OMPL_UTILS_H
#define CANNON_RESEARCH_PARL_PLANNING_OMPL_UTILS_H 

#include <Eigen/Dense>
#include <cannon/research/parl/environment.hpp>

using namespace Eigen;

namespace cannon {
  namespace research {
    namespace parl {

      VectorXd get_coords_from_ompl_state(std::shared_ptr<Environment> env, ob::State*);

      // Currently assumes RealVectorControlSpace::ControlType
      VectorXd get_control_from_ompl_control(std::shared_ptr<Environment> env, oc::Control* c);

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_PLANNING_OMPL_UTILS_H */
