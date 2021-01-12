#ifndef CANNON_RESEARCH_PARL_EXPERIMENT_H
#define CANNON_RESEARCH_PARL_EXPERIMENT_H 

#include <memory>
#include <utility>

#include <Eigen/Dense>
#include <ompl/base/StateSpace.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <cannon/physics/systems/system.hpp>

namespace ob = ompl::base;
namespace oc = ompl::control;

using namespace Eigen;
using namespace cannon::physics::systems;

namespace cannon {
  namespace research {
    namespace parl {

      class Environment {
        public:

          virtual std::shared_ptr<ob::StateSpace> get_state_space() const = 0;
          virtual std::shared_ptr<oc::RealVectorControlSpace> get_action_space() const = 0;

          virtual VectorXd get_state() const = 0;

          virtual std::shared_ptr<System> get_ode_sys() const = 0;

          virtual MatrixXd sample_random_refs(int num_refs) const {
            auto state_space = get_state_space();
            auto sampler = state_space->allocStateSampler();
            auto tmp_state = state_space->allocState();
            std::vector<double> tmp_vec(state_space->getDimension());

            MatrixXd ret_refs(state_space->getDimension(), num_refs);

            for (int i = 0; i < num_refs; i++) {
              sampler->sampleUniform(tmp_state);
              state_space->copyToReals(tmp_vec, tmp_state);
              VectorXd c = Map<VectorXd, Unaligned>(tmp_vec.data(), tmp_vec.size());
              ret_refs.col(i) = c;
            }
            
            return ret_refs;
          };

          virtual MatrixXd sample_grid_refs(int rows, int cols) const = 0;

          virtual std::tuple<VectorXd, double, bool> step(VectorXd control) = 0;

          virtual void render() = 0;

          virtual VectorXd reset() = 0;

          virtual void register_ep_reward(float ep_reward) = 0;

          virtual ~Environment() {};

      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_EXPERIMENT_H */
