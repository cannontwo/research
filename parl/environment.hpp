#ifndef CANNON_RESEARCH_PARL_EXPERIMENT_H
#define CANNON_RESEARCH_PARL_EXPERIMENT_H 

#include <memory>
#include <utility>

#include <Eigen/Dense>
#include <ompl/base/StateSpace.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <cannon/utils/class_forward.hpp>

namespace ob = ompl::base;
namespace oc = ompl::control;

using namespace Eigen;

namespace cannon {

  namespace physics {
    namespace systems {
      CANNON_CLASS_FORWARD(System);
    }
  }

  namespace research {
    namespace parl {

      /*!
       * \brief Interface representing an environment that can be stepped in
       * discrete time to do learned control.
       */
      class Environment {
        public:

          /*!
           * \brief Get the OMPL state space for this environment.
           *
           * \returns Shared pointer to the observable state space for this environment.
           */
          virtual std::shared_ptr<ob::StateSpace> get_state_space() const = 0;
          
          /*!
           * \brief Get the OMPL control space for this environment.
           *
           * \returns Shared pointer to the action space for this environment.
           */
          virtual std::shared_ptr<oc::RealVectorControlSpace> get_action_space() const = 0;

          /*!
           * \brief Get the current state of this environment as a vector of
           * real numbers.
           *
           * \returns Current state.
           */
          virtual VectorXd get_state() const = 0;

          /*!
           * \brief Get the differential equation system defining this environment.
           *
           * \returns Pointer to the system.
           */
          virtual cannon::physics::systems::SystemPtr get_ode_sys() const = 0;

          /*!
           * \brief Sample the given number of random states for this environment.
           *
           * \param num_refs Number of reference states to sample.
           *
           * \returns A matrix containing the sampled refs, one in each column.
           */
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

          /*!
           * \brief Sample a grid of reference states in this environment.
           *
           * \param grid_sizes Vector of sizes for each dimension of the
           * reference state grid. Should have size > 0.
           *
           * \returns Matrix containing sampled reference states, one in each column.
           */
          virtual MatrixXd sample_grid_refs(std::vector<int> grid_sizes) const = 0;

          /*!
           * \brief Step the state of this environment, advancing time by the discrete time step.
           *
           * \param control The control to step this environment with.
           *
           * \returns A tuple containing the new state of the environment, the
           * environment reward for this step, and whether the current episode
           * of execution in this environment should be terminated.
           */
          virtual std::tuple<VectorXd, double, bool> step(const VectorXd& control) = 0;

          /*!
           * \brief Get the constrained/clipped control corresponding to the
           * input unconstrained control in this environment.
           *
           * \param control Unconstrained commanded control.
           *
           * \returns Constrained control
           */
          virtual VectorXd get_constrained_control(const Ref<const VectorXd>& control) const {
            auto action_space = get_action_space();
            auto bounds = action_space->getBounds();

            assert(static_cast<long int>(bounds.low.size()) == control.size());

            VectorXd ret_control = VectorXd::Zero(control.size());
            for (unsigned int i = 0; i < control.size(); ++i) {
              ret_control[i] = std::min(bounds.high[i], std::max(bounds.low[i], control[i]));
            }

            return ret_control;
          }

          /*!
           * \brief Get the constrained/clipped state corresponding to the
           * input unconstrained state in this environment.
           *
           * \param control Unconstrained state.
           *
           * \returns Constrained state
           */
          virtual VectorXd get_constrained_state(const Ref<const VectorXd>& state) const {
            std::vector<double> state_vec(state.data(), state.data() + state.size());

            auto state_space = get_state_space();
            ob::State *s = state_space->allocState();

            state_space->copyFromReals(s, state_vec); 
            state_space->enforceBounds(s);
            state_space->copyToReals(state_vec, s);
            state_space->freeState(s);

            VectorXd ret_state = Map<VectorXd, Unaligned>(state_vec.data(), state_vec.size());

            return ret_state;
          }

          /*!
           * \brief Render the state of this environment. By default this is a no-op.
           */
          virtual void render() {}

          /*!
           * \brief Reset the state of this environment to its default.
           *
           * \returns The reset state of the environment.
           */
          virtual VectorXd reset() = 0;

          /*!
           * \brief Register the current episode reward for display.
           *
           * \param ep_reward Current episode reward.
           */
          virtual void register_ep_reward(float ep_reward) = 0;

          /*!
           * \brief Destructor.
           */
          virtual ~Environment() {};

          /*!
           * \brief Get the discrete time step of this environment.
           *
           * \returns Environment time step.
           */
          virtual double get_time_step() = 0;

      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_EXPERIMENT_H */
