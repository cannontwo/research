#pragma once
#ifndef CANNON_RESEARCH_PARL_LQR_STATE_PROPAGATOR_H
#define CANNON_RESEARCH_PARL_LQR_STATE_PROPAGATOR_H 

/*!
 * \file cannon/research/parl_planning/lqr_state_propagator.hpp
 * \brief File containing LQRStatePropagator class definition.
 */

#include <Eigen/Dense>

#include <ompl/control/StatePropagator.h>
#include <ompl/control/ODESolver.h>

namespace ob = ompl::base;
namespace oc = ompl::control;

using namespace Eigen;

namespace cannon {
  namespace research {
    namespace parl {

      /*!
       * \brief Class that propagates a state in an ODE using LQR controls.
       */
      class LQRStatePropagator : public oc::StatePropagator {
        public:

          /*!
           * \brief Function type for linearizations of an ODE
           */
          using ODELinearization =
              std::function<void(const oc::ODESolver::StateType & /* q0 */,
                                 const oc::ODESolver::StateType & /* u0 */,
                                 Ref<MatrixXd> /* A */,
                                 Ref<MatrixXd> /* B */)>;

          /*!
           * \brief Constructor taking space information, an ODE, its
           * linearization, and an optional post propagation event.
           */
          LQRStatePropagator(
              oc::SpaceInformationPtr si, oc::ODESolver::ODE ode,
              ODELinearization linearization,
              const oc::ODESolver::PostPropagationEvent &postEvent = nullptr);

          // === Methods from oc::StatePropagator ===

          /*!
           * \brief Inherited from oc::StatePropagator.
           */
          virtual void propagate(const ob::State* state,
                                 const oc::Control* control, double duration,
                                 ob::State *result) const override;

          /*!
           * \brief Inherited from oc::StatePropagator.
           */
          virtual bool steer(const ob::State *from, const ob::State *to,
                             oc::Control *result,
                             double &duration) const override;

          /*!
           * \brief Inherited from oc::StatePropagator.
           */
          virtual bool canSteer() const override {
            return true;
          }

          // === Methods specific to LQRStatePropagator ===
          
          /*!
           * \brief Get the state cost mastrix used for LQR computations.
           *
           * \returns Reference to state cost matrix.
           */
          const MatrixXd& get_state_cost_matrix() const {
            return Q_;
          }

          /*!
           * \brief Set the state cost matrix used for LQR computations.
           *
           * \param state_cost The new state cost matrix.
           */
          void set_state_cost_matrix(const Ref<MatrixXd>& state_cost);

          /*!
           * \brief Get decomposition of control cost matrix used for LQR
           * computations.
           *
           * \returns Cholesky decomposition of control cost matrix
           */
          const LLT<MatrixXd>& get_control_cost_decomposition() const {
            return R_cholesky_;
          }

          /*!
           * \brief Set the control cost matrix used for LQR computations.
           *
           * \param control_cost The new control cost matrix.
           */
          void set_control_cost_matrix(const Ref<MatrixXd>& control_cost);

        protected: 

          /*!
           * \brief Solve the continuous-time algebraic riccati equation for
           * the input state and control derivative matrices A and B, and
           * return the resulting LQR gain.
           *
           * \param A Partial derivatives of ode with respect to state.
           * \param B Partial derivatives of ode with respect to control.
           *
           * \returns CARE solution gain.
           */
          MatrixXd continuous_algebraic_riccati_equation(
              const Ref<const MatrixXd> &A, const Ref<const MatrixXd> &B) const;

          MatrixXd Q_; //!< LQR state cost matrix
          LLT<MatrixXd> R_cholesky_; //!< Cholesky decomposition of control cost matrix

          oc::ODESolverPtr solver_; //!< ODE solver
          ODELinearization linearization_; //!< ODE linearization
          oc::ODESolver::PostPropagationEvent post_event_; //!< Post-propagation event
          
      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_LQR_STATE_PROPAGATOR_H */
