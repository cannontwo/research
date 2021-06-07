#include <cannon/research/parl_planning/lqr_state_propagator.hpp>

#include <boost/numeric/odeint.hpp>

#include <cannon/research/parl_planning/lqr_control_space.hpp>

using namespace cannon::research::parl;

LQRStatePropagator::LQRStatePropagator(
    oc::SpaceInformationPtr si, oc::ODESolver::ODE ode,
    ODELinearization linearization,
    const oc::ODESolver::PostPropagationEvent &postEvent)
    : StatePropagator(si),
      Q_(MatrixXd::Identity(si->getStateDimension(), si->getStateDimension())),
      R_cholesky_(MatrixXd::Identity(si->getControlSpace()->getDimension(),
                                     si->getControlSpace()->getDimension())),
      solver_(
          std::make_shared<oc::ODEAdaptiveSolver<
              odeint::runge_kutta_dopri5<oc::ODESolver::StateType>>>(si, ode)),
      linearization_(std::move(linearization)),
      post_event_(std::move(postEvent)) {}

void LQRStatePropagator::propagate(const ob::State *state,
                                   const oc::Control *control, double duration,
                                   ob::State *result) const {
  oc::ODESolver::StateType reals;
  si_->getStateSpace()->copyToReals(reals, state);
  solver_->solve(reals, control, duration);
  si_->getStateSpace()->copyFromReals(result, reals);
  if (post_event_)
    post_event_(state, control, duration, result);
}

bool LQRStatePropagator::steer(const ob::State *from, const ob::State *to,
                   oc::Control *result,
                   double &duration) const {
  // TODO
}


void LQRStatePropagator::set_state_cost_matrix(const Ref<MatrixXd>& state_cost) {
  // TODO
}

void LQRStatePropagator::set_control_cost_matrix(const Ref<MatrixXd>& control_cost) {
  // TODO
}

MatrixXd LQRStatePropagator::continuous_algebraic_riccati_equation(
    const Ref<const MatrixXd> &A, const Ref<const MatrixXd> &B) const {
  // TODO
}
