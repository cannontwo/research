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
  unsigned int n = si_->getStateDimension();
  unsigned int m = si_->getControlSpace()->getDimension();
  oc::ODESolver::StateType q_final(n), q_final_dot(n), u_final(m, 1e-2);

  // Copy "to" state to q0
  si_->getStateSpace()->copyToReals(q_final, to);
  VectorXd q0 = Map<VectorXd>(&q_final[0], n);
  VectorXd q0_dot = Map<VectorXd>(&q_final_dot[0], n);

  solver_->getODE()(q_final, result, q_final_dot);

  // Linearized dynamics:
  // qdot ~= f(q0, u0) + A * (q - q0) + B * (u - u0)
  //       = q0dot + A * (q - q0) + B * (u - u0)
  // LQR control policy:
  // u* = -K * (q - q0) + u0
  
  // Compute A, B, K
  MatrixXd A = MatrixXd(n, n);
  MatrixXd B = MatrixXd(n, m);
  linearization_(q_final, u_final, A, B);
  auto lqr_control = static_cast<LQRControlSpace::ControlType*>(result);
  lqr_control->q0 = q0;
  lqr_control->K = continuous_algebraic_riccati_equation(A, B);

  // Attempt to steer
  unsigned int step = 0;
  unsigned int max_steps = si_->getMaxControlDuration();
  const double step_size = si_->getPropagationStepSize();
  ob::State* state = si_->cloneState(from);
  ob::State* state_prev = si_->cloneState(from);
  double dist = 0.0;
  double dist_prev = si_->distance(state_prev, to);

  while (step < max_steps) {
    propagate(state, result, step_size, state);
    dist = si_->distance(state, to);

    // If moving away from the goal, backtrack and terminate
    if (dist > dist_prev) {
      std::swap(state, state_prev);
      break;
    }

    ++step;

    // If we reach the "to" state, terminate
    if (dist < 1e-3)
      break;

    std::swap(state, state_prev);
    dist_prev = dist;
  }

  duration = step * step_size;
  si_->freeState(state);
  si_->freeState(state_prev);
  return step > 0u;
}


void LQRStatePropagator::set_state_cost_matrix(const Ref<MatrixXd>& state_cost) {
  if (state_cost.rows() == Q_.rows() && state_cost.cols() == Q_.cols())
    Q_ = state_cost;
  else
    OMPL_ERROR("LQRControlSpace: state cost matrix is the wrong size: %ux%u "
               "(should be %ux%u)",
               state_cost.rows(), state_cost.cols(), Q_.rows(), Q_.cols());
}

void LQRStatePropagator::set_control_cost_matrix(const Ref<MatrixXd>& control_cost) {
  if (control_cost.rows() == R_cholesky_.rows() && control_cost.cols() == R_cholesky_.cols()) {
    R_cholesky_ = LLT<MatrixXd>(control_cost);
    if (R_cholesky_.info() != Eigen::Success)
      OMPL_ERROR("LQRControlSpace: the control cost matrix must be positive definite");
  } else
    OMPL_ERROR("LQRControlSpace: control cost matrix is the wrong size: %ux%u "
               "(should be %ux%u)",
               control_cost.rows(), control_cost.cols(), R_cholesky_.rows(),
               R_cholesky_.cols());
}

// Code taken from the Drake project, covered by the same 3-clause BSD license as OMPL code. See:
// https://github.com/RobotLocomotion/drake/blob/master/math/continuous_algebraic_riccati_equation.cc
// https://github.com/RobotLocomotion/drake/blob/master/LICENSE.TXT
MatrixXd LQRStatePropagator::continuous_algebraic_riccati_equation(
    const Ref<const MatrixXd> &A, const Ref<const MatrixXd> &B) const {
  // R. Byers. Solving the algebraic Riccati equation with the matrix sign
  // function. Linear Algebra Appl., 85:267–279, 1987
  // Added determinant scaling to improve convergence (converges in rough half
  // the iterations with this)
  
  const Index n = B.rows();
  MatrixXd H(2*n, 2*n);

  H << A,   B * R_cholesky_.solve(B.transpose()),
       Q_, -A.transpose();

  MatrixXd Z = H;
  MatrixXd Z_old;

  // Configurable options
  const double tolerance = 1e-9;
  const double max_iterations = 100;

  double relative_norm;
  size_t iteration = 0;
  const double p = static_cast<double>(Z.rows());

  do {
    Z_old = Z;

    double ck = std::pow(std::abs(Z.determinant()), -1.0 / p);
    Z *= ck;
    Z = Z - 0.5 * (Z - Z.inverse());
    relative_norm = (Z - Z_old).norm();

    iteration++;
  } while (iteration < max_iterations && relative_norm > tolerance);

  MatrixXd W11 = Z.block(0, 0, n, n);
  MatrixXd W12 = Z.block(0, n, n, n);
  MatrixXd W21 = Z.block(n, 0, n, n);
  MatrixXd W22 = Z.block(n, n, n, n);

  MatrixXd lhs(2*n, n);
  MatrixXd rhs(2*n, n);
  MatrixXd eye = MatrixXd::Identity(n, n);
  lhs << W12, 
         W22 + eye;
  rhs << W11 + eye,
         W21;

  JacobiSVD<MatrixXd> svd(lhs, Eigen::ComputeThinU | Eigen::ComputeThinV);

  return R_cholesky_.solve(B.transpose() * svd.solve(rhs));
}
