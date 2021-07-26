#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <cannon/control/lqr.hpp>
#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl/runner.hpp>
#include <cannon/research/parl/envs/linear.hpp>
#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl_stability/transition_map.hpp>

using namespace cannon::control;
using namespace cannon::research::parl;

int main() {
  Hyperparams params;

  double timestep = 0.01;

  MatrixXd A(2, 2);
  A << 0, 15,
       3, 0;

  MatrixXd B(2, 1);
  B << 0, 
       3;

  auto lin_func = [A, B](const Ref<const VectorXd>&, Ref<MatrixXd> ret_A, Ref<MatrixXd> ret_B) {
    ret_A = A;
    ret_B = B;
  };

  LQRController controller(Vector2d::Zero(), 1, lin_func);
  log_info("LQR controller is", controller.get_linear_gain());

  // For discrete-time system
  A *= timestep;
  A += MatrixXd::Identity(2, 2);
  B *= timestep;
  
  VectorXd c(Vector2d::Zero());

  MatrixXd Q(2, 2);
  Q << 1, 0,
       0, 0.1;

  MatrixXd R(1, 1);
  R << 0.001;

  auto env = std::make_shared<LQREnvironment>(A, B, c, Q, R, VectorXd::Ones(2), VectorXd::Zero(2));

  auto initial_control_func = [&](const Ref<const VectorXd>& x) {
    return controller.compute_control(x);
  };

  Runner r(env,
           "/home/cannon/Documents/cannon/cannon/research/experiments/"
           "parl_configs/r1c1_linear.yaml",
           initial_control_func, false, false);

  r.run();

  // Plot value function
  auto parl = r.get_agent();
  auto value_func = [&](const Vector2d& x) {
    return parl->predict_value(x);
  };

  // Plot control function
  auto control_func = [&](const Vector2d& x) {
    return parl->get_action(x, true)(0, 0);
  };

  {
  Plotter plotter;
  plotter.plot(value_func, 10, -1.0, 1.0, -1.0, 1.0);
  plotter.render();
  }

  {
  Plotter plotter;
  plotter.plot(control_func, 10, -1.0, 1.0, -1.0, 1.0);
  plotter.render();
  }

  auto controlled_system = r.get_agent()->get_controlled_system();
  auto diagram = compute_voronoi_diagram(r.get_agent()->get_dynam_refs());
  auto parl_pwa_func = compute_parl_pwa_func(r.get_agent(), diagram);
  save_pwa(parl_pwa_func, std::string("models/parl_lqr_pwa_") +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
      + std::string(".h5"));

  log_info("K is ", parl->get_K_matrix_idx_(0));
  log_info("k is ", parl->get_k_vector_idx_(0));
}
