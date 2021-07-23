#include <chrono>
#include <thread>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl/runner.hpp>
#include <cannon/research/parl/envs/linear.hpp>
#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl_stability/transition_map.hpp>

using namespace cannon::research::parl;

int main() {
  Hyperparams params;

  // For discrete-time system
  double timestep = 0.01;

  MatrixXd A(2, 2);
  A << 0, 15,
       3, 0;
  A *= timestep;
  A += MatrixXd::Identity(2, 2);

  MatrixXd B(2, 1);
  B << 0, 
       3;

  B *= timestep;
  
  VectorXd c(Vector2d::Zero());

  MatrixXd Q(2, 2);
  Q << 1, 0,
       0, 0.1;

  MatrixXd R(1, 1);
  R << 0.001;

  auto env = std::make_shared<LQREnvironment>(A, B, c, Q, R, VectorXd::Ones(2), VectorXd::Zero(2));

  Runner r(env, "/home/cannon/Documents/cannon/cannon/research/experiments/parl_configs/r1c1_linear.yaml", false, false);
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

}
