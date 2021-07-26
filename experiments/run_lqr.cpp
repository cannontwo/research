#include <thread>

#include <cannon/control/lqr.hpp>
#include <cannon/plot/plotter.hpp>

#include <cannon/research/parl/envs/linear.hpp>

using namespace cannon::research::parl;
using namespace cannon::plot;
using namespace cannon::control;

int main() {
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
  env->reset();

  std::vector<double> rewards;
  unsigned int num_tests = 10;
  unsigned int num_timesteps = 200;
  
  for (int j = 0; j < 100; ++j) {

    double total_test_reward = 0.0;
    for (int i = 0; i < num_tests; i++) {
      double test_reward = 0.0;
      VectorXd state = env->reset();

      for (int j = 0; j < num_timesteps; j++) {
        VectorXd new_state;
        double reward;
        bool done;

        VectorXd action = controller.compute_control(state);
        std::tie(new_state, reward, done) = env->step(action);

        total_test_reward += reward;
        test_reward += reward;

        state = new_state;

        // During testing we don't break on done in order to get consistent test
        // rewards
      }
    }

    double normalized_test_reward = total_test_reward / (float)(num_tests);
    log_info("Avg test reward is", normalized_test_reward);

    rewards.push_back(normalized_test_reward);
  }

  {
    Plotter r_plotter;
    r_plotter.plot(rewards);
    r_plotter.render();
  }
}
