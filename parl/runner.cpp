#include <cannon/research/parl/runner.hpp>

#include <cannon/research/parl/environment.hpp>
#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl/hyperparams.hpp>
#include <cannon/log/registry.hpp>
#include <cannon/math/multivariate_normal.hpp>
#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl_stability/transition_map.hpp>
#include <cannon/graphics/random_color.hpp>

# ifdef CANNON_BUILD_GRAPHICS
  #include <cannon/plot/plotter.hpp>
  using namespace cannon::plot;
# endif

using namespace cannon::log;
using namespace cannon::research::parl;
using namespace cannon::math;

// Public methods
Runner::Runner(EnvironmentPtr env, std::shared_ptr<Parl> p,
               const std::string &config_filename, bool render)
    : env_(env), parl_(p), render_(render) {

  load_config(config_filename);

  env_->reset();
}

Runner::Runner(EnvironmentPtr env, const std::string &config_filename,
               bool render, bool stability)
    : env_(env), render_(render) {

  load_config(config_filename);
  MatrixXd refs = sample_refs_();

# ifdef CANNON_BUILD_GRAPHICS
  if (refs.rows() == 2 && render_) {
    log_info("Plotting refs");
    MatrixX2f tmp = refs.transpose().cast<float>();

    Plotter plotter;
    plotter.plot_points(tmp);
    plotter.render();
  }
# endif

  Hyperparams params;
  params.load_config(config_filename);
  parl_ = std::make_shared<Parl>(env_->get_state_space(),
      env->get_action_space(), refs, params, 0, stability);

  env_->reset();

}

void Runner::load_config(const std::string& filename) {
  YAML::Node config = YAML::LoadFile(filename);

  if (!config["experiment"])
    throw std::runtime_error("Config file did not have an 'experiment' map.");

  YAML::Node experiment_params = config["experiment"];

  env_name = safe_get_param_<std::string>(experiment_params, "env_name");
  exp_id = safe_get_param_<std::string>(experiment_params, "exp_id");

  training_iterations = safe_get_param_<int>(experiment_params, "training_iterations");
  update_interval = safe_get_param_<int>(experiment_params, "update_interval");
  test_interval = safe_get_param_<int>(experiment_params, "test_interval");
  num_timesteps = safe_get_param_<int>(experiment_params, "num_timesteps");
  num_tests = safe_get_param_<int>(experiment_params, "num_tests");

  use_grid_refs = safe_get_param_<bool>(experiment_params, "use_grid_refs");
  ref_rows = safe_get_param_<int>(experiment_params, "ref_rows");
  ref_cols = safe_get_param_<int>(experiment_params, "ref_cols");
  random_refs = safe_get_param_<int>(experiment_params, "random_refs");
}

void Runner::do_initial_training_(int num_rollouts) {
  log_info("Doing initial training");
  for (int i = 0; i < num_rollouts; i++) {
    log_info("\tOn rollout", i);
    VectorXd state = env_->reset();

    for (int j = 0; j < num_timesteps; j++) {
      VectorXd new_state;
      double reward;
      bool done;

      int action_dim = env_->get_action_space()->getDimension();
      MultivariateNormal d(MatrixXd::Identity(action_dim, action_dim));

      VectorXd action = d.sample();
      std::tie(new_state, reward, done) = env_->step(action);

      parl_->process_datum(state, action, reward, new_state);
      state = new_state;
    }
  }
}

void Runner::do_value_grad_update_() {
  for (auto& p : transitions_) {
    parl_->value_grad_update_controller(p.first);
  }
  transitions_.clear();
}

void Runner::do_tests_(int ep_num) {
  double total_test_reward = 0.0;
  
  for (int i = 0; i < num_tests; i++) {
    double test_reward = 0.0;
    VectorXd state = env_->reset();

    for (int j = 0; j < num_timesteps; j++) {
      VectorXd new_state;
      double reward;
      bool done;

      VectorXd action = parl_->get_action(state, true);
      std::tie(new_state, reward, done) = env_->step(action);

      if (i == 0 && render_) {
        env_->render();
      }

      total_test_reward += reward;
      test_reward += reward;

      state = new_state;
    }

    test_reward_file_ << std::to_string(ep_num) << "," <<
      std::to_string(test_reward) << "," << std::endl;
  }

  double normalized_test_reward = total_test_reward / (float)(num_tests);
  log_info("Avg test reward is", normalized_test_reward);

  test_rewards_.push_back(normalized_test_reward);
}

void Runner::run() {
  do_initial_training_();

  // Open train/test reward files
  std::string log_path = std::string("logs/") + exp_id;
  std::filesystem::create_directories(log_path);

  train_reward_file_.open(log_path + "/training_rewards.csv", std::ios::trunc);
  test_reward_file_.open(log_path + "/test_rewards.csv", std::ios::trunc);

  train_reward_file_ << "episode,reward," << std::endl;
  test_reward_file_ << "episode,reward," << std::endl;

  int action_dim = env_->get_action_space()->getDimension();
  MultivariateNormal dist(MatrixXd::Identity(action_dim, action_dim));
  
  // Main experiment loop
  for (int i = 0; i < training_iterations; i++) {
    double ep_reward = 0.0;
    VectorXd state = env_->reset();

    for (int j = 0; j < num_timesteps; j++) {
      VectorXd new_state;
      double reward;
      bool done;

      VectorXd action = parl_->get_action(state);
      action += dist.sample() * 0.1;
      std::tie(new_state, reward, done) = env_->step(action);
      ep_reward += reward;

      transitions_.push_back(std::make_pair(state, action));
      
      parl_->process_datum(state, action, reward, new_state, done);
      state = new_state;
    }

    env_->register_ep_reward(ep_reward);
    
    train_reward_file_ << std::to_string(i) << "," << std::to_string(ep_reward)
      << "," << std::endl;
    
    //log_info("On episode", i, "reward is", ep_reward);
    ep_rewards_.push_back(ep_reward);

    if (i % update_interval == 0 && i != 0) {
      do_value_grad_update_();
    }

    if (i % test_interval == 0) {
      do_tests_(i);
    }
  }
}

std::shared_ptr<Parl> Runner::get_agent() {
  return parl_;
}

std::shared_ptr<Environment> Runner::get_env() {
  return env_;
}

// Private methods
MatrixXd Runner::sample_refs_() {
  if (use_grid_refs)
    return env_->sample_grid_refs(ref_rows, ref_cols);
  else
    return env_->sample_random_refs(random_refs);
}
