#include <cannon/research/parl/runner.hpp>

using namespace cannon::research::parl;

// Public methods
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
  
  // Main experiment loop
  for (int i = 0; i < training_iterations; i++) {
    double ep_reward = 0.0;
    VectorXd state = env_->reset();

    for (int j = 0; j < num_timesteps; j++) {
      VectorXd new_state;
      double reward;
      bool done;

      VectorXd action = parl_->get_action(state);
      VectorXd pred_state = parl_->predict_next_state(state, action);
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
