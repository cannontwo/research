#ifndef CANNON_RESEARCH_PARL_RUNNER_H
#define CANNON_RESEARCH_PARL_RUNNER_H 

#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <filesystem> // C++17
#include <functional>

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

#include <cannon/utils/class_forward.hpp>

using namespace Eigen;

namespace cannon {
  namespace research {
    namespace parl {

      CANNON_CLASS_FORWARD(Environment);
      CANNON_CLASS_FORWARD(Parl);

      using ControlFunc = std::function<VectorXd(const Ref<const VectorXd>&)>;

      class Runner {
        public:
          Runner() = delete;

          /*!
           * \brief Constructor taking environment, PARL pointer, and config
           * file to load.
           */
          Runner(EnvironmentPtr env, ParlPtr p,
              const std::string& config_filename, bool render = false);

          /*!
           * \brief Constructor taking environment and config filename.
           */
          Runner(EnvironmentPtr env, const std::string&
              config_filename, bool render = false, bool stability = false);

          /*!
           * \brief Constructor taking environment, config filename, and initial controller.
           */
          Runner(EnvironmentPtr env, const std::string &config_filename,
                 ControlFunc initial_controller, bool render = false,
                 bool stability = false);

          void load_config(const std::string& filename);
          void run();

          ParlPtr get_agent();
          EnvironmentPtr get_env();

          std::string env_name;
          std::string exp_id;
          int training_iterations;
          int update_interval;
          int test_interval;
          int num_timesteps;
          int num_tests;

          bool use_grid_refs;
          int ref_rows;
          int ref_cols;

          bool use_value_refs;
          int value_ref_rows;
          int value_ref_cols;

          int random_refs;

        private:
          template <typename T>
          T safe_get_param_(YAML::Node config, const std::string& param) {
            if (!config[param]) {
              throw std::runtime_error(std::string("Could not get param ") + param + " in YAML config.");
            } else {
              return config[param].as<T>();
            }
          }

          MatrixXd sample_refs_();
          MatrixXd sample_value_refs_();

          void do_initial_training_(int num_rollouts = 100);
          void do_value_grad_update_();
          void do_tests_(int ep_num);

          EnvironmentPtr env_;
          ParlPtr parl_;
          ControlFunc initial_controller_;

          std::vector<double> ep_rewards_;
          std::vector<double> test_rewards_;
          std::vector<std::pair<VectorXd, VectorXd>> transitions_;

          std::ofstream train_reward_file_;
          std::ofstream test_reward_file_;

          bool render_;

      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_RUNNER_H */
