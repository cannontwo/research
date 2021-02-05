#ifndef CANNON_RESEARCH_PARL_RUNNER_H
#define CANNON_RESEARCH_PARL_RUNNER_H 

#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <filesystem> // C++17

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl/hyperparams.hpp>
#include <cannon/research/parl/environment.hpp>
#include <cannon/log/registry.hpp>
#include <cannon/math/multivariate_normal.hpp>
#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl_stability/transition_map.hpp>
#include <cannon/graphics/random_color.hpp>

# ifdef CANNON_BUILD_GRAPHICS
  #include <cannon/plot/plotter.hpp>
  using namespace cannon::plot;
# endif

using namespace Eigen;

using namespace cannon::log;
using namespace cannon::math;

namespace cannon {
  namespace research {
    namespace parl {

      class Runner {
        public:
          Runner() = delete;

          Runner(std::shared_ptr<Environment> env, std::shared_ptr<Parl> p,
              const std::string& config_filename, bool render = false) :
            env_(env), parl_(p), render_(render) {
            
            load_config(config_filename);

            env_->reset();
          }

          Runner(std::shared_ptr<Environment> env, const std::string&
              config_filename, bool render = false) : env_(env), render_(render) {

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
                env->get_action_space(), refs, params);

            env_->reset();

          }

          void load_config(const std::string& filename);
          void run();

          std::shared_ptr<Parl> get_agent();
          std::shared_ptr<Environment> get_env();

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

          void do_initial_training_(int num_rollouts = 100);
          void do_value_grad_update_();
          void do_tests_(int ep_num);


          std::shared_ptr<Environment> env_;
          std::shared_ptr<Parl> parl_;

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
