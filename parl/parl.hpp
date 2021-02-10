#ifndef CANNON_RESEARCH_PARL_PARL_H
#define CANNON_RESEARCH_PARL_PARL_H 

#include <stdexcept>
#include <vector>
#include <cassert>
#include <utility>
#include <random>

#include <Eigen/Dense>
#include <ompl/base/StateSpace.h>
#include <ompl/base/ScopedState.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <cannon/geom/kd_tree_indexed.hpp>
#include <cannon/control/affine_controller.hpp>
#include <cannon/ml/piecewise_lstd.hpp>
#include <cannon/ml/rls.hpp>
#include <cannon/research/parl/hyperparams.hpp>
#include <cannon/research/parl/linear_params.hpp>
#include <cannon/math/multivariate_normal.hpp>


using namespace Eigen;

using namespace cannon::geom;
using namespace cannon::control;
using namespace cannon::ml;
using namespace cannon::math;

namespace ob = ompl::base;
namespace oc = ompl::control;

namespace cannon {
  namespace research {
    namespace parl {

      class Parl {
        public:
          Parl() = delete;
          
          Parl(const std::shared_ptr<ob::StateSpace> state_space, const
              std::shared_ptr<oc::RealVectorControlSpace> action_space, const MatrixXd&
              refs, Hyperparams params, int seed = 0) :
            state_dim_(state_space->getDimension()),
            action_dim_(action_space->getDimension()),
            state_space_(state_space), action_space_(action_space),
            seed_(seed), refs_(refs), params_(params),
          value_model_(state_dim_, refs.cols(), params.discount_factor),
          ref_tree_(state_dim_) {

            if (refs_.rows() != state_dim_)
              throw std::runtime_error("PARL reference points have the wrong dimension");
            num_refs_ = refs_.cols();

            if (seed != 0) {
              // TODO Set seed
            }

            if (params_.use_line_search && params_.controller_learning_rate != 1.0) {
              throw std::runtime_error("When using line search, learning rate should be 1");
            }
            
            
            ref_tree_.insert(refs_);

            for (int i = 0; i < num_refs_; i++) {
              // The dynamics models predict on (state, action) -> state
              dynamics_models_.emplace_back(state_dim_ + action_dim_,
                  state_dim_, params_.alpha, params_.forgetting_factor);

              controllers_.emplace_back(state_dim_, action_dim_,
                  params_.controller_learning_rate, params_.use_adam);

              // Checking KDT construction
              assert(ref_tree_.get_nearest_idx(refs_.col(i)) == i);
            }
          }

          void process_datum(const VectorXd& state, const VectorXd& action,
              double reward, const VectorXd& next_state, bool done = false,
              bool use_local = false);
          void value_grad_update_controller(const VectorXd& state);

          VectorXd predict_next_state(const VectorXd& state, const VectorXd& action, 
              bool use_local = false);
          double predict_value(const VectorXd& state, bool use_local = false);
          VectorXd compute_value_gradient(const VectorXd& state);
          VectorXd get_unconstrained_action(const VectorXd& state);
          VectorXd get_action(const VectorXd& state, bool testing = false);
          VectorXd simulated_step(const VectorXd& state, const VectorXd& action);
          std::pair<VectorXd, MatrixXd> calculate_approx_value_gradient(const VectorXd& state);
          std::pair<VectorXd, MatrixXd> line_search(const VectorXd& k_grad,
              const MatrixXd& K_grad, const VectorXd& state, const VectorXd& k,
              const MatrixXd& K, double tau = 0.5, double c = 0.5);

          void reset_value_model();
          void save();

          MatrixXd get_refs();
          MatrixXd get_B_matrix_idx_(int idx);
          MatrixXd get_A_matrix_idx_(int idx);
          MatrixXd get_K_matrix_idx_(int idx);
          VectorXd get_k_vector_idx_(int idx);

          std::vector<AutonomousLinearParams> get_controlled_system();
          std::vector<AutonomousLinearParams> get_min_sat_controlled_system();
          std::vector<AutonomousLinearParams> get_max_sat_controlled_system();

        private:
          VectorXd make_combined_vec_(const VectorXd& state, const VectorXd& action);
          VectorXd make_local_state_(const VectorXd& global_state);

          double calculate_td_target_(double reward, const VectorXd&
              next_state, bool done = false, bool use_local = false);

          VectorXd get_V_matrix_(const VectorXd& state);
          MatrixXd get_B_matrix_(const VectorXd& state);
          MatrixXd get_pred_covar_(const VectorXd& state);

          inline void check_state_dim_(const VectorXd& state) const;
          inline void check_action_dim_(const VectorXd& action) const;
          inline VectorXd constrain_state_(const VectorXd& state) const;
          inline VectorXd constrain_action_(const VectorXd& action) const;

          unsigned int state_dim_;
          unsigned int action_dim_;
          std::shared_ptr<ob::StateSpace> state_space_;
          std::shared_ptr<oc::RealVectorControlSpace> action_space_;
          int seed_;

          // TODO Move to hyperparams
          double action_noise_scale_ = 0.1;

          int num_refs_;
          MatrixXd refs_;
          Hyperparams params_;

          std::vector<RLSFilter> dynamics_models_;
          std::vector<AffineController> controllers_;
          PiecewiseLSTDFilter value_model_;

          KDTreeIndexed ref_tree_;
      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_PARL_H */
