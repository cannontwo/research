#ifndef CANNON_RESEARCH_PARL_PARL_H
#define CANNON_RESEARCH_PARL_PARL_H 

#include <vector>
#include <utility>

#include <Eigen/Dense>
#include <ompl/base/StateSpace.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <cannon/control/affine_controller.hpp>
#include <cannon/ml/piecewise_lstd.hpp>
#include <cannon/ml/piecewise_recursive_lstd.hpp>
#include <cannon/ml/rls.hpp>
#include <cannon/research/parl/hyperparams.hpp>
#include <cannon/utils/class_forward.hpp>

using namespace Eigen;

using namespace cannon::control;
using namespace cannon::ml;

namespace ob = ompl::base;
namespace oc = ompl::control;

namespace cannon {

  namespace geom {
    CANNON_CLASS_FORWARD(KDTreeIndexed);
  }

  namespace research {
    namespace parl {

      CANNON_CLASS_FORWARD(AutonomousLinearParams);

      class Parl {
        public:
          Parl() = delete;

          /*!
           * \brief Constructor
           */
          Parl(const std::shared_ptr<ob::StateSpace> state_space,
               const std::shared_ptr<oc::RealVectorControlSpace> action_space,
               const MatrixXd &refs, Hyperparams params, int seed = 0,
               bool stability = false);

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
          unsigned int get_nearest_ref_idx(const VectorXd& query);

          MatrixXd get_A_matrix_idx_(int idx);
          MatrixXd get_B_matrix_idx_(int idx);
          VectorXd get_c_vector_idx_(int idx);

          void set_dynamics_idx_(int idx, const Ref<const MatrixXd> &A,
                                 const Ref<const MatrixXd> &B,
                                 const Ref<const VectorXd> &c,
                                 const Ref<const VectorXd> &in_mean);

          MatrixXd get_K_matrix_idx_(int idx);
          void set_K_matrix_idx_(int idx, const Ref<const MatrixXd>& K);

          VectorXd get_k_vector_idx_(int idx);
          void set_k_matrix_idx_(int idx, const Ref<const VectorXd>& k);

          std::vector<AutonomousLinearParams> get_controlled_system();
          std::vector<AutonomousLinearParams> get_min_sat_controlled_system();
          std::vector<AutonomousLinearParams> get_max_sat_controlled_system();

          friend class AggregateModel;

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

          // Refs whose Voronoi regions contain zero
          std::vector<unsigned int> zero_ref_idxs_;

          std::vector<RLSFilter> dynamics_models_;
          std::vector<AffineController> controllers_;
          //PiecewiseLSTDFilter value_model_;
          PiecewiseRecursiveLSTDFilter value_model_;

          geom::KDTreeIndexedPtr ref_tree_;

          bool stability_;
      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_PARL_H */
