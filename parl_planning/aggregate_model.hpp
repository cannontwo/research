#ifndef CANNON_RESEARCH_PARL_PLANNING_AGGREGATE_MODEL_H
#define CANNON_RESEARCH_PARL_PLANNING_AGGREGATE_MODEL_H 

#include <map>
#include <cassert>

#include <Eigen/Dense>
#include <ompl/control/PathControl.h>

#include <cannon/ml/rls.hpp>
#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl_planning/ompl_utils.hpp>
#include <cannon/research/parl/linear_params.hpp>

using namespace Eigen;

using namespace cannon::ml;

namespace cannon {
  namespace research {
    namespace parl {

      using VectorXu = Matrix<unsigned int, Dynamic, 1>;

      bool vector_comp(const VectorXu& v1, const VectorXu& v2);

      class AggregateModel : public System {
        public:
          AggregateModel() = delete;

          AggregateModel(std::shared_ptr<System> nominal_model, unsigned int
              state_dim, unsigned int action_dim, unsigned int grid_size,
              MatrixX2d& bounds, double time_delta) :
            nominal_model_(nominal_model), state_dim_(state_dim), action_dim_(action_dim),
            grid_size_(grid_size), bounds_(bounds), time_delta_(time_delta), parameters_(vector_comp) {

            assert(bounds_.rows() == state_dim_);
            assert(time_delta >= 0.0);

            // bounds[:, 0] is lower bounds, bounds[:, 1] is upper bounds
            cell_extent_ = VectorXd::Zero(state_dim_);
            for (unsigned int i = 0; i < state_dim_; i++) {
              assert(bounds_(i, 1) > bounds_(i, 0));
              double length = bounds_(i, 1) - bounds_(i, 0);
              
              cell_extent_[i] = length / (double)grid_size_;
            }
          }

          virtual void operator()(const VectorXd& s, VectorXd& dsdt, const double t) override;

          virtual void ompl_ode_adaptor(const oc::ODESolver::StateType& q,
              const oc::Control* control, oc::ODESolver::StateType& qdot) override;

          void add_local_model(const RLSFilter& model, const VectorXd&
              ref_state, const VectorXd& next_ref_state, 
              const VectorXd& ref_control, double tau, double tau_delta);

          void process_path_parl(std::shared_ptr<Environment> env,
              std::shared_ptr<Parl> model, oc::PathControl& path);

          LinearParams get_local_model_for_state(const VectorXd& state);

          VectorXu get_grid_coords(const VectorXd& query) const;

        private:
          std::shared_ptr<System> nominal_model_;

          unsigned int state_dim_;
          unsigned int action_dim_;
          unsigned int grid_size_;

          MatrixX2d bounds_;
          VectorXd cell_extent_;

          double time_delta_;
          
          std::map<VectorXu, LinearParams, std::function<bool(const VectorXu&,
              const VectorXu&)>, aligned_allocator<std::pair<VectorXu,
            LinearParams>>> parameters_;

      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_PLANNING_AGGREGATE_MODEL_H */
