#ifndef CANNON_RESEARCH_PARL_PLANNING_AGGREGATE_MODEL_H
#define CANNON_RESEARCH_PARL_PLANNING_AGGREGATE_MODEL_H 

#include <map>
#include <cassert>

#include <Eigen/Dense>
#include <ompl/control/PathControl.h>

#include <cannon/ml/rls.hpp>
#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl_planning/ompl_utils.hpp>

using namespace Eigen;

using namespace cannon::ml;

namespace cannon {
  namespace research {
    namespace parl {

      using VectorXu = Matrix<unsigned int, Dynamic, 1>;

      bool vector_comp(const VectorXu& v1, const VectorXu& v2);

      struct LinearParams {
        LinearParams() = delete;

        // Copy constructor
        LinearParams(const LinearParams& o) : A_(o.A_), B_(o.B_), c_(o.c_),
        num_data_(o.num_data_), state_dim_(o.state_dim_), action_dim_(o.action_dim_) {}

        // Move constructor
        LinearParams(LinearParams&& o) : A_(std::move(o.A_)),
        B_(std::move(o.B_)), c_(std::move(o.c_)),
        num_data_(o.num_data_), state_dim_(o.state_dim_), action_dim_(o.action_dim_) {}

        // Copy Assignment
        LinearParams& operator=(const LinearParams& o) {
          A_ = o.A_;
          B_ = o.B_;
          c_ = o.c_;
          num_data_ = o.num_data_;
          state_dim_ = o.state_dim_;
          action_dim_ = o.action_dim_;

          return *this;
        }

        ~LinearParams() {}

        LinearParams(const MatrixXd& A, const MatrixXd& B, const VectorXd c,
            unsigned int num_data) {
          assert(c.size() == A.rows());
          assert(c.size() == A.cols());

          assert(c.size() == B.rows());

          state_dim_ = c.size();
          action_dim_ = B.cols();
          
          A_ = A;
          B_ = B;
          c_ = c;
          num_data_ = num_data;
        }

        LinearParams(unsigned int state_dim, unsigned int action_dim) :
          state_dim_(state_dim), action_dim_(action_dim)  {

          A_ = MatrixXd::Zero(state_dim_, state_dim_);
          B_ = MatrixXd::Zero(state_dim_, action_dim_);
          c_ = VectorXd::Zero(state_dim_);
          num_data_ = 0;
        }

        // Merge another LinearParams object via data-weighted averaging
        void merge(const LinearParams& o) {
          assert(o.state_dim_ == state_dim_);
          assert(o.action_dim_ == action_dim_);

          unsigned int total_datapoints = o.num_data_ + num_data_;
          double t = (double)num_data_ / (double)total_datapoints;

          A_ = t*A_ + (1.0 - t)*o.A_;
          B_ = t*B_ + (1.0 - t)*o.B_;
          c_ = t*c_ + (1.0 - t)*o.c_;

          num_data_ += o.num_data_;
        }

        MatrixXd A_;
        MatrixXd B_;
        VectorXd c_;
        unsigned int num_data_;

        unsigned int state_dim_;
        unsigned int action_dim_;
      };

      class AggregateModel {
        public:
          AggregateModel() = delete;

          AggregateModel(unsigned int state_dim, unsigned int action_dim,
              unsigned int grid_size, Matrix2Xd& bounds, double time_delta) :
            state_dim_(state_dim), action_dim_(action_dim),
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

          void add_local_model(const RLSFilter& model, const VectorXd&
              ref_state, const VectorXd& next_ref_state, 
              const VectorXd& ref_control, double tau, double tau_delta);

          void process_path_parl(std::shared_ptr<Environment> env,
              std::shared_ptr<Parl> model, oc::PathControl& path);

          // TODO Write function to use this model for planning, in combination with nominal model

          VectorXu get_grid_coords(const VectorXd& query) const;

        private:
          unsigned int state_dim_;
          unsigned int action_dim_;
          unsigned int grid_size_;

          Matrix2Xd bounds_;
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
