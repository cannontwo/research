#ifndef CANNON_RESEARCH_PARL_LINEAR_PARAMS_H
#define CANNON_RESEARCH_PARL_LINEAR_PARAMS_H 

#include <Eigen/Dense>

using namespace Eigen;

namespace cannon {
  namespace research {
    namespace parl {

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

        ~LinearParams() {}

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

      struct AutonomousLinearParams {
        AutonomousLinearParams() = delete;

        // Copy constructor
        AutonomousLinearParams(const AutonomousLinearParams& o) : A_(o.A_), c_(o.c_),
        num_data_(o.num_data_), state_dim_(o.state_dim_) {}

        // Move constructor
        AutonomousLinearParams(AutonomousLinearParams&& o) : A_(std::move(o.A_)), c_(std::move(o.c_)),
        num_data_(o.num_data_), state_dim_(o.state_dim_) {}

        // Copy Assignment
        AutonomousLinearParams& operator=(const AutonomousLinearParams& o) {
          A_ = o.A_;
          c_ = o.c_;
          num_data_ = o.num_data_;
          state_dim_ = o.state_dim_;

          return *this;
        }

        ~AutonomousLinearParams() {}

        AutonomousLinearParams(const MatrixXd& A, const VectorXd c,
            unsigned int num_data) {
          assert(c.size() == A.rows());
          assert(c.size() == A.cols());


          state_dim_ = c.size();
          
          A_ = A;
          c_ = c;
          num_data_ = num_data;
        }

        AutonomousLinearParams(unsigned int state_dim) :
          state_dim_(state_dim) {

          A_ = MatrixXd::Zero(state_dim_, state_dim_);
          c_ = VectorXd::Zero(state_dim_);
          num_data_ = 0;
        }

        // Merge another LinearParams object via data-weighted averaging
        void merge(const AutonomousLinearParams& o) {
          assert(o.state_dim_ == state_dim_);

          unsigned int total_datapoints = o.num_data_ + num_data_;
          double t = (double)num_data_ / (double)total_datapoints;

          A_ = t*A_ + (1.0 - t)*o.A_;
          c_ = t*c_ + (1.0 - t)*o.c_;

          num_data_ += o.num_data_;
        }

        MatrixXd A_;
        VectorXd c_;
        unsigned int num_data_;

        unsigned int state_dim_;
      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_LINEAR_PARAMS_H */
