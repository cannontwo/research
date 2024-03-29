#ifndef CANNON_RESEARCH_PARL_ENVS_LINEAR_H
#define CANNON_RESEARCH_PARL_ENVS_LINEAR_H 

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <cannon/research/parl/environment.hpp>
#include <cannon/physics/systems/kinematic_car.hpp>

using namespace cannon::physics;

namespace cannon {
  namespace research {
    namespace parl {

      class LQREnvironment : public Environment {
        public:
          LQREnvironment(const Ref<const MatrixXd> &A,
                         const Ref<const MatrixXd> &B, const Ref<VectorXd> &c,
                         const Ref<MatrixXd> &Q, const Ref<MatrixXd> &R,
                         const VectorXd &s,
                         const VectorXd &g)
              : state_dim_(A.rows()), control_dim_(B.cols()), A_(A), B_(B),
                c_(c), Q_(Q), R_(R), state_(s), start_(s), goal_(g) {

            assert(A_.cols() == state_dim_);
            assert(B_.rows() == state_dim_);
            assert(c_.size() == state_dim_);
            assert(Q_.rows() == state_dim_);
            assert(Q_.cols() == state_dim_);
            assert(R_.rows() == control_dim_);
            assert(R_.cols() == control_dim_);

            state_space_ = std::make_shared<ob::RealVectorStateSpace>(state_dim_);
            ob::RealVectorBounds sb(state_dim_);
            sb.setLow(-1.0);
            sb.setHigh(1.0);
            state_space_->setBounds(sb);
            state_space_->setup();

            action_space_ = std::make_shared<oc::RealVectorControlSpace>(state_space_, control_dim_);
            ob::RealVectorBounds ab(control_dim_);
            ab.setLow(-2.0);
            ab.setHigh(2.0);
            action_space_->setBounds(ab);
            action_space_->setup();

            std::random_device rd;
            gen_ = std::mt19937(rd());  
            xy_dis_ = std::uniform_real_distribution<double>(-1.0, 1.0);
          }

          virtual std::shared_ptr<ob::StateSpace> get_state_space() const override {
            return state_space_;
          }

          virtual std::shared_ptr<oc::RealVectorControlSpace> get_action_space() const override {
            return action_space_;
          }

          virtual std::shared_ptr<systems::System> get_ode_sys() const override {
            // Not really the representative system
            return std::make_shared<systems::KinCarSystem>();
          }

          virtual VectorXd get_state() const override {
            return state_;
          }
          
          virtual MatrixXd sample_grid_refs(std::vector<int> grid_sizes) const override {
            // TODO This is not quite correct

            grid_sizes.resize(2, 1);

            int total_refs = 1;
            for (unsigned int i = 0; i < 2; ++i) {
              total_refs *= grid_sizes[i]; 
            }

            MatrixXd refs(2, total_refs);
            VectorXd xs = VectorXd::LinSpaced(grid_sizes[0],
                state_space_->getBounds().low[0],
                state_space_->getBounds().high[0]);
            VectorXd ys = VectorXd::LinSpaced(grid_sizes[1],
                state_space_->getBounds().low[1],
                state_space_->getBounds().high[1]);

            unsigned int idx = 0;
            for (int i = 0; i < grid_sizes[0]; i++) {
              for (int j = 0; j < grid_sizes[1]; j++) {
                refs(0, idx) = xs[i];
                refs(1, idx) = ys[j];

                ++idx;
              }
            }

            return refs;
          }

          virtual std::tuple<VectorXd, double, bool> step(const VectorXd& control) override {
            if (control.size() != control_dim_)
              throw std::runtime_error("Control passed to kinematic car had wrong dimension.");

            MatrixXd reward_mat = (state_.transpose() * Q_ * state_) + (control.transpose() * R_ * control);
            assert(reward_mat.rows() == 1);
            assert(reward_mat.cols() == 1);
            double reward = -reward_mat(0, 0);

            VectorXd clipped_control(control);
            clipped_control[0] = std::max(-2.0, std::min(clipped_control[0], 2.0));

            auto old_state = state_;
            state_ = A_ * state_ + B_ * clipped_control + c_;

            bool done = false;
            if (state_[0] >= 1.0 || state_[0] <= -1.0 ||
                state_[1] >= 1.0 || state_[1] <= -1.0) {
              done = true;
            }

            state_[0] = std::max(-1.0, std::min(1.0, state_[0]));
            state_[1] = std::max(-1.0, std::min(1.0, state_[1]));

            return std::make_tuple(state_, reward, done);
          }

          virtual void render() override {
            // Intentionally blank
          }

          virtual VectorXd reset() override {
            // Use atomic to update pendulum theta for rendering thread
            state_[0] = xy_dis_(gen_);
            state_[1] = xy_dis_(gen_);
            return state_;
          }

          VectorXd reset(const VectorXd& s) {
            assert(s.size() == state_dim_);
            state_ = s;
            return state_;
          }

          void register_ep_reward(float) override {}

          double get_time_step() override {
            return 0.01; 
          }

        private:
          std::shared_ptr<ob::RealVectorStateSpace> state_space_;
          std::shared_ptr<oc::RealVectorControlSpace> action_space_;

          unsigned int state_dim_;
          unsigned int control_dim_;

          MatrixXd A_;
          MatrixXd B_;
          VectorXd c_;

          MatrixXd Q_;
          MatrixXd R_;

          VectorXd state_;
          VectorXd start_;
          VectorXd goal_;

          std::mt19937 gen_;
          std::uniform_real_distribution<double> xy_dis_;

      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_ENVS_LINEAR_H */
