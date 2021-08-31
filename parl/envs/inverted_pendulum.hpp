#ifndef CANNON_RESEARCH_PARL_ENVS_INVERTED_PENDULUM_H
#define CANNON_RESEARCH_PARL_ENVS_INVERTED_PENDULUM_H 

#include <utility>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

#include <cannon/research/parl/environment.hpp>
#include <cannon/physics/systems/inverted_pendulum.hpp>
#include <cannon/utils/blocking_queue.hpp>

#ifdef CANNON_BUILD_GRAPHICS
  #include <cannon/graphics/deferred_renderer.hpp>
  #include <cannon/graphics/geometry/model.hpp>
  #include <cannon/graphics/viewer_3d.hpp>
  #include <cannon/graphics/window.hpp>
  using namespace cannon::graphics;
#endif

using namespace cannon::physics::systems;
using namespace cannon::utils;

namespace cannon {
  namespace research {
    namespace parl {

      class InvertedPendulumEnvironment : public Environment {
        public:
          InvertedPendulumEnvironment() : state_(2) {
            auto th_space = std::make_shared<ob::SO2StateSpace>();
            auto th_dot_space = std::make_shared<ob::RealVectorStateSpace>(1);
            ob::RealVectorBounds b(1);
            b.setLow(-8.0);
            b.setHigh(8.0);
            th_dot_space->setBounds(b);

            state_space_ = std::make_shared<ob::CompoundStateSpace>();
            state_space_->addSubspace(th_space, 1.0);
            state_space_->addSubspace(th_dot_space, 1.0);
            state_space_->setup();

            action_space_ = std::make_shared<oc::RealVectorControlSpace>(state_space_, 1);
            ob::RealVectorBounds b2(1);
            b2.setLow(-2.0);
            b2.setHigh(2.0);
            action_space_->setBounds(b2);
            action_space_->setup();
          }

          virtual std::shared_ptr<ob::StateSpace> get_state_space() const override {
            return state_space_;
          }

          virtual std::shared_ptr<oc::RealVectorControlSpace> get_action_space() const override {
            return action_space_;
          }

          virtual VectorXd get_state() const override {
            return state_;
          }

          virtual std::shared_ptr<System> get_ode_sys() const override {
            return std::make_shared<PendSystem>(pend_.s_);
          }

          virtual MatrixXd sample_grid_refs(int rows, int cols) const override {
            MatrixXd refs(2, rows * cols);

            VectorXd xs = VectorXd::LinSpaced(cols, -M_PI, M_PI);
            VectorXd ys = VectorXd::LinSpaced(rows, -8.0, 8.0);

            for (int i = 0; i < rows; i++) {
              for (int j = 0; j < cols; j++) {
                int idx = (i * cols) + j;
                refs(0, idx) = xs[j];
                refs(1, idx) = ys[i];
              }
            }

            return refs;
          }

          virtual std::tuple<VectorXd, double, bool> step(const VectorXd& control) override {
            if (control.size() != 1)
              throw std::runtime_error("Control passed to inverted pendulum had wrong dimension.");

            double reward;
            std::tie(state_, reward) = pend_.step(control[0]);

            return std::make_tuple(state_, reward, false);
          }

          virtual void render() override {
#ifdef CANNON_BUILD_GRAPHICS
            // Don't start rendering thread until we're asked to render for the first time
            if (!rendering_initialized_) {
              start_rendering_thread_();
              rendering_initialized_ = true;
            }

            // Use atomic to update pendulum theta for rendering thread
            render_theta_.store(state_[0]);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
#endif
          }

          virtual VectorXd reset() override {
            // Use atomic to update pendulum theta for rendering thread
            state_ = pend_.reset();
            return state_;
          }

          virtual void register_ep_reward(float ep_reward) override {
            reward_queue_.push(ep_reward);
          }

          virtual double get_time_step() override {
            return 0.01;
          }

        private:
#ifdef CANNON_BUILD_GRAPHICS
          void start_rendering_thread_() {
            render_thread_ = std::thread([&](){
                // Load pendulum geometry and add to viewer
                DeferredRenderer renderer;
                auto pend_model = renderer.viewer->spawn_model("assets/inverted_pendulum/inverted_pendulum.obj");
                pend_model->set_pos({0.0, 0.0, 0.0});
                pend_model->set_scale(0.1);

                AngleAxisf rot(0.0, Vector3f::UnitZ());
                pend_model->set_rot(rot);

                renderer.render_loop([&] {
                    static CircularBuffer reward_cbuf;
                    static int eps;
                    float value;
                    if (reward_queue_.try_pop(value)) {
                      reward_cbuf.add_point(value);
                      eps += 1;
                    }

                    ImGui::Begin("Rewards");
                    ImGui::Text("Episode %d", eps);
                    ImGui::PlotLines("Rewards", reward_cbuf.data,
                        IM_ARRAYSIZE(reward_cbuf.data), reward_cbuf.offset, NULL,
                        -1500, -100, ImVec2(0, 80));
                    ImGui::End();

                    double theta = render_theta_.load();
                    AngleAxisf rot(theta, Vector3f::UnitZ());
                    pend_model->set_rot(rot);
                    });
                });              
            render_thread_.detach();
          };
#endif

          std::shared_ptr<ob::CompoundStateSpace> state_space_;
          std::shared_ptr<oc::RealVectorControlSpace> action_space_;

          InvertedPendulum pend_;

          bool rendering_initialized_ = false;

          VectorXd state_;
          std::thread render_thread_;

          std::atomic<double> render_theta_;

          BlockingQueue<float> reward_queue_;
      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_ENVS_INVERTED_PENDULUM_H */
