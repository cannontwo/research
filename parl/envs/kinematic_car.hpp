#ifndef CANNON_RESEARCH_PARL_ENVS_KINEMATIC_CAR_H
#define CANNON_RESEARCH_PARL_ENVS_KINEMATIC_CAR_H 

#include <atomic>
#include <thread>

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <cannon/research/parl/environment.hpp>
#include <cannon/physics/systems/kinematic_car.hpp>
#include <cannon/utils/blocking_queue.hpp>

#ifdef CANNON_BUILD_GRAPHICS
  #include <cannon/graphics/deferred_renderer.hpp>
  #include <cannon/graphics/geometry/model.hpp>
  using namespace cannon::graphics;
#endif

using namespace cannon::utils;
using namespace cannon::physics;

namespace cannon {
  namespace research {
    namespace parl {

      class KinematicCarEnvironment : public Environment {
        public:

          KinematicCarEnvironment(const Vector3d& s = Vector3d::Zero(), const
              Vector3d& g = Vector3d::Ones()) : kc_(s, g), state_(3), start_(s), goal_(g)
          {
            state_space_ = std::make_shared<ob::SE2StateSpace>();
            ob::RealVectorBounds sb(2);
            sb.setLow(-2.0);
            sb.setHigh(2.0);
            state_space_->setBounds(sb);
            state_space_->setup();

            action_space_ = std::make_shared<oc::RealVectorControlSpace>(state_space_, 2);
            ob::RealVectorBounds ab(2);
            ab.setLow(0, -1.0);
            ab.setLow(1, -M_PI * 30.0 / 180.0);
            ab.setHigh(0, 1.0);
            ab.setHigh(1, M_PI * 30.0 / 180.0);
            action_space_->setBounds(ab);
            action_space_->setup();

            // TODO Obstacle geometry? (see OMPL.app for inspiration)
          }

          virtual std::shared_ptr<ob::StateSpace> get_state_space() const override {
            return state_space_;
          }

          virtual std::shared_ptr<oc::RealVectorControlSpace> get_action_space() const override {
            return action_space_;
          }

          virtual std::shared_ptr<System> get_ode_sys() const override {
            return std::make_shared<KinCarSystem>(kc_.s_);
          }

          virtual VectorXd get_state() const override {
            return state_;
          }
          
          virtual MatrixXd sample_grid_refs(int rows, int cols) const override {
            MatrixXd refs(3, rows * cols);

            VectorXd xs = VectorXd::LinSpaced(cols,
                state_space_->getBounds().low[0],
                state_space_->getBounds().high[0]);
            VectorXd ys = VectorXd::LinSpaced(rows,
                state_space_->getBounds().low[1],
                state_space_->getBounds().high[1]);

            for (int i = 0; i < rows; i++) {
              for (int j = 0; j < cols; j++) {
                int idx = (i * cols) + j;
                refs(0, idx) = xs[j];
                refs(1, idx) = ys[i];
                refs(2, idx) = 0.0;
              }
            }

            // TODO Also sample theta dimension?

            return refs;
          }

          virtual std::tuple<VectorXd, double, bool> step(VectorXd control) override {
            if (control.size() != 2)
              throw std::runtime_error("Control passed to kinematic car had wrong dimension.");

            double reward;
            std::tie(state_, reward) = kc_.step(control[0], control[1]);

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
            render_x_.store(state_[0]);
            render_y_.store(state_[1]);
            render_th_.store(state_[2]);
#endif
            std::this_thread::sleep_for(std::chrono::duration<double, std::ratio<1>>(kc_.time_step));
          }

          virtual VectorXd reset() override {
            // Use atomic to update pendulum theta for rendering thread
            state_ = kc_.reset();
            return state_;
          }

          VectorXd reset(const VectorXd& s) {
            assert(s.size() == 3);
            state_ = kc_.reset(s);
            return state_;
          }

          virtual void register_ep_reward(float ep_reward) override {
            reward_queue_.push(ep_reward);
          }

          double get_time_step() {
            return kc_.time_step;
          }

        private:
#ifdef CANNON_BUILD_GRAPHICS
          void start_rendering_thread_() {
            render_thread_ = std::thread([&](){
                // Load pendulum geometry and add to viewer
                DeferredRenderer renderer;

                // TODO Make actual car model
                auto car_model = renderer.viewer.spawn_model("assets/car/car.obj");
                car_model->set_pos({0.0, 0.0, 0.0});
                car_model->set_scale(0.1);

                auto goal_sphere_model = renderer.viewer.spawn_model("assets/sphere/sphere.obj");
                float goal_x = goal_[0];
                float goal_y = goal_[1];
                goal_sphere_model->set_pos({goal_x, 0.0, goal_y});
                goal_sphere_model->set_scale(0.05);

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

                    double x = render_x_.load();
                    double y = render_y_.load();
                    double th = render_th_.load();
                    Vector3f pos;
                    pos[0] = x;
                    pos[1] = 0.0;
                    pos[2] = y;
                    car_model->set_pos(pos);

                    AngleAxisf rot(th, Vector3f::UnitY());
                    car_model->set_rot(rot);
                    
                    });
                });              
            render_thread_.detach();
          }
#endif

          std::shared_ptr<ob::SE2StateSpace> state_space_;
          std::shared_ptr<oc::RealVectorControlSpace> action_space_;

          systems::KinematicCar kc_;

          VectorXd state_;
          Vector3d start_;
          Vector3d goal_;

          bool rendering_initialized_ = false;

          std::thread render_thread_;
          std::atomic<double> render_x_;
          std::atomic<double> render_y_;
          std::atomic<double> render_th_;

          BlockingQueue<float> reward_queue_;
      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_ENVS_KINEMATIC_CAR_H */
