#ifndef CANNON_RESEARCH_PARL_ENVS_DYNAMIC_CAR_H
#define CANNON_RESEARCH_PARL_ENVS_DYNAMIC_CAR_H 

#include <atomic>
#include <thread>

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <cannon/research/parl/environment.hpp>
#include <cannon/physics/systems/dynamic_car.hpp>
#include <cannon/utils/blocking_queue.hpp>
#include <cannon/utils/class_forward.hpp>

#ifdef CANNON_BUILD_GRAPHICS
  #include <cannon/graphics/deferred_renderer.hpp>
  #include <cannon/graphics/geometry/model.hpp>
  #include <cannon/graphics/viewer_3d.hpp>
  #include <cannon/graphics/window.hpp>
  using namespace cannon::graphics;
#endif

using namespace cannon::utils;
using namespace cannon::physics;

namespace cannon {
  namespace research {
    namespace parl {

      class DynamicCarEnvironment : public Environment {
        public:
          DynamicCarEnvironment(const Ref<const VectorXd> &s = VectorXd::Zero(5),
                                const Ref<const VectorXd> &g = VectorXd::Ones(5))
              : kc_(s, g), state_(5), start_(s), goal_(g) {
            auto se2_part = std::make_shared<ob::SE2StateSpace>();
            ob::RealVectorBounds sb(2);
            sb.setLow(-2.0);
            sb.setHigh(2.0);
            se2_part->setBounds(sb);
            se2_part->setup();
            se2_part->setSubspaceWeight(1, 10.0);

            auto deriv_part = std::make_shared<ob::RealVectorStateSpace>(2);
            ob::RealVectorBounds vb(2);
            vb.low[0] = -1.0;
            vb.high[0] = 1.0;
            vb.low[1] = -M_PI * 30 / 180.0;
            vb.high[1] = M_PI * 30 / 180.0;

            deriv_part->setBounds(vb);
            deriv_part->setup();

            state_space_ = std::make_shared<ob::CompoundStateSpace>();
            state_space_->addSubspace(se2_part, 1.0);
            state_space_->addSubspace(deriv_part, 0.3);
            state_space_->setup();

            action_space_ = std::make_shared<oc::RealVectorControlSpace>(state_space_, 2);
            ob::RealVectorBounds ab(2);
            ab.setLow(0, -2.0);
            ab.setLow(1, -M_PI);
            ab.setHigh(0, 2.0);
            ab.setHigh(1, M_PI);
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

          virtual std::shared_ptr<systems::System> get_ode_sys() const override {
            return std::make_shared<systems::DynamicCarSystem>(kc_.s_);
          }

          virtual VectorXd get_state() const override {
            return state_;
          }
          
          virtual MatrixXd sample_grid_refs(int rows, int cols) const override {
            MatrixXd refs(5, rows * cols);

            VectorXd xs = VectorXd::LinSpaced(cols,
                state_space_->getSubspace(1)->as<ob::RealVectorStateSpace>()->getBounds().low[0],
                state_space_->getSubspace(1)->as<ob::RealVectorStateSpace>()->getBounds().high[0]);
            VectorXd ys = VectorXd::LinSpaced(rows,
                state_space_->getSubspace(1)->as<ob::RealVectorStateSpace>()->getBounds().low[1],
                state_space_->getSubspace(1)->as<ob::RealVectorStateSpace>()->getBounds().high[1]);

            for (int i = 0; i < rows; i++) {
              for (int j = 0; j < cols; j++) {
                int idx = (i * cols) + j;
                refs(0, idx) = 0.0;
                refs(1, idx) = 0.0;
                refs(2, idx) = xs[i];
                refs(3, idx) = ys[i];
                refs(4, idx) = 0.0;
              }
            }

            // TODO Also sample other dimensions?

            return refs;
          }

          virtual std::tuple<VectorXd, double, bool> step(const VectorXd& control) override {
            if (control.size() != 2)
              throw std::runtime_error("Control passed to dynamic car had wrong dimension.");

            double reward;
            std::tie(state_, reward) = kc_.step(control[0], control[1]);

            state_[0] = std::max(-2.0, std::min(2.0, state_[0]));
            state_[1] = std::max(-2.0, std::min(2.0, state_[1]));
            state_[2] = std::atan2(std::sin(state_[2]), std::cos(state_[2]));
            state_[3] = std::max(-1.0, std::min(1.0, state_[3]));
            state_[4] = std::max(-M_PI * 30 / 180.0, std::min(M_PI * 30 / 180.0, state_[4]));

            kc_.reset(state_);

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
            assert(s.size() == 5);
            state_ = kc_.reset(s);
            return state_;
          }

          virtual void register_ep_reward(float ep_reward) override {
            reward_queue_.push(ep_reward);
          }

          virtual double get_time_step() override {
            return kc_.time_step;
          }


          // TODO Generalize to interface?
          MatrixX2d get_state_space_bounds() {
            MatrixX2d ret_bounds = MatrixX2d::Zero(5, 2);
            auto ss_bounds = state_space_->getSubspace(0)->as<ob::SE2StateSpace>()->getBounds();
            auto deriv_bounds = state_space_->getSubspace(1)->as<ob::RealVectorStateSpace>()->getBounds();

            ret_bounds(0, 0) = ss_bounds.low[0];
            ret_bounds(0, 1) = ss_bounds.high[0];
            ret_bounds(1, 0) = ss_bounds.low[1];
            ret_bounds(1, 1) = ss_bounds.high[1];
            ret_bounds(2, 0) = -M_PI;
            ret_bounds(2, 1) = M_PI;
            ret_bounds(3, 0) = deriv_bounds.low[0];
            ret_bounds(3, 1) = deriv_bounds.high[0];
            ret_bounds(4, 0) = deriv_bounds.low[1];
            ret_bounds(4, 1) = deriv_bounds.high[1];

            return ret_bounds;
          }

        private:
#ifdef CANNON_BUILD_GRAPHICS
          void start_rendering_thread_() {
            render_thread_ = std::thread([&](){
                // Load pendulum geometry and add to viewer
                DeferredRenderer renderer;

                auto car_model = renderer.viewer->spawn_model("assets/car/lowpoly_car_revised.obj");
                car_model->set_pos({0.0, 0.0, 0.0});
                car_model->set_scale(0.01);

                auto goal_sphere_model = renderer.viewer->spawn_model("assets/sphere/sphere.obj");
                float goal_x = goal_[0];
                float goal_y = goal_[1];
                goal_sphere_model->set_pos({goal_x, 0.0, goal_y});
                goal_sphere_model->set_scale(0.05);

                // Wall geometry
                auto north_wall = renderer.viewer->spawn_plane();
                north_wall->set_pos({0.0, 0.0, 2.0});
                AngleAxisf north_rot(M_PI, Vector3f::UnitY());
                north_wall->set_rot(north_rot);
                north_wall->set_scale(5.0);

                auto south_wall = renderer.viewer->spawn_plane();
                south_wall->set_pos({0.0, 0.0, -2.0});
                AngleAxisf south_rot(0, Vector3f::UnitY());
                south_wall->set_rot(south_rot);
                south_wall->set_scale(5.0);

                auto east_wall = renderer.viewer->spawn_plane();
                east_wall->set_pos({2.0, 0.0, 0.0});
                AngleAxisf east_rot(-M_PI / 2, Vector3f::UnitY());
                east_wall->set_rot(east_rot);
                east_wall->set_scale(5.0);

                auto west_wall = renderer.viewer->spawn_plane();
                west_wall->set_pos({-2.0, 0.0, 0.0});
                AngleAxisf west_rot(M_PI / 2, Vector3f::UnitY());
                west_wall->set_rot(west_rot);
                west_wall->set_scale(5.0);

                auto north_back_wall = renderer.viewer->spawn_plane();
                north_back_wall->set_pos({0.0, 0.0, 2.0});
                AngleAxisf north_back_rot(0.0, Vector3f::UnitY());
                north_back_wall->set_rot(north_back_rot);
                north_back_wall->set_scale(5.0);

                auto south_back_wall = renderer.viewer->spawn_plane();
                south_back_wall->set_pos({0.0, 0.0, -2.0});
                AngleAxisf south_back_rot(M_PI, Vector3f::UnitY());
                south_back_wall->set_rot(south_back_rot);
                south_back_wall->set_scale(5.0);

                auto east_back_wall = renderer.viewer->spawn_plane();
                east_back_wall->set_pos({2.0, 0.0, 0.0});
                AngleAxisf east_back_rot(M_PI / 2, Vector3f::UnitY());
                east_back_wall->set_rot(east_back_rot);
                east_back_wall->set_scale(5.0);

                auto west_back_wall = renderer.viewer->spawn_plane();
                west_back_wall->set_pos({-2.0, 0.0, 0.0});
                AngleAxisf west_back_rot(-M_PI / 2, Vector3f::UnitY());
                west_back_wall->set_rot(west_back_rot);
                west_back_wall->set_scale(5.0);

                // Set camera position
                renderer.viewer->c.set_pos({-3.0, 7.0, -3.0});
                Vector3f dir = renderer.viewer->c.get_pos() - Vector3f::Zero();
                dir.normalize();
                renderer.viewer->c.set_direction(dir);

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
                        -1000, -0, ImVec2(0, 80));
                    ImGui::End();

                    double x = render_x_.load();
                    double y = render_y_.load();
                    double th = render_th_.load();
                    Vector3f pos;
                    pos[0] = x;
                    pos[1] = 0.0;
                    pos[2] = y;
                    car_model->set_pos(pos);

                    AngleAxisf rot(-th + (M_PI / 2.0), Vector3f::UnitY());
                    car_model->set_rot(rot);
                    
                    });
                });              
            render_thread_.detach();
          }
#endif

          std::shared_ptr<ob::CompoundStateSpace> state_space_;
          std::shared_ptr<oc::RealVectorControlSpace> action_space_;

          systems::DynamicCar kc_;

          VectorXd state_;
          VectorXd start_;
          VectorXd goal_;

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

#endif /* ifndef CANNON_RESEARCH_PARL_ENVS_DYNAMIC_CAR_H */
