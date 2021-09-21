#include <ompl/config.h>
#include <ompl/control/ODESolver.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/base/goals/GoalRegion.h>

#include <cannon/research/parl/envs/dynamic_car.hpp>
#include <cannon/control/pid.hpp>
#include <cannon/math/interp.hpp>
#include <cannon/geom/graph.hpp>
#include <cannon/geom/trajectory.hpp>
#include <cannon/plot/plotter.hpp>
#include <cannon/physics/systems/dynamic_car.hpp>

using namespace cannon::research::parl;
using namespace cannon::control;
using namespace cannon::math;
using namespace cannon::geom;
using namespace cannon::plot;
using namespace cannon::physics::systems;

unsigned int node_id(unsigned int cols, unsigned int i, unsigned int j) {
  return j * cols + i;
}

Graph construct_grid(unsigned int rows, unsigned int cols) {
  Graph g;

  for (unsigned int i = 0; i < cols; ++i) {
    for (unsigned int j = 0; j < rows; ++j) {
      unsigned int current = node_id(cols, i, j);

      if (i > 0)
        g.add_edge(current, node_id(cols, i - 1, j), 1.0);

      if (i < cols - 1)
        g.add_edge(current, node_id(cols, i + 1, j), 1.0);

      if (j > 0)
        g.add_edge(current, node_id(cols, i, j - 1), 1.0);

      if (j < rows - 1)
        g.add_edge(current, node_id(cols, i, j + 1), 1.0);
    }
  }

  return g;
}

class SubsetGoalRegion : public ob::GoalRegion {
  public:
    SubsetGoalRegion(const ob::SpaceInformationPtr &si,
                     const Ref<const Vector2d> &geom_goal)
        : ob::GoalRegion(si), geom_goal_(geom_goal) {
      setThreshold(0.1);
    }

    virtual double distanceGoal(const ob::State *st) const override {

      auto se2 =
        si_->getStateSpace()->as<ob::CompoundStateSpace>()->as<ob::SE2StateSpace>(0);
      auto r2 = se2->as<ob::RealVectorStateSpace>(0);

      auto current_geom = st->as<ob::CompoundStateSpace::StateType>()
                              ->as<ob::SE2StateSpace::StateType>(0)
                              ->as<ob::RealVectorStateSpace::StateType>(0);
      auto goal_geom = r2->allocState()->as<ob::RealVectorStateSpace::StateType>();
      goal_geom->values[0] = geom_goal_[0];
      goal_geom->values[1] = geom_goal_[1];
      double distance = r2->distance(current_geom, goal_geom);
      r2->freeState(goal_geom);

      return distance;
    }

  private:
    Vector2d geom_goal_;
};

ControlledTrajectory plan_sst_traj() {

  auto env = std::make_shared<DynamicCarEnvironment>();
  env->reset();

  auto sys = DynamicCarSystem();

  oc::SimpleSetup ss(env->get_action_space());
  oc::SpaceInformationPtr si = ss.getSpaceInformation();

  auto ode_solver = std::make_shared<oc::ODEBasicSolver<>>(
      si, [&](const oc::ODESolver::StateType &q, const oc::Control *control,
              oc::ODESolver::StateType &qdot) {
        return sys.ompl_ode_adaptor(q, control, qdot);
      });

  si->setStatePropagator(oc::ODESolver::getStatePropagator(ode_solver));

  si->setStateValidityChecker([si](const ob::State *state) {
    // TODO Make real validity checker with obstacle geometry
    return si->satisfiesBounds(state);
  });

  ob::ScopedState<ob::CompoundStateSpace> start(env->get_state_space());
  start->as<ob::SE2StateSpace::StateType>(0)->setX(0.0);
  start->as<ob::SE2StateSpace::StateType>(0)->setY(0.0);
  start->as<ob::SE2StateSpace::StateType>(0)->setYaw(0.0);
  start->as<ob::RealVectorStateSpace::StateType>(1)->values[0] = 0.0;
  start->as<ob::RealVectorStateSpace::StateType>(1)->values[1] = 0.0;

  ss.setStartState(start);
  VectorXd geom_goal(2);
  geom_goal << 1.0, 1.0;
  ss.setGoal(std::make_shared<SubsetGoalRegion>(ss.getSpaceInformation(), geom_goal));
  
  ss.getSpaceInformation()->setPropagationStepSize(env->get_time_step() * 10);
  log_info("Propagation step size is",
           ss.getSpaceInformation()->getPropagationStepSize());

  auto planner = std::make_shared<oc::SST>(ss.getSpaceInformation());
  ss.setPlanner(planner);
  ss.setup();

  double plan_time = 10.0;
  bool have_plan = false;

  ControlledTrajectory traj;

  while (!have_plan && plan_time < 60.0) {
    log_info("Planning for", plan_time, "seconds");
    ob::PlannerStatus solved = ss.solve(plan_time);
    if (solved) {
      if (ss.haveExactSolutionPath()) {
        std::cout << "Found solution:" << std::endl;

        // Handle planned LQR controls 
        oc::PathControl ompl_path = ss.getSolutionPath();

        // TODO Extract my version of trajectory
        double time = 0.0;
        auto states = ompl_path.getStates();
        auto controls = ompl_path.getControls();
        auto durations = ompl_path.getControlDurations();
        for (unsigned int i = 0; i < controls.size(); i++) {
          std::vector<double> state_vec(5);
          std::vector<double> control_vec(2);

          VectorXd state = VectorXd::Zero(5);
          VectorXd control = VectorXd::Zero(2);

          si->getStateSpace()->copyToReals(state_vec, states[i]);
          for (unsigned int idx = 0; idx < 5; ++idx)
            state[idx] = state_vec[idx];
          
          control[0] = controls[i]->as<oc::RealVectorControlSpace::ControlType>()->values[0];
          control[1] = controls[i]->as<oc::RealVectorControlSpace::ControlType>()->values[1];

          traj.push_back(state, control, time);
          time += durations[i];
        }

        return traj;
      } else {
        plan_time *= 2.0;
      }
    } else {
      plan_time *= 2.0;
    }
  }

  return traj;
}

std::pair<std::vector<Vector2d>, std::vector<Vector2d>>
plot_pid_traj(const ControlledTrajectory& traj, const Ref<const MatrixXd> &kp,
              const Ref<const MatrixXd> &ki, const Ref<const MatrixXd> &kd) {
  auto env = std::make_shared<DynamicCarEnvironment>(VectorXd::Zero(5), VectorXd::Ones(5));
  PidController controller(2, 2, env->get_time_step());

  controller.proportional_gain() = kp;
  controller.integral_gain() = ki;
  controller.derivative_gain() = kd;

  double time = 0.0;
  VectorXd state = env->reset(VectorXd::Zero(5));
  std::vector<Vector2d> executed, controls;
  for (unsigned int i = 0; i < 100 * traj.length(); ++i) {
    executed.push_back(state.head(2));

    controller.ref() = traj(time).first.head(2);
    VectorXd pid_action = controller.get_control(state.head(2));

    VectorXd combined_action = pid_action + traj(time).second;

    controls.push_back(env->get_constrained_control(combined_action));

    double reward;
    bool done;
    std::tie(state, reward, done) = env->step(combined_action);

    time += env->get_time_step();
  }

  executed.push_back(state.head(2));
  return std::make_pair(executed, controls);
}

int main() {
  auto traj = plan_sst_traj();
  traj.save("logs/sst_dynamic_car_plan.h5");

  MatrixXf Kp(2, 2), Ki(2, 2), Kd(2, 2);

  Kp << 0.0, 0.0,
        0.0, 0.0;

  Ki << 0.0, 0.0,
        0.0, 0.0;

  Kd << 0.0, 0.0,
        0.0, 0.0;

  Plotter plotter;
  plotter.render([&]() {
    bool changed = false;
    static int current_mode = 0;
    if (ImGui::BeginMainMenuBar()) {
      if (ImGui::BeginMenu("PID")) {
        changed = changed || ImGui::InputFloat("Kp(0, 0)", &Kp(0, 0), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Kp(0, 1)", &Kp(0, 1), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Kp(1, 0)", &Kp(1, 0), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Kp(1, 1)", &Kp(1, 1), -100.0, 100.0);

        changed = changed || ImGui::InputFloat("Ki(0, 0)", &Ki(0, 0), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Ki(0, 1)", &Ki(0, 1), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Ki(1, 0)", &Ki(1, 0), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Ki(1, 1)", &Ki(1, 1), -100.0, 100.0);

        changed = changed || ImGui::InputFloat("Kd(0, 0)", &Kd(0, 0), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Kd(0, 1)", &Kd(0, 1), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Kd(1, 0)", &Kd(1, 0), -100.0, 100.0);
        changed = changed || ImGui::InputFloat("Kd(1, 1)", &Kd(1, 1), -100.0, 100.0);

        static const char* modes[] = {"Full", "X", "Y", "Controls"};
        changed = changed || ImGui::Combo("Display Mode", &current_mode, modes, IM_ARRAYSIZE(modes));
        
        ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }

    if (changed) {
      plotter.clear();
      auto [pts, controls] = plot_pid_traj(traj, Kp.cast<double>(), Ki.cast<double>(),
                                    Kd.cast<double>());

      
      double time = 0.0;
      std::vector<Vector2d> times, pts_oned;
      switch (current_mode) {
        case 0:
          plotter.plot([&](double t){return traj(t).first;}, 200, 0.0, traj.length());
          plotter.plot(pts);
          break;
        case 1:
          plotter.plot([&](double t) { return traj(t).first[0]; }, 200, 0.0,
                       traj.length());
          for (auto& p : pts) {
            Vector2d tmp;
            tmp << time, p[0];
            pts_oned.push_back(tmp);
            time += 0.01;
          }
          plotter.plot(pts_oned);
          break;
        case 2:
          plotter.plot([&](double t) { return traj(t).first[1]; }, 200, 0.0,
                       traj.length());
          for (auto& p : pts) {
            Vector2d tmp;
            tmp << time, p[1];
            pts_oned.push_back(tmp);
            time += 0.01;
          }
          plotter.plot(pts_oned);
          break;
        case 3:
          for (auto& p : controls) {
            Vector2d tmp;
            tmp << time, p[0];
            pts_oned.push_back(tmp);
            time += 0.01;
          }
          plotter.plot(pts_oned);
          pts_oned.clear();
          time = 0.0;
          for (auto& p : controls) {
            Vector2d tmp;
            tmp << time, p[1];
            pts_oned.push_back(tmp);
            time += 0.01;
          }
          plotter.plot(pts_oned);
          break;
      }
    }


  });
}
