#include <ompl/base/ProjectionEvaluator.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/control/ODESolver.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/base/goals/GoalRegion.h>

#include <cannon/research/parl_planning/lqr_control_space.hpp>
#include <cannon/research/parl_planning/lqr_state_propagator.hpp>

namespace ob = ompl::base;
namespace oc = ompl::control;

using namespace cannon::research::parl;

static constexpr double length_inv = 1.0, mass = 1.0;

class DynamicCarProjector : public ob::ProjectionEvaluator
{
public:
    DynamicCarProjector(const ob::StateSpace *space) : ob::ProjectionEvaluator(space)
    {
    }
    unsigned int getDimension() const override
    {
        return 3u;
    }
    void project(const ob::State *state, Eigen::Ref<Eigen::VectorXd> projection) const override
    {
        auto cstate = state->as<ob::CompoundStateSpace::StateType>();
        projection[0] = cstate->as<ob::SE2StateSpace::StateType>(0)->getX();
        projection[1] = cstate->as<ob::SE2StateSpace::StateType>(0)->getY();
        projection[2] = cstate->as<ob::SE2StateSpace::StateType>(0)->getYaw();
    }
};

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

      //auto current_geom = st->as<ob::CompoundStateSpace::StateType>()
      //                        ->as<ob::SE2StateSpace::StateType>(0);
      //auto goal_geom = se2->allocState()->as<ob::SE2StateSpace::StateType>();
      //goal_geom->setX(geom_goal_[0]);
      //goal_geom->setY(geom_goal_[1]);
      //goal_geom->setYaw(geom_goal_[2]);
      //double distance = se2->distance(current_geom, goal_geom);
      //se2->freeState(goal_geom);

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

void dynamic_car_ODE(const oc::ODESolver::StateType &q,
                     const Ref<const Vector2d> &u,
                     oc::ODESolver::StateType &qdot) {
  qdot.resize(q.size(), 0);

  qdot[0] = q[3] * cos(q[2]);
  qdot[1] = q[3] * sin(q[2]);
  qdot[2] = q[3] * mass * length_inv * tan(q[4]);
  qdot[3] = u[0];
  qdot[4] = u[1];
}

void dynamic_car_LQR_ode(const oc::ControlSpacePtr &cspace,
                         const oc::ODESolver::StateType &q,
                         const oc::Control *control,
                         oc::ODESolver::StateType &qdot) {
  Eigen::VectorXd u(2);
  cspace->as<LQRControlSpace>()->compute_u_star(control, q, u);
  dynamic_car_ODE(q, u, qdot);
}

void dynamic_car_linearization(const oc::ODESolver::StateType &q,
                               const oc::ODESolver::StateType &,
                               Eigen::Ref<Eigen::MatrixXd> A,
                               Eigen::Ref<Eigen::MatrixXd> B) {
  double c = cos(q[2]), s = sin(q[2]), c4 = cos(q[4]);
  A = 1e-3 * Eigen::MatrixXd::Identity(5, 5);
  A(0, 2) = -q[3] * s;
  A(0, 3) = c;
  A(1, 2) = q[3] * c;
  A(1, 3) = s;
  A(2, 3) = mass * length_inv * tan(q[4]);
  A(2, 4) = q[3] * mass * length_inv / (c4 * c4);

  B = Eigen::MatrixXd::Zero(5, 2);
  B(3, 0) = B(4, 1) = 1.;
}

void dynamic_car_postintegration(ob::StateSpacePtr space, const ob::State *, const oc::Control *, const double, ob::State *result)
{
    // Normalize orientation between 0 and 2*pi
    const ob::SO2StateSpace* SO2 = space->as<ob::CompoundStateSpace>()
                                      ->as<ob::SE2StateSpace>(0)
                                      ->as<ob::SO2StateSpace>(1);
    SO2->enforceBounds(result->as<ob::CompoundStateSpace::StateType>()
                           ->as<ob::SE2StateSpace::StateType>(0)
                           ->as<ob::SO2StateSpace::StateType>(1));
}

ob::StateSpacePtr make_dynamic_car_state_space() {
  auto state_space(std::make_shared<ob::CompoundStateSpace>());
  auto se2 = std::make_shared<ob::SE2StateSpace>();

  se2->setSubspaceWeight(1, 10.);
  ob::RealVectorBounds bounds(2);
  bounds.low[0] = -2.;
  bounds.high[0] = 2.;
  bounds.low[1] = -2.;
  bounds.high[1] = 2.;
  se2->setBounds(bounds);

  state_space->addSubspace(se2, 1.);
  state_space->addSubspace(std::make_shared<ob::RealVectorStateSpace>(2), 0.3);
  state_space->lock();

  bounds.low[0] = -1.;
  bounds.high[0] = 1.;
  bounds.low[1] = -M_PI * 30.0 / 180.0;
  bounds.high[1] = M_PI * 30.0 / 180.0;
  state_space->as<ob::RealVectorStateSpace>(1)->setBounds(bounds);

  return state_space;
}

oc::ControlSpacePtr make_dynamic_car_control_space(ob::StateSpacePtr state_space) {
  auto control_space(std::make_shared<LQRControlSpace>(state_space, 2));
  
  ob::RealVectorBounds bounds(2);
  bounds.low[0] = -0.5;
  bounds.high[0] = 0.5;
  bounds.low[1] = -M_PI * 2.0 / 180.0;
  bounds.high[1] = M_PI * 2.0 / 180.0;

  control_space->setBounds(bounds);

  return control_space;
}


int main() {
  // construct the state space we are planning in
  auto space = make_dynamic_car_state_space();
  auto cspace = make_dynamic_car_control_space(space);

  space->registerDefaultProjection(std::make_shared<DynamicCarProjector>(space.get()));

  // define a simple setup class
  oc::SimpleSetup ss(cspace);

  // set state validity checking for this space
  ss.setStateValidityChecker([&](const ob::State *state) { return space->satisfiesBounds(state); });

  // set LQR state propagator
  auto ode = [cspace](const oc::ODESolver::StateType &q,
                      const oc::Control *control,
                      oc::ODESolver::StateType &qdot) {
    dynamic_car_LQR_ode(cspace, q, control, qdot);
  };

  auto post_integration = [&](const ob::State * state, const oc::Control *control, const double duration, ob::State *result) {
    dynamic_car_postintegration(space, state, control, duration, result);
  };

  auto propagator = std::make_shared<LQRStatePropagator>(
      ss.getSpaceInformation(), ode, dynamic_car_linearization,
      post_integration);

  // Eigen::Matrix<double, 1, 1> R(50.);
  // propagator->setControlCostMatrix(R);
  ss.setStatePropagator(propagator);
  ss.getSpaceInformation()->setPropagationStepSize(0.1);

  // create start & goal states
  ob::ScopedState<ob::CompoundStateSpace> start(space), goal(space);
  ob::SE2StateSpace::StateType& pose = *start->as<ob::SE2StateSpace::StateType>(0);
  ob::RealVectorStateSpace::StateType& vel = *start->as<ob::RealVectorStateSpace::StateType>(1);

  pose.setX(0.0);
  pose.setY(0.0);
  pose.setYaw(0.0);
  vel.values[0] = 0.0;
  vel.values[1] = 0.0;

  ob::SE2StateSpace::StateType& goal_pose = *goal->as<ob::SE2StateSpace::StateType>(0);
  ob::RealVectorStateSpace::StateType& goal_vel = *goal->as<ob::RealVectorStateSpace::StateType>(1);
  goal_pose.setX(1.0);
  goal_pose.setY(1.0);
  goal_pose.setYaw(0.0);
  goal_vel.values[0] = 0.0;
  goal_vel.values[1] = 0.0;
  
  // set the start and goal states
  //ss.setStartAndGoalStates(start, goal, 0.05);
  ss.setStartState(start);

  Vector3d geom_goal(2);
  geom_goal << 1.0, 1.0;
  ss.setGoal(std::make_shared<SubsetGoalRegion>(ss.getSpaceInformation(), geom_goal));

  auto planner = std::make_shared<oc::RRT>(ss.getSpaceInformation());
  //auto planner = std::make_shared<oc::SST>(ss.getSpaceInformation());
  ss.setPlanner(planner);
  ss.setup();
  ss.print();

  ob::PlannerStatus solved = ss.solve(10.0);

  if (solved && ss.haveExactSolutionPath())
  {
    std::cout << "Found solution:" << std::endl;
    ss.getSpaceInformation()->setPropagationStepSize(0.1);
    auto path(ss.getSolutionPath());
    //path.interpolate();
    path.printAsMatrix(std::cout);
  }
  else
    std::cout << "No solution found" << std::endl;

}
