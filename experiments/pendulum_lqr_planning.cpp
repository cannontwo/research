#include <ompl/base/ProjectionEvaluator.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/control/ODESolver.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <cannon/research/parl_planning/lqr_control_space.hpp>
#include <cannon/research/parl_planning/lqr_state_propagator.hpp>

namespace ob = ompl::base;
namespace oc = ompl::control;

using namespace cannon::research::parl;

static constexpr double b = 0.1, g = 9.81;

class PendulumProjector : public ob::ProjectionEvaluator
{
public:
    PendulumProjector(const ob::StateSpace *space) : ob::ProjectionEvaluator(space)
    {
    }
    unsigned int getDimension() const override
    {
        return 2u;
    }
    void project(const ob::State *state, Eigen::Ref<Eigen::VectorXd> projection) const override
    {
        auto cstate = state->as<ob::CompoundStateSpace::StateType>();
        projection[0] = cstate->as<ob::SO2StateSpace::StateType>(0)->value;
        projection[1] = cstate->as<ob::RealVectorStateSpace::StateType>(1)->values[0];
    }
};


void pendulumODE(const oc::ODESolver::StateType &q, double u, oc::ODESolver::StateType &qdot)
{
    qdot[0] = q[1];
    qdot[1] = u - b * q[1] - g * cos(q[0]);
}
void pendulumLQRODE(const oc::ControlSpacePtr &cspace, const oc::ODESolver::StateType &q, const oc::Control *control, oc::ODESolver::StateType &qdot)
{
    Eigen::VectorXd u(1);
    cspace->as<LQRControlSpace>()->compute_u_star(control, q, u);
    pendulumODE(q, u[0], qdot);
}
void pendulumRandomODE(const oc::ODESolver::StateType &q, const oc::Control *control, oc::ODESolver::StateType &qdot)
{
    pendulumODE(q, control->as<oc::RealVectorControlSpace::ControlType>()->values[0], qdot);
}


void pendulumLinearization(const oc::ODESolver::StateType &q, const oc::ODESolver::StateType &,
                           Eigen::Ref<Eigen::MatrixXd> A, Eigen::Ref<Eigen::MatrixXd> B)
{
    A << 0, 1, g * sin(q[0]), -b;
    B << 0, 1;
}

void pendulumPostIntegration(const ob::State *, const oc::Control *, const double, ob::State *result)
{
    // Normalize orientation between 0 and 2*pi
    static const ob::SO2StateSpace SO2;
    SO2.enforceBounds(result->as<ob::CompoundStateSpace::StateType>()->as<ob::SO2StateSpace::StateType>(0));
}


int main() {
  // construct the state space we are planning in
  auto orientation(std::make_shared<ob::SO2StateSpace>());
  auto rot_velocity(std::make_shared<ob::RealVectorStateSpace>(1));
  ob::StateSpacePtr space = orientation + rot_velocity;
  space->registerDefaultProjection(std::make_shared<PendulumProjector>(space.get()));

  ob::RealVectorBounds bounds(1);
  bounds.setLow(-10);
  bounds.setHigh(10);
  rot_velocity->setBounds(bounds);

  oc::ControlSpacePtr cspace;
  ob::RealVectorBounds cbounds(1);
  cbounds.setLow(-3);
  cbounds.setHigh(3);

  auto lqrspace = std::make_shared<LQRControlSpace>(space, 1);
  lqrspace->setBounds(cbounds);
  cspace = lqrspace;

  // define a simple setup class
  oc::SimpleSetup ss(cspace);

  // set state validity checking for this space
  ss.setStateValidityChecker([](const ob::State *) { return true; });

  // set LQR state propagator
  auto ode = [cspace](const oc::ODESolver::StateType &q,
                      const oc::Control *control,
                      oc::ODESolver::StateType &qdot) {
    pendulumLQRODE(cspace, q, control, qdot);
  };

  auto propagator = std::make_shared<LQRStatePropagator>(
      ss.getSpaceInformation(), ode, pendulumLinearization,
      pendulumPostIntegration);

  // Eigen::Matrix<double, 1, 1> R(50.);
  // propagator->setControlCostMatrix(R);
  ss.setStatePropagator(propagator);
  ss.getSpaceInformation()->setPropagationStepSize(0.1);

  // create start & goal states
  ob::ScopedState<> start(space), goal(space);
  start[0] = -M_PI/2.0;
  start[1] = 0.;
  goal[0] = M_PI/2.0;
  goal[1] = 0.;
  // set the start and goal states
  ss.setStartAndGoalStates(start, goal, 0.05);

  auto planner = std::make_shared<oc::RRT>(ss.getSpaceInformation());
  ss.setPlanner(planner);
  ss.setup();
  ss.print();

  ob::PlannerStatus solved = ss.solve(10.0);

  if (solved && ss.haveExactSolutionPath())
  {
    std::cout << "Found solution:" << std::endl;
    ss.getSpaceInformation()->setPropagationStepSize(0.1);
    auto path(ss.getSolutionPath());
    path.interpolate();
    path.asGeometric().printAsMatrix(std::cout);
  }
  else
    std::cout << "No solution found" << std::endl;

}
