#ifndef CANNON_RESEARCH_PARL_PLANNING_PARL_PLANNER_H
#define CANNON_RESEARCH_PARL_PLANNING_PARL_PLANNER_H 

#include <memory>

#include <ompl/config.h>
#include <ompl/base/Planner.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/ScopedState.h>
#include <ompl/control/PathControl.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/ODESolver.h>

#include <cannon/research/parl/environment.hpp>
#include <cannon/research/parl/hyperparams.hpp>
#include <cannon/research/parl/parl.hpp>

using post_int_func = std::function<void(const ob::State*, const oc::Control*,
    const double, ob::State*)>;

namespace cannon {
  namespace research {
    namespace parl {

      class ParlPlanner {
        public:
          ParlPlanner() = delete;

          ParlPlanner(std::shared_ptr<Environment> env,
              std::shared_ptr<ob::Planner> planner, const Hyperparams& params)
            : planner_(planner), params_(params) {}

          oc::PathControl plan_to_goal(std::shared_ptr<Environment> env,
              post_int_func ompl_post_integration, 
              const ob::ScopedState<ob::StateSpace>& start, 
              const ob::ScopedState<ob::StateSpace>& goal);

          // TODO PARL should be reinitialized each time a new plan is requested.
          // Previous PARL model is analyzed, added to aggregate before creating a new one. 

          std::shared_ptr<Parl> parl_;

        private:
          std::shared_ptr<ob::Planner> planner_;
          Hyperparams params_;

      };

      // Free Functions

      std::shared_ptr<ob::StateSpace> make_error_state_space(const
          std::shared_ptr<Environment> env, double duration);

      MatrixXd make_error_system_refs(oc::PathControl& path, unsigned int state_dim);

      std::vector<double> get_ref_point_times(oc::PathControl& path);

      VectorXd compute_error_state(const VectorXd& ref, const VectorXd& actual, double time);


    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_PLANNING_PARL_PLANNER_H */
