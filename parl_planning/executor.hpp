#ifndef CANNON_RESEARCH_PARL_PLANNING_EXECUTOR_H
#define CANNON_RESEARCH_PARL_PLANNING_EXECUTOR_H 

#include <ompl/config.h>
#include <ompl/control/PathControl.h>

#include <cannon/research/parl/environment.hpp>
#include <cannon/research/parl_planning/parl_planner.hpp>

namespace cannon {
  namespace research {
    namespace parl {

      void noop_post_integration(const ob::State*, const oc::Control*, const double, ob::State*); 

      class Executor {
        public:
          Executor() = delete;

          Executor(std::shared_ptr<Environment> env,
              std::shared_ptr<ParlPlanner> planner, const VectorXd& goal,
              double tracking_threshold=0.5) :
            tracking_threshold_(tracking_threshold), env_(env),
            planner_(planner), goal_(goal) {
              assert(goal_.size() == env_->get_state_space()->getDimension());
            }

          void execute(post_int_func ompl_post_integration=noop_post_integration);
          void execute_path(oc::PathControl& path);
          void execute_control_for_duration(const VectorXd& s, const VectorXd& next_s, 
              const VectorXd& c, double start_time,
              std::chrono::milliseconds control_dur);

        private:
          double tracking_threshold_;
          std::shared_ptr<Environment> env_;
          std::shared_ptr<ParlPlanner> planner_;
          VectorXd goal_;

          VectorXd get_coords_from_ompl_state(ob::State*);
        
      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_PLANNING_EXECUTOR_H */
