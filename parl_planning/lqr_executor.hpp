#ifndef CANNON_RESEARCH_PARL_PLANNING_EXECUTOR_H
#define CANNON_RESEARCH_PARL_PLANNING_EXECUTOR_H 

#include <chrono>

#include <Eigen/Dense>

#include <ompl/config.h>
#include <ompl/control/PathControl.h>

#include <cannon/utils/class_forward.hpp>

using namespace Eigen;

namespace oc = ompl::control;
namespace ob = ompl::base;

namespace cannon {

  namespace utils {
    CANNON_CLASS_FORWARD(ExperimentWriter);
  }

  namespace physics {
    namespace systems {
      CANNON_CLASS_FORWARD(System);
    }
  }

  namespace research {
    namespace parl {

      CANNON_CLASS_FORWARD(Parl);
      CANNON_CLASS_FORWARD(Environment);

      class LQRExecutor {
        public:
          LQRExecutor() = delete;

          LQRExecutor(EnvironmentPtr env, const VectorXd &goal,
                      double tracking_threshold = 0.5, bool learn = false,
                      bool render = false, int max_overall_timestep = 1e4,
                      int controller_update_interval = 10)
              : tracking_threshold_(tracking_threshold), env_(env), goal_(goal),
                overall_timestep_(0),
                max_overall_timestep_(max_overall_timestep),
                controller_update_interval_(controller_update_interval),
                learn_(learn), render_(render) {
            assert(goal_.size() <= env_->get_state_space()->getDimension());
          }

          /*!
           * \brief Execute a path.
           *
           * \param nominal_sys System path was planned for.
           * \param path The path to execute.
           * \param w ExperimentWriter for recording execution statistics
           * \param seed Seed for random number generator during execution
           *
           * \returns The PARL learner used to execute the path
           */
          ParlPtr execute_path(physics::systems::SystemPtr nominal_sys, oc::PathControl &path,
                               utils::ExperimentWriter &w, int seed);

          /*!
           * \brief Get the number of timesteps executed by this object.
           *
           * \returns Overall timestep.
           */
          int get_overall_timestep();

          /*!
           * \brief Get the maximum number of timesteps to execute.
           *
           * \returns Maximum overall timestep.
           */
          int get_max_overall_timestep();

        private:

          /*!
           * \brief Get state halfway along line between input waypoints.
           *
           * \param w0 First waypoint
           * \param w1 Second waypoint
           *
           * \returns Interpolated waypoint
           */
          VectorXd interp_waypoints(const Ref<const VectorXd>& w0, const Ref<const VectorXd>& w1);

          /*!
           * \brief Make PARL references for executing input path by interpolating waypoints.
           *
           * \param path The path to create reference points along.
           *
           * \returns References along path.
           */
          MatrixXd make_path_refs(oc::PathControl& path);

          /*!
           * \brief Construct PARL agent for executing path. 
           *
           * \param nominal_sys System that path was planned for.
           * \param path Path to construct PARL agent for.
           * \param seed Random seed for PARL.
           *
           * \returns Initialized PARL agent.
           */
          ParlPtr
          construct_parl_agent_for_path(physics::systems::SystemPtr nominal_sys,
                                        oc::PathControl &path, int seed);

          /*!
           * \brief Write executed trajectory to file.
           */
          void write_executed_traj_line_(utils::ExperimentWriter &w, const VectorXd& c);  

          /*!
           * \brief Write planned trajectory to file.
           */
          void write_planned_traj_line_(utils::ExperimentWriter &w, const
              VectorXd& interp_ref, const VectorXd& c);

          /*!
           * \brief Write distances to file.
           */
          void write_distances_line_(utils::ExperimentWriter &w, const VectorXd& s);

          EnvironmentPtr env_; //!< Environment to execute controls in
          double tracking_threshold_; //!< Tracking threshold at which replanning is triggered
          VectorXd goal_; //!< Goal for plans
          int overall_timestep_; //!< Total number of timesteps executed in environment
          int max_overall_timestep_; //!< Total number of timesteps executed in environment
          int controller_update_interval_; //!< Step interval for controller updates when learning

          bool learn_; //!< Whether to execute learning
          bool render_; //!< Whether to render during execution
        
      };


    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_PLANNING_EXECUTOR_H */
