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

  namespace research {
    namespace parl {

      CANNON_CLASS_FORWARD(Parl);
      CANNON_CLASS_FORWARD(Environment);

      void noop_post_integration(const ob::State*, const oc::Control*, const double, ob::State*); 

      class ErrorSpaceExecutor {
        public:
          ErrorSpaceExecutor() = delete;

          ErrorSpaceExecutor(EnvironmentPtr env, const VectorXd &goal,
                             double tracking_threshold = 0.5,
                             bool learn = false, bool render = false,
                             int max_overall_timestep = 1e4)
              : tracking_threshold_(tracking_threshold), env_(env), goal_(goal),
                overall_timestep_(0),
                max_overall_timestep_(max_overall_timestep), learn_(learn),
                render_(render) {
            assert(goal_.size() <= env_->get_state_space()->getDimension());
          }

          /*!
           * \brief Execute a path.
           *
           * \param path The path to execute.
           * \param parl Parl learner to optionally learn around the path
           * \param w ExperimentWriter for recording execution statistics
           */
          void execute_path(oc::PathControl &path, ParlPtr parl,
                            utils::ExperimentWriter &w);

          /*!
           * \brief Execute a control in the environment for a given amount of time.
           * 
           * \param parl Parl learner responsible for learning from this execution
           * \param s Previous waypoint
           * \param next_s Next waypoint
           * \param c Control to be applied
           * \param start_time Time in execution of trajectory when control started
           * \param control_dur Amount of time to execute control
           * \param w ExperimentWriter for recording statistics
           */
          void execute_control_for_duration(ParlPtr parl, const VectorXd &s,
                                            const VectorXd &next_s,
                                            const VectorXd &c, double start_time,
                                            double control_dur,
                                            utils::ExperimentWriter &w);

          /*!
           * \brief Execute a single timestep of a control in this environment.
           *
           * \param c Control to execute this timestep
           * \param w ExperimentWriter for recording statistics
           *
           * \returns Stepped environment state 
           */
          VectorXd execute_timestep(const VectorXd &c, utils::ExperimentWriter &w);

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
           * \brief Create an error state representation of the input state
           * with respect to a reference state.
           *
           * \param ref Reference to compute error with respect to.
           * \param actual Observed environment state.
           * \param time Execution time, which will be incorporated into error state.
           *
           * \returns The error state
           */
          VectorXd compute_error_state_(const VectorXd &ref, const VectorXd
              &actual, double time);

          /*!
           * \brief Compute interpolated reference point and corresponding
           * error state for current environment state.
           *
           * \param t Interpolation parameter
           * \param s First waypoint
           * \param next_s Next waypoint
           * \param time Execution time
           *
           * \returns A pair containing the interpolated reference point and
           * error state for the current environment state.
           */
          std::pair<VectorXd, VectorXd> compute_interpolated_error_state_(
              double t, const VectorXd &s, const VectorXd &next_s, double time);

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

          bool learn_; //!< Whether to execute learning
          bool render_; //!< Whether to render during execution
        
      };


    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_PLANNING_EXECUTOR_H */
