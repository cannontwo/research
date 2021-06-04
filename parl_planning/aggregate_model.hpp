#ifndef CANNON_RESEARCH_PARL_PLANNING_AGGREGATE_MODEL_H
#define CANNON_RESEARCH_PARL_PLANNING_AGGREGATE_MODEL_H 

#include <map>
#include <cassert>

#include <Eigen/Dense>
#include <ompl/control/PathControl.h>

#include <cannon/physics/systems/system.hpp>
#include <cannon/research/parl/linear_params.hpp>
#include <cannon/utils/class_forward.hpp>

using namespace Eigen;

using namespace cannon::physics::systems;

namespace cannon {

  namespace ml {
    CANNON_CLASS_FORWARD(RLSFilter);
  }

  namespace research {
    namespace parl {

      CANNON_CLASS_FORWARD(Parl);
      CANNON_CLASS_FORWARD(Environment);

      using VectorXu = Matrix<unsigned int, Dynamic, 1>;

      bool vector_comp(const VectorXu& v1, const VectorXu& v2);

      class AggregateModel : public System {
        public:
          AggregateModel() = delete;

          AggregateModel(std::shared_ptr<System> nominal_model, unsigned int
              state_dim, unsigned int action_dim, unsigned int grid_size,
              MatrixX2d& bounds, double time_delta, bool learn=true) :
            nominal_model_(nominal_model), state_dim_(state_dim), action_dim_(action_dim),
            grid_size_(grid_size), bounds_(bounds), time_delta_(time_delta), parameters_(vector_comp), learn_(learn) {

            assert(bounds_.rows() == state_dim_);
            assert(time_delta >= 0.0);

            // bounds[:, 0] is lower bounds, bounds[:, 1] is upper bounds
            cell_extent_ = VectorXd::Zero(state_dim_);
            for (unsigned int i = 0; i < state_dim_; i++) {
              assert(bounds_(i, 1) > bounds_(i, 0));
              double length = bounds_(i, 1) - bounds_(i, 0);
              
              cell_extent_[i] = length / (double)grid_size_;
            }
          }

          virtual ~AggregateModel() {}

          virtual void operator()(const VectorXd& s, VectorXd& dsdt, const double t) override;

          virtual void ompl_ode_adaptor(const oc::ODESolver::StateType& q,
              const oc::Control* control, oc::ODESolver::StateType& qdot) override;

          virtual std::tuple<MatrixXd, MatrixXd, VectorXd> get_linearization(const VectorXd& x) override;

          void add_local_model(const ml::RLSFilter& model, const VectorXd&
              ref_state, const VectorXd& next_ref_state, 
              const VectorXd& ref_control, double tau, double tau_delta);

          void process_path_parl(EnvironmentPtr env,
              ParlPtr model, oc::PathControl& path);

          LinearParams get_local_model_for_state(const VectorXd& state);

          VectorXu get_grid_coords(const VectorXd& query) const;

          /*!
           * \brief Save the learned portion of this aggregate model to an HDF5 file.
           *
           * The structure of the stored data is:
           *
           * /coords/i - Coordinates in grid for learned dynamics with index i.
           * /A_mats/i - Autonomous portion of learned dynamics.
           * /B_mats/i - Controlled portion of learned dynamics.
           * /c_vecs/i - Offset portion of learned dynamics.
           * /num_data/i - Number of data points contributing to learned dynamics.
           *
           * \param path The path to save to.
           */
          void save(const std::string& path);

          /*!
           * \brief Load learned portion of aggregate model from an HDF5 file,
           * overriding whatever is currently in this object.
           *
           * The structure of the stored data should be:
           *
           * /coords/i - Coordinates in grid for learned dynamics with index i.
           * /A_mats/i - Autonomous portion of learned dynamics.
           * /B_mats/i - Controlled portion of learned dynamics.
           * /c_vecs/i - Offset portion of learned dynamics.
           * /num_data/i - Number of data points contributing to learned dynamics.
           *
           * \param path The path to load. 
           */
          void load(const std::string& path);

          /*!
           * \brief Compute overall Frobenius norm between linearizations of
           * this model and linearizations of the input environment. 
           * 
           * \param env The "true" system to compute error with respect to.
           *
           * \returns Total Frobenius norm between linearizations in each cell.
           */
          double compute_model_error(EnvironmentPtr env);

        private:
          std::shared_ptr<System> nominal_model_;

          unsigned int state_dim_;
          unsigned int action_dim_;
          unsigned int grid_size_;

          MatrixX2d bounds_;
          VectorXd cell_extent_;

          double time_delta_;
          
          std::map<VectorXu, LinearParams, std::function<bool(const VectorXu&,
              const VectorXu&)>, aligned_allocator<std::pair<VectorXu,
            LinearParams>>> parameters_;

          bool learn_;

      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_PLANNING_AGGREGATE_MODEL_H */
