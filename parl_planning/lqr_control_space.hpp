#pragma once
#ifndef CANNON_RESEARCH_PARL_LQR_CONTROL_SPACE_H
#define CANNON_RESEARCH_PARL_LQR_CONTROL_SPACE_H 

/*!
 * \file cannon/research/parl_planning/lqr_control_space.hpp
 * File containing LQRControlSpace and LQRControlSampler class definitions.
 */

#include <Eigen/Dense>

#include <ompl/control/ControlSpace.h>

using namespace Eigen;

namespace oc = ompl::control;
namespace ob = ompl::base;

namespace cannon {
  namespace research {
    namespace parl {

      /*!
       * \brief Class representing a control space in which each control is an
       * LQR controller toward a specified state. A linearization of system
       * dynamics is used to compute the LQR controller.
       */
      class LQRControlSpace : public oc::ControlSpace {
        public:

          /*!
           * \brief Constructor taking statespace that these controls will be
           * applied in and the dimension of the controls to be applied.
           */
          LQRControlSpace(const ob::StateSpacePtr &state_space, unsigned int dim);

          /*!
           * \brief Destructor.
           */
          ~LQRControlSpace() override = default;

          /*!
           * \brief Set bounds for each dimension of control.
           */
          void setBounds(const ob::RealVectorBounds &bounds);

          // === Methods from oc::ControlSpace ===
          /*!
           * \brief Inherited from oc::ControlSpace
           */
          unsigned int getDimension() const override;

          /*!
           * \brief Inherited from oc::ControlSpace
           */
          void copyControl(oc::Control *destination, const oc::Control* source) const override;

          /*!
           * \brief Inherited from oc::ControlSpace
           */
          bool equalControls(const oc::Control *control1, const oc::Control *control2) const override;

          /*!
           * \brief Inherited from oc::ControlSpace
           */
          oc::ControlSamplerPtr allocDefaultControlSampler() const override;

          /*!
           * \brief Inherited from oc::ControlSpace
           */
          oc::Control* allocControl() const override;

          /*!
           * \brief Inherited from oc::ControlSpace
           */
          void freeControl(oc::Control* control) const override;
          
          /*!
           * \brief Inherited from oc::ControlSpace
           */
          void nullControl(oc::Control* control) const override;

          /*!
           * \brief Inherited from oc::ControlSpace
           */
          void printControl(const oc::Control* control, std::ostream &out) const override;

          /*!
           * \brief Inherited from oc::ControlSpace
           */
          double* getValueAddressAtIndex(oc::Control* control, unsigned int index) const override;

          /*!
           * \brief Inherited from oc::ControlSpace
           */
          void printSettings(std::ostream &out) const override;

          /*!
           * \brief Inherited from oc::ControlSpace
           */
          void setup() override;

          /*!
           * \brief Inherited from oc::ControlSpace
           */
          unsigned int getSerializationLength() const override;

          /*!
           * \brief Inherited from oc::ControlSpace
           */
          void serialize(void *serialization, const oc::Control* ctrl) const override;

          /*!
           * \brief Inherited from oc::ControlSpace
           */
          void deserialize(oc::Control* ctrl, const void* serialization) const override;

          // === Specific to LQRControlSpace ===

          void compute_u_star(const oc::Control *control,
                              const std::vector<double> &q,
                              Ref<VectorXd> u) const;

        private:
          unsigned int dimension_; //!< Dimension of state space.
          ob::RealVectorBounds bounds_; //!< Control bounds

          std::size_t control_bytes_; //!< Number of bytes making up control

        public:

          /*!
           * \brief Class representing a control in this control space. For
           * LQR, a control is a reference state q0 and a control gain matrix
           * K.
           */
          class ControlType : public oc::Control {
            public:
              
              /*!
               * \brief Constructor taking state dimension and control
               * dimension.
               */
              ControlType(unsigned state_dim, unsigned control_dim);

            public:
              VectorXd q0; //!< Reference state for LQR
              MatrixXd K; //!< Control gain matrix
          }; // ControlType
      }; // LQRControlSpace

      /*!
       * \brief Class that samples LQRControlSpace controls for planning.
       */
      class LQRControlSampler : public oc::ControlSampler {
        public:
          
          /*!
           * \brief Constructor taking the control space to sample for.
           */
          LQRControlSampler(const oc::ControlSpace *space)
              : oc::ControlSampler(space) {}

          /*!
           * \brief Inherited from oc::ControlSampler
           */
          void sample(oc::Control* control) override;

      }; // LQRControlSampler


    } // namespace parl
  } // namespace research
} // namespace cannon


#endif /* ifndef CANNON_RESEARCH_PARL_LQR_CONTROL_SPACE_H */
