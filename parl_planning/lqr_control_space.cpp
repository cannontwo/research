#include <cannon/research/parl_planning/lqr_control_space.hpp>

#include <ompl/util/Exception.h>

using namespace cannon::research::parl;

// LQRControlSpace

LQRControlSpace::LQRControlSpace(const ob::StateSpacePtr &state_space, unsigned int dim) : ControlSpace(state_space), dimension_(dim), bounds_(dim), control_bytes_((state_space->getDimension() + dim * dim) * sizeof(double)) {
  setName("LQR" + getName());
  type_ = oc::CONTROL_SPACE_TYPE_COUNT + 1;
}

void LQRControlSpace::setBounds(const ob::RealVectorBounds &bounds) {
  bounds.check();
  if (bounds.low.size() != dimension_)
    throw ompl::Exception(
        "Bounds do not match dimension of control space: expected dimension " +
        std::to_string(dimension_) + " but got dimension " +
        std::to_string(bounds.low.size()));

  bounds_ = bounds;
}

unsigned int LQRControlSpace::getDimension() const {
  return dimension_;
}

void LQRControlSpace::copyControl(oc::Control *destination, const oc::Control* source) const {
  auto dest = static_cast<LQRControlSpace::ControlType*>(destination);
  auto src = static_cast<const LQRControlSpace::ControlType*>(source);
  dest->q0 = src->q0;
  dest->K = src->K;
}

bool LQRControlSpace::equalControls(const oc::Control *control1, const oc::Control *control2) const {
  auto ctrl1 = static_cast<const LQRControlSpace::ControlType*>(control1);
  auto ctrl2 = static_cast<const LQRControlSpace::ControlType*>(control2);
  return ctrl1->q0.isApprox(ctrl2->q0) && ctrl1->K.isApprox(ctrl2->K);
}

oc::ControlSamplerPtr LQRControlSpace::allocDefaultControlSampler() const {
  return std::make_shared<LQRControlSampler>(this);
}

oc::Control* LQRControlSpace::allocControl() const {
  return new LQRControlSpace::ControlType(getStateSpace()->getDimension(), getDimension());
}

void LQRControlSpace::freeControl(oc::Control* control) const {
  delete static_cast<ControlType*>(control);
}

void LQRControlSpace::nullControl(oc::Control* control) const {
  const unsigned int sdim = getStateSpace()->getDimension();
  const unsigned int cdim = getDimension();
  auto *lqr_control = static_cast<LQRControlSpace::ControlType*>(control);
  lqr_control->q0.setZero(sdim);
  lqr_control->K.setZero(cdim, sdim);
}

void LQRControlSpace::printControl(const oc::Control* control, std::ostream &out) const {
  auto lqr_control = static_cast<const LQRControlSpace::ControlType*>(control);
  out << "q0^T: " << lqr_control->q0.transpose() << "\nK: " << lqr_control->K << std::endl;
}

double* LQRControlSpace::getValueAddressAtIndex(oc::Control* control, unsigned int index) const {
  std::size_t sdim = getStateSpace()->getDimension(), max_index = dimension_ * dimension_ + sdim;

  if (index < sdim)
    return control->as<ControlType>()->q0.data() + index;
  else if (index < max_index)
    return control->as<ControlType>()->K.data() + index - sdim;
  else
    return nullptr;
}

void LQRControlSpace::printSettings(std::ostream &out) const {
  out << "LQR control space '" << getName() << "' with bounds: " << std::endl;
  out << "  - min: ";
  for (unsigned int i = 0; i < dimension_; i++)
    out << bounds_.low[i] << " ";
  out << std::endl;
  out << "  - max: ";
  for (unsigned int i = 0; i < dimension_; i++)
    out << bounds_.high[i] << " ";
  out << std::endl;
}

void LQRControlSpace::setup() {
  ControlSpace::setup();
  bounds_.check();
}

unsigned int LQRControlSpace::getSerializationLength() const {
  return control_bytes_;
}

void LQRControlSpace::serialize(void *serialization, const oc::Control* ctrl) const {
  std::size_t vec_length = getStateSpace()->getDimension() * sizeof(double);
  memcpy(serialization, ctrl->as<ControlType>()->q0.data(), vec_length);
  memcpy((char *)serialization + vec_length, ctrl->as<ControlType>()->K.data(),
         control_bytes_ - vec_length);
}

void LQRControlSpace::deserialize(oc::Control* ctrl, const void* serialization) const {
  std::size_t vec_length = getStateSpace()->getDimension() * sizeof(double);
  memcpy(ctrl->as<ControlType>()->q0.data(), serialization, vec_length);
  memcpy(ctrl->as<ControlType>()->K.data(), (char*)serialization + vec_length,
         control_bytes_ - vec_length);
}

void LQRControlSpace::compute_u_star(const oc::Control *control,
                                     const std::vector<double> &q,
                                     Ref<VectorXd> u) const {
  auto lqr_control = static_cast<const ControlType*>(control);
  Map<const VectorXd> qvec(q.data(), q.size());
  u = (-lqr_control->K * (qvec - lqr_control->q0))
          .array()
          .max(Map<const ArrayXd>(bounds_.low.data(), dimension_))
          .min(Map<const ArrayXd>(bounds_.high.data(), dimension_));
}

// LQRControlSpace::ControlType

LQRControlSpace::ControlType::ControlType(unsigned int state_dim,
                                          unsigned int control_dim)
    : q0(VectorXd::Zero(state_dim)), K(MatrixXd::Zero(control_dim, state_dim)) {}

// LQRControlSampler

void LQRControlSampler::sample(oc::Control* control) {
  space_->nullControl(control);
  OMPL_WARN("LQRControlSampler: control set to 0");
}
