#include <cannon/research/parl_stability/voronoi.hpp>

using namespace cannon::research::parl;

VD cannon::research::parl::compute_voronoi_diagram(std::shared_ptr<Parl> parl) {
  MatrixXd refs = parl->get_refs();
  unsigned int dim = refs.rows();

  if (dim != 2) {
    throw std::runtime_error("Non-planar voronoi diagram generation not implemented yet");
  }

  // Adapted from
  // https://doc.cgal.org/latest/Voronoi_diagram_2/index.html#Chapter_2D_Voronoi_Diagram_Adaptor
  VD diagram;
  for (unsigned int i = 0; i < refs.cols(); i++) {
    AT::Site_2 site(refs.col(i)[0], refs.col(i)[1]);
    diagram.insert(site);

    log_info("Processing point", i, "/", refs.cols(), "into Voronoi diagram");
  }

  // TODO Also add bounds on state space to Voronoi diagram

  assert(diagram.is_valid());

  return diagram;
}
