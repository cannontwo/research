#include <cannon/research/parl_stability/transition_map.hpp>

using namespace cannon::research::parl;

std::map<std::pair<unsigned int, unsigned int>, Polygon_2> 
    cannon::research::parl::compute_transition_map(std::shared_ptr<Parl> parl, VD diagram) {

  auto voronoi_polygons = create_bounded_voronoi_polygons(parl, diagram);
  log_info("Voronoi diagram has", voronoi_polygons.size(), "bounded polygons");

  auto dynamics = parl->get_controlled_system();
 
  // Compute transition map
  std::map<std::pair<unsigned int, unsigned int>, Polygon_2> transition_map;
  for (auto const& pair : voronoi_polygons) {
    unsigned int i = pair.first;

    // If the dynamics are not invertible, we can't compute a transition map for this region
    // TODO Should the transition map just be the whole region? This represents
    // a lack of data in this region, so it's not clear what the correct
    // approach is.
    if (dynamics[i].A_.determinant() == 0) 
      continue;

    std::pair<Polygon_2, Transformation> transformed_poly_pair =
      affine_map_polygon(pair.second, dynamics[i]);
    Polygon_2 trans_poly = transformed_poly_pair.first;

    assert(trans_poly.is_simple());

    if (trans_poly.orientation() != CGAL::COUNTERCLOCKWISE) {
      trans_poly.reverse_orientation();
    }

    for (auto& pair2 : voronoi_polygons) {
      unsigned int j = pair2.first;

      std::vector<Polygon_with_holes_2> intersection_polys;
      CGAL::intersection(trans_poly, pair2.second,
          std::back_inserter(intersection_polys));

      // For intersections of voronoi polygons, all regions should be convex
      assert(intersection_polys.size() == 0 || intersection_polys.size() == 1);

      if (intersection_polys.size() == 1) {
        // There shouldn't be any holes for Voronoi regions
        assert(intersection_polys[0].number_of_holes() == 0);
        Polygon_2 intersection_poly = intersection_polys[0].outer_boundary();

        Polygon_2 original_set =
          CGAL::transform(transformed_poly_pair.second, intersection_poly);
        transition_map.insert(std::make_pair(std::make_pair(i, j), original_set));
      }
    }
  }

  log_info("Computed transition map with", transition_map.size(), "elements");

  return transition_map;
}

std::map<unsigned int, Polygon_2> cannon::research::parl::create_bounded_voronoi_polygons(
    std::shared_ptr<Parl> parl, VD diagram) {

  MatrixXd refs = parl->get_refs();
  std::map<unsigned int, Polygon_2> ret_map;

  for (unsigned int i = 0; i < refs.cols(); i++) {
    VD::Point_2 p(refs(0, i), refs(1, i));
    auto locate_result = diagram.locate(p);

    // Locate result should be a face, since this is a ref
    if (VD::Face_handle* fh = boost::get<VD::Face_handle>(&locate_result)) {

      // For now, only adding bounded faces
      if (!(*fh)->is_unbounded()) {
        Polygon_2 polygon;
        VD::Ccb_halfedge_circulator ec_start = (*fh)->ccb();
        VD::Ccb_halfedge_circulator ec = ec_start;

        do {
          assert(ec->has_target());
          polygon.push_back(ec->target()->point());
        } while (++ec != ec_start);

        ret_map.insert(std::make_pair(i, polygon));
      }
      // TODO Add else case, simply intersect unbounded polygon with bounding rectangle
    } else {
      throw std::runtime_error("Ref point query did not result in Voronoi face.");
    }
  }

  return ret_map;
}

std::pair<Polygon_2, Transformation> cannon::research::parl::affine_map_polygon(const Polygon_2& p, 
    const AutonomousLinearParams& map) {
  assert(map.state_dim_ == 2);

  Transformation aff_trans(map.A_(0, 0), map.A_(0, 1), map.c_[0],
                           map.A_(1, 0), map.A_(1, 1), map.c_[1]);

  auto ret_poly = CGAL::transform(aff_trans, p);

  return std::make_pair(ret_poly, aff_trans.inverse());
}
