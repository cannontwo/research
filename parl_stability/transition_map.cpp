#include <cannon/research/parl_stability/transition_map.hpp>

using namespace cannon::research::parl;

std::map<std::pair<unsigned int, unsigned int>, Polygon_2> 
    cannon::research::parl::compute_transition_map(std::shared_ptr<Parl> parl, VD diagram) {

  auto voronoi_polygons = create_bounded_voronoi_polygons(parl, diagram);
  log_info("Voronoi diagram has", voronoi_polygons.size(), "bounded polygons");

  unsigned int sat_regions = 0;
  for (auto& pair : voronoi_polygons) {
    log_info("Checking saturation of region", pair.first);

    Polygon_2 new_linear_poly, min_sat_poly, max_sat_poly;

    std::tie(new_linear_poly, min_sat_poly, max_sat_poly) =
      get_saturated_polygons(pair.second, parl->get_K_matrix_idx_(pair.first),
          parl->get_k_vector_idx_(pair.first)[0]);


    if (min_sat_poly.size() != 0 || max_sat_poly.size() != 0) {
      // TODO Adjust Voronoi polygon collection
      
      sat_regions += 1;
    }
  }

  log_info(sat_regions, "Voronoi cells exhibited controller saturation");

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
      // TODO Add else case, intersect unbounded polygon with bounding
      // rectangle. Turns out that this is nontrivial, and for the inverted
      // pendulum we would only be bounding the velocity dimension. For now I'm
      // not going to worry about it
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

std::tuple<Polygon_2, Polygon_2, Polygon_2>
cannon::research::parl::get_saturated_polygons(const Polygon_2&
    voronoi_polygon, const RowVector2d& K, double k, double lower, double upper) {

  if (K[0] == 0 && K[1] == 0) {
    // Line is degenerate, and likely controller hasn't been updated at all
    Polygon_2 ret_poly;
    return std::make_tuple(ret_poly, ret_poly, ret_poly);
  }

  assert(voronoi_polygon.is_simple());
  std::vector<Extended_kernel::Standard_point_2> vor_pts;
  for (auto& pt : voronoi_polygon) {
    vor_pts.push_back(Extended_kernel::Standard_point_2(pt.x(), pt.y()));
  }

  Nef_polyhedron voronoi_nef_poly(vor_pts.begin(), vor_pts.end());
  //log_info("Voronoi nef poly is", voronoi_nef_poly);

  // Upper
  Nef_polyhedron::Line upper_line(K[0], K[1], k - upper); // K * x + k>= upper
  Nef_polyhedron upper_poly(upper_line, Nef_polyhedron::INCLUDED);
  Nef_polyhedron upper_intersection = upper_poly.intersection(voronoi_nef_poly);
  Polygon_2 upper_intersection_poly;
  if (!upper_intersection.is_empty()) {
    log_info("Found nonempty upper saturated region");
    Nef_polyhedron::Explorer e = upper_intersection.explorer();
    upper_intersection_poly = extract_finite_face_polygon(e);
  }
  
  // Lower
  Nef_polyhedron::Line lower_line(-K[0], -K[1], lower - k); // K * x + k <= lower, reoriented
  Nef_polyhedron lower_poly(lower_line, Nef_polyhedron::INCLUDED);
  Nef_polyhedron lower_intersection = lower_poly.intersection(voronoi_nef_poly);
  Polygon_2 lower_intersection_poly;
  if (!lower_intersection.is_empty()) {
    log_info("Found nonempty lower saturated region");
    Nef_polyhedron::Explorer e = lower_intersection.explorer();
    lower_intersection_poly = extract_finite_face_polygon(e);
  }

  Polygon_2 diff_poly;
  if (!upper_intersection.is_empty() || !lower_intersection.is_empty()) {
    Nef_polyhedron tmp_diff = voronoi_nef_poly.difference(upper_intersection);
    Nef_polyhedron diff = tmp_diff.difference(lower_intersection);

    if (!diff.is_empty()) {
      Nef_polyhedron::Explorer e = diff.explorer();
      diff_poly = extract_finite_face_polygon(e);
    }
  }
  
  return std::make_tuple(diff_poly, lower_intersection_poly, upper_intersection_poly);
}


Polygon_2 cannon::research::parl::extract_finite_face_polygon(const Nef_polyhedron::Explorer& e) {
  Polygon_2 ret_poly;
  
  for (auto fit = e.faces_begin(); fit != e.faces_end(); fit++) {
    auto hafc = e.face_cycle(fit);

    if (hafc == Halfedge_around_face_const_circulator()) {
      //log_info("Face has no outer face cycle, skipping.");
      continue;
    }

    Halfedge_around_face_const_circulator done(hafc); // Circulator start
    bool found_infinite = false;
    do {
      if (e.is_frame_edge(hafc)) {
        //log_info("Found frame edge, skipping");
        found_infinite = true;
        break;
      }

      hafc++;
    } while (hafc != done);

    if (!found_infinite) {
      // This could be the finite face we're looking for
      Hole_const_iterator hit = e.holes_begin(fit), end = e.holes_end(fit); 
      assert(hit == end); // We should not have any holes for Voronoi regions under affine maps

      // Loop back over halfedges and construct polygon
      log_info("Constructing polygon:");
      do {
        Vertex_const_handle vh = e.target(hafc);
        if (e.is_standard(vh))
          log_info("\t Point:", e.point(vh));
        else
          log_info("\t Ray: ", e.ray(vh));
        // If this face contains extended points (rays) then it's not the face
        // we're looking for
        if (!e.is_standard(vh)) {
          //log_info("Face has a non-standard vertex, skipping");
          found_infinite = true;
          ret_poly = Polygon_2();
          break;
        }

        ret_poly.push_back(K::Point_2(e.point(vh).x(), e.point(vh).y()));

        hafc++;
      } while (hafc != done);
      
      if (!found_infinite) {
        return ret_poly; 
      }
    }
  }

  throw std::runtime_error("Did not find finite face in nef_polyhedron explorer");
}
