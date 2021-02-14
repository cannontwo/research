#include <cannon/research/parl_stability/transition_map.hpp>

using namespace cannon::research::parl;

std::map<std::pair<unsigned int, unsigned int>, Polygon_2> 
    cannon::research::parl::compute_transition_map(std::shared_ptr<Parl> parl, VD diagram) {

  auto voronoi_polygons = create_bounded_voronoi_polygons(parl, diagram);
  log_info("Voronoi diagram has", voronoi_polygons.size(), "bounded polygons");

  std::map<unsigned int, std::tuple<Polygon_2, Polygon_2, Polygon_2>> sat_voronoi_polygons;
  for (auto& pair : voronoi_polygons) {
    log_info("Checking saturation of region", pair.first);

    //Polygon_2 new_linear_poly, min_sat_poly, max_sat_poly;
    //std::tie(new_linear_poly, min_sat_poly, max_sat_poly) =
    //  get_saturated_polygons(pair.second, parl->get_K_matrix_idx_(pair.first),
    //      parl->get_k_vector_idx_(pair.first)[0]);

    sat_voronoi_polygons[pair.first] = get_saturated_polygons(pair.second,
        parl->get_K_matrix_idx_(pair.first),
        parl->get_k_vector_idx_(pair.first)[0]);
  }

  auto dynamics = parl->get_controlled_system();
  auto min_sat_dynamics = parl->get_min_sat_controlled_system();
  auto max_sat_dynamics = parl->get_max_sat_controlled_system();
 
  // Compute transition map
  std::map<std::pair<unsigned int, unsigned int>, Polygon_2> transition_map;
  for (auto const& pair : sat_voronoi_polygons) {
    unsigned int i = pair.first;

    // If the dynamics are not invertible, we can't compute a transition map for this region
    // TODO Should the transition map just be the whole region? This represents
    // a lack of data in this region, so it's not clear what the correct
    // approach is.
    if (dynamics[i].A_.determinant() == 0) 
      continue;

    Polygon_2 linear_poly, min_sat_poly, max_sat_poly;
    std::tie(linear_poly, min_sat_poly, max_sat_poly) = sat_voronoi_polygons[pair.first];
    for (auto& pair2 : voronoi_polygons) {
      unsigned int j = pair2.first;
      std::vector<Polygon_2> premap_polys;

      if (min_sat_poly.size() != 0) {
        // Handle min sat poly 
        Polygon_2 original_set =  compute_premap_set(min_sat_poly,
            pair2.second, min_sat_dynamics[i]);

        if (original_set.size() != 0) {
          premap_polys.push_back(original_set);
        }
      }

      if (max_sat_poly.size() != 0) {
        // Handle max sat poly
        Polygon_2 original_set =  compute_premap_set(max_sat_poly,
            pair2.second, max_sat_dynamics[i]);

        if (original_set.size() != 0) {
          premap_polys.push_back(original_set);
        }
      }

      if (linear_poly.size() != 0) {
        Polygon_2 original_set =  compute_premap_set(linear_poly, pair2.second, dynamics[i]);

        if (original_set.size() != 0) {
          premap_polys.push_back(original_set);
        }
      }

      if (premap_polys.size() > 0) {
        if (premap_polys.size() == 1) {
          transition_map.insert(std::make_pair(std::make_pair(i, j), premap_polys[0]));
        } else {
          Polygon_2 premap_union;
          Polygon_with_holes_2 tmp_union;

          // premap_polys either has size 2 or 3, via if statements above
          CGAL::join(premap_polys[0], premap_polys[1], tmp_union);

          if (premap_polys.size() == 3) {
            Polygon_with_holes_2 tmp_union_2(tmp_union);
            CGAL::join(tmp_union_2, premap_polys[2], tmp_union);
          }

          // Since the map is continuous within each Voronoi region, there should
          // be no holes and we can just take the outer boundary
          assert(tmp_union.number_of_holes() == 0);
          premap_union = tmp_union.outer_boundary();
          transition_map.insert(std::make_pair(std::make_pair(i, j), premap_union));
        }
      }

    } // for (pair2)
  } // for (pair)

  return transition_map;
}

std::pair<std::map<std::pair<unsigned int, unsigned int>, Polygon_2>,
  std::map<unsigned int, std::vector<Polygon_2>>> cannon::research::parl::compute_transition_map(const
      std::vector<std::pair<Polygon_2, AutonomousLinearParams>>& pwa_func) {

  log_info("Computing transition map for PWA system with", pwa_func.size(), "regions");

  std::map<std::pair<unsigned int, unsigned int>, Polygon_2> transition_map;
  std::map<unsigned int, std::vector<Polygon_2>> out_of_bounds_polys;
  for (unsigned int i = 0; i < pwa_func.size(); i++) {
    // If the dynamics are not invertible, we can't compute a transition map for this region
    // TODO Should the transition map just be the whole region? This represents
    // a lack of data in this region, so it's not clear what the correct
    // approach is.
    auto pair = pwa_func[i];
    if (pair.second.A_.determinant() == 0) 
      continue;

    Polygon_set_2 premap_union;
    for (unsigned int j = 0; j < pwa_func.size(); j++) {
      auto pair2 = pwa_func[j];

      // Handle min sat poly 
      Polygon_2 premap_set =  compute_premap_set(pair.first, pair2.first,
          pair.second);

      if (premap_set.size() != 0) {
        if (premap_set.orientation() != CGAL::COUNTERCLOCKWISE) {
          premap_set.reverse_orientation();
        }
        transition_map.insert(std::make_pair(std::make_pair(i, j), premap_set));

        if (premap_union.is_empty())
          premap_union.insert(premap_set);
        else {
          premap_union.join(premap_set);
        }
      }
    }

    // Compute portion of this polygon mapped outside of state space (not in another polygon)
    Polygon_set_2 diff_set;
    diff_set.insert(pair.first);
    diff_set.difference(premap_union);

    std::list<Polygon_with_holes_2> res;
    std::list<Polygon_with_holes_2>::const_iterator it;
    diff_set.polygons_with_holes (std::back_inserter (res));

    // There may be multiple disconnected polgyons mapped out of bounds, but we
    // don't expect any of them to have holes.
    std::vector<Polygon_2> diff_polys;
    for (auto& poly_wh : res) {
      assert(poly_wh.number_of_holes() == 0);
      Polygon_2 diff_poly = poly_wh.outer_boundary();

      if (diff_poly.size() != 0) {
        if (diff_poly.orientation() != CGAL::COUNTERCLOCKWISE) {
          diff_poly.reverse_orientation();
        }

        diff_polys.push_back(diff_poly);
      }
    }
    out_of_bounds_polys.insert(std::make_pair(i, diff_polys));
  }

  return std::make_pair(transition_map, out_of_bounds_polys);
}

std::vector<std::pair<Polygon_2, AutonomousLinearParams>>
cannon::research::parl::compute_parl_pwa_func(std::shared_ptr<Parl> parl, VD
    diagram) {

  std::vector<std::pair<Polygon_2, AutonomousLinearParams>> pwa_func;

  auto voronoi_polygons = create_bounded_voronoi_polygons(parl, diagram);
  log_info("Voronoi diagram has", voronoi_polygons.size(), "bounded polygons");

  std::map<unsigned int, std::tuple<Polygon_2, Polygon_2, Polygon_2>> sat_voronoi_polygons;
  for (auto& pair : voronoi_polygons) {
    log_info("Checking saturation of region", pair.first);

    //Polygon_2 new_linear_poly, min_sat_poly, max_sat_poly;
    //std::tie(new_linear_poly, min_sat_poly, max_sat_poly) =
    //  get_saturated_polygons(pair.second, parl->get_K_matrix_idx_(pair.first),
    //      parl->get_k_vector_idx_(pair.first)[0]);

    sat_voronoi_polygons[pair.first] = get_saturated_polygons(pair.second,
        parl->get_K_matrix_idx_(pair.first),
        parl->get_k_vector_idx_(pair.first)[0]);
  }

  auto dynamics = parl->get_controlled_system();
  auto min_sat_dynamics = parl->get_min_sat_controlled_system();
  auto max_sat_dynamics = parl->get_max_sat_controlled_system();
 
  for (auto const& pair : sat_voronoi_polygons) {
    unsigned int i = pair.first;

    Polygon_2 linear_poly, min_sat_poly, max_sat_poly;
    std::tie(linear_poly, min_sat_poly, max_sat_poly) = sat_voronoi_polygons[pair.first];

    if (min_sat_poly.size() != 0) {
      pwa_func.push_back(std::make_pair(min_sat_poly, min_sat_dynamics[i]));
    }

    if (max_sat_poly.size() != 0) {
      pwa_func.push_back(std::make_pair(max_sat_poly, max_sat_dynamics[i]));
    }

    if (linear_poly.size() != 0) {
      pwa_func.push_back(std::make_pair(linear_poly, dynamics[i]));
    }
  }


  return pwa_func;
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
      // Handle bounded face
      if (!(*fh)->is_unbounded()) {
        Polygon_2 polygon;
        VD::Ccb_halfedge_circulator ec_start = (*fh)->ccb();
        VD::Ccb_halfedge_circulator ec = ec_start;

        do {
          assert(ec->has_target());
          polygon.push_back(ec->target()->point());
        } while (++ec != ec_start);

        ret_map.insert(std::make_pair(i, polygon));
      } else {
        // TODO Clean up this unbounded polygon handling into separate function
        
        // State space bounds
        Nef_polyhedron::Point p0(-M_PI, -8.0), p1(M_PI, -8.0), p2(M_PI, 8.0), p3(-M_PI, 8.0);
        Nef_polyhedron::Point bound_rect[4] = {p0, p1, p2, p3};
        Nef_polyhedron bound_rect_nef_poly(bound_rect, bound_rect+4);
        Nef_polyhedron region_nef_poly(bound_rect_nef_poly);

        Nef_polyhedron::Point ref_point(refs(0, i), refs(1, i));
        auto nef_locate_result = region_nef_poly.locate(ref_point);
        assert(region_nef_poly.contains(nef_locate_result));

        VD::Ccb_halfedge_circulator ec_start = (*fh)->ccb();
        VD::Ccb_halfedge_circulator ec = ec_start;

        log_info("Constructing polygon for unbounded face");

        do {
          // Iterate around unbounded face, intersecting each half-plane
          // corresponding to an edge with polyhedron
          if (ec->is_segment()) {
            // Handle finite segment
            assert(ec->has_source() && ec->has_target());
            Nef_polyhedron::Point src(ec->source()->point().x(),
                ec->source()->point().y());
            Nef_polyhedron::Point tgt(ec->target()->point().x(),
                ec->target()->point().y());

            Nef_polyhedron::Line edge_line(src, tgt);
            assert(edge_line.has_on_positive_side(ref_point));

            Nef_polyhedron halfplane(edge_line, Nef_polyhedron::INCLUDED);

            Nef_polyhedron tmp_intersection = region_nef_poly.intersection(halfplane);
            assert(!tmp_intersection.is_empty());

            region_nef_poly = tmp_intersection;

            auto nef_locate_result = region_nef_poly.locate(ref_point);
            assert(region_nef_poly.contains(nef_locate_result));
          } else {
            // Handle ray
            VD::Delaunay_edge dual_edge = ec->dual();

            // This is pretty jank, but might be the most parsimonious way to
            // construct the ray
            CGAL::Object ray_obj = diagram.dual().dual(dual_edge);  
            const K::Ray_2 *ray = CGAL::object_cast<K::Ray_2>(&ray_obj);
            assert(ray);

            K::Direction_2 ray_dir = ray->direction();
            K::Point_2 ray_src = ray->source();

            Nef_polyhedron::Point src(ray_src.x(), ray_src.y());
            Nef_polyhedron::Point tgt(ray_src.x() + ray_dir.dx(),
                ray_src.y() + ray_dir.dy());

            Nef_polyhedron::Line edge_line;
            if (ec->has_source()) {
              edge_line = Nef_polyhedron::Line(src, tgt);
            } else if (ec->has_target()) {
              edge_line = Nef_polyhedron::Line(tgt, src);
            } else {
              throw std::runtime_error("We should not be here");
            }

            assert(edge_line.has_on_positive_side(ref_point));

            Nef_polyhedron halfplane(edge_line, Nef_polyhedron::INCLUDED);

            Nef_polyhedron tmp_intersection = region_nef_poly.intersection(halfplane);
            assert(!tmp_intersection.is_empty());

            region_nef_poly = tmp_intersection;

            auto nef_locate_result = region_nef_poly.locate(ref_point);
            assert(region_nef_poly.contains(nef_locate_result));
          }

        } while (++ec != ec_start);

        Polygon_2 polygon = extract_finite_face_polygon(region_nef_poly.explorer());
        assert(polygon.is_simple() && polygon.size() > 0);

        if (polygon.orientation() != CGAL::COUNTERCLOCKWISE) {
          polygon.reverse_orientation();
        }

        ret_map.insert(std::make_pair(i, polygon));

      }
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

Polygon_2 cannon::research::parl::compute_premap_set(const Polygon_2& map_poly,
    const Polygon_2& test_poly, const AutonomousLinearParams& map) {

  std::pair<Polygon_2, Transformation> transformed_poly_pair =
    affine_map_polygon(map_poly, map);
  Polygon_2 trans_poly = transformed_poly_pair.first;

  assert(trans_poly.is_simple());

  if (trans_poly.orientation() != CGAL::COUNTERCLOCKWISE) {
    trans_poly.reverse_orientation();
  }

  std::vector<Polygon_with_holes_2> intersection_polys;
  CGAL::intersection(trans_poly, test_poly,
      std::back_inserter(intersection_polys));

  // For intersections of voronoi polygons, all regions should be convex
  assert(intersection_polys.size() == 0 || intersection_polys.size() == 1);

  Polygon_2 original_set;
  if (intersection_polys.size() == 1) {
    // There shouldn't be any holes for Voronoi regions
    assert(intersection_polys[0].number_of_holes() == 0);
    Polygon_2 intersection_poly = intersection_polys[0].outer_boundary();

    original_set = CGAL::transform(transformed_poly_pair.second,
        intersection_poly);
  }

  return original_set;
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
      do {
        Vertex_const_handle vh = e.target(hafc);
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

bool cannon::research::parl::is_inside(const Vector2d& state, const Polygon_2& poly) {
  K::Point_2 query(state[0], state[1]);
  switch (CGAL::bounded_side_2(poly.begin(), poly.end(), query, K())) {
    case CGAL::ON_BOUNDED_SIDE:
      return true;
    case CGAL::ON_BOUNDARY:
      return true;
    case CGAL::ON_UNBOUNDED_SIDE:
      return false;
  }  
}
