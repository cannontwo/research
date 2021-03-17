#include <cannon/research/parl_stability/transition_map.hpp>

using namespace cannon::research::parl;

std::pair<TransitionMap, OutMap> cannon::research::parl::compute_transition_map(const
      std::vector<std::pair<Polygon_2, AutonomousLinearParams>>& pwa_func) {

  log_info("Computing transition map for PWA system with", pwa_func.size(), "regions");

  TransitionMap transition_map;
  OutMap out_of_bounds_polys;
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

      std::vector<Polygon_2> premap_set = compute_premap_set(pair.first, pair2.first,
          pair.second);

      std::vector<Polygon_2> poly_vec;

      for (auto& poly : premap_set) {
        if (poly.size() != 0) {
          if (poly.orientation() != CGAL::COUNTERCLOCKWISE) {
            poly.reverse_orientation();
          }

          poly_vec.push_back(poly);

          if (premap_union.is_empty())
            premap_union.insert(poly);
          else {
            premap_union.join(poly);
          }
        }
      }

      if (poly_vec.size() != 0) {
        transition_map.insert(std::make_pair(std::make_pair(i, j), poly_vec));
      }
    }

    // Compute portion of this polygon mapped outside of state space (not in another polygon)
    Polygon_set_2 diff_set;
    diff_set.insert(pair.first);
    diff_set.difference(premap_union);

    std::list<Polygon_with_holes_2> res;
    std::list<Polygon_with_holes_2>::const_iterator it;
    diff_set.polygons_with_holes (std::back_inserter (res));

    // There may be multiple disconnected polygons mapped out of bounds, but we
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
    if (diff_polys.size() != 0) {
      out_of_bounds_polys.insert(std::make_pair(i, diff_polys));
    }
  }

  return std::make_pair(transition_map, out_of_bounds_polys);
}

std::pair<TransitionMap, OutMap>
cannon::research::parl::compute_transition_map(const PWAFunc& pwa_func, const
    TransitionMap& old_transition_map, const std::multimap<unsigned int,
    unsigned int>& correspondence) {

  log_info("Computing transition map for PWA system with", pwa_func.size(), "regions");

  TransitionMap transition_map;
  OutMap out_of_bounds_polys;

  std::vector<Polygon_set_2> premap_unions(pwa_func.size());

  for (auto& idx_pair : old_transition_map) {
    unsigned int i, j;
    std::tie(i, j) = idx_pair.first;

    auto i_newpoly_range = correspondence.equal_range(i);
    auto j_newpoly_range = correspondence.equal_range(j);

    for (auto new_idx = i_newpoly_range.first; new_idx !=
        i_newpoly_range.second; new_idx++) {

      // TODO
      auto pair = pwa_func[new_idx->second];
      if (pair.second.A_.determinant() == 0) 
        continue;

      for (auto new_target_idx = j_newpoly_range.first; new_target_idx !=
          j_newpoly_range.second; new_target_idx++) {

        auto pair2 = pwa_func[new_target_idx->second];

        std::vector<Polygon_2> premap_set = compute_premap_set(pair.first, pair2.first,
            pair.second);

        std::vector<Polygon_2> poly_vec;

        for (auto& poly : premap_set) {
          if (poly.size() != 0) {
            if (poly.orientation() != CGAL::COUNTERCLOCKWISE) {
              poly.reverse_orientation();
            }

            poly_vec.push_back(poly);

            if (premap_unions[new_idx->second].is_empty())
              premap_unions[new_idx->second].insert(poly);
            else {
              premap_unions[new_idx->second].join(poly);
            }
          }
        }

        if (poly_vec.size() != 0) {
          transition_map.insert(std::make_pair(std::make_pair(new_idx->second,
                  new_target_idx->second), poly_vec));
        }
      }   
    }
  }

  for (unsigned int i = 0; i < pwa_func.size(); i++) {
    // Compute portion of this polygon mapped outside of state space (not in another polygon)
    Polygon_set_2 diff_set;
    diff_set.insert(pwa_func[i].first);
    diff_set.difference(premap_unions[i]);

    std::list<Polygon_with_holes_2> res;
    std::list<Polygon_with_holes_2>::const_iterator it;
    diff_set.polygons_with_holes (std::back_inserter (res));

    // There may be multiple disconnected polygons mapped out of bounds, but we
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
    if (diff_polys.size() != 0) {
      out_of_bounds_polys.insert(std::make_pair(i, diff_polys));
    }
  }


  return std::make_pair(transition_map, out_of_bounds_polys);

}

std::vector<std::pair<Polygon_2, AutonomousLinearParams>>
cannon::research::parl::compute_parl_pwa_func(std::shared_ptr<Parl> parl, VD
    diagram) {

  std::vector<std::pair<Polygon_2, AutonomousLinearParams>> pwa_func;

  auto voronoi_polygons = create_bounded_voronoi_polygons(parl->get_refs(), diagram);
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

  // TODO This is a hack 
  // Correct dynamics in regions containing zero to avoid numerical errors
  for (auto& pair : pwa_func) {
    if (is_inside(Vector2d::Zero(), pair.first)) {
      pair.second.c_ = Vector2d::Zero();
    }
  }

  return pwa_func;
}

std::pair<Polygon_2, Transformation> cannon::research::parl::affine_map_polygon(const Polygon_2& p, 
    const AutonomousLinearParams& map) {
  assert(map.state_dim_ == 2);

  Transformation aff_trans(map.A_(0, 0), map.A_(0, 1), map.c_[0],
                           map.A_(1, 0), map.A_(1, 1), map.c_[1]);

  auto ret_poly = CGAL::transform(aff_trans, p);

  return std::make_pair(ret_poly, aff_trans.inverse());
}

std::vector<Polygon_2> cannon::research::parl::compute_premap_set(const Polygon_2& map_poly,
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
  
  std::vector<Polygon_2> ret_vec;
  for (auto& poly : intersection_polys) {
    assert(poly.number_of_holes() == 0);

    Polygon_2 intersection_poly = poly.outer_boundary();
    ret_vec.push_back(CGAL::transform(transformed_poly_pair.second,
          intersection_poly));
  }

  return ret_vec;
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

void cannon::research::parl::plot_transition_map(const std::pair<TransitionMap, OutMap>&
    transition_map_pair, const PWAFunc& pwa_func, bool save) {
  Plotter p;

  for (auto const& pair : transition_map_pair.first) {
    float hue_1 = pair.first.first  * (360.0 / (float)pwa_func.size()); 
    float hue_2 = pair.first.second * (360.0 / (float)pwa_func.size());
    auto rgb = hsv_to_rgb((hue_1 + hue_2) / 2.0, 0.75, 0.9);
    Vector4f color = Vector4f::Ones();
    color.head(3) = rgb;
    for (auto const& poly : pair.second) {
      //p.plot_polygon(poly, color);
      p.plot_polygon(poly, generate_random_color());
    }
  }


  for (auto const& pair : transition_map_pair.second) {
    float hue = pair.first  * (360.0 / (float)pwa_func.size()); 
    Vector4f color = Vector4f::Ones();
    auto rgb = hsv_to_rgb(hue, 0.75, 0.25);
    color.head(3) = rgb;
    for (auto const& poly : pair.second) {
      //p.plot_polygon(poly, color);
      p.plot_polygon(poly, generate_random_color(0.75, 0.25));
    }
  }
  
  if (save) {
    std::filesystem::create_directory("img");
    std::string path("img/transition_map_");
    path += std::to_string(glfwGetTime()) + std::string(".png");
    p.save_image(path);
  } else {
    p.render();
  }

  p.close();
}

void cannon::research::parl::save_pwa(const PWAFunc& pwa, const std::string& path) {
  H5Easy::File file(path, H5Easy::File::Overwrite);
  std::string polygon_path("/polygons/");
  std::string A_path("/A_mats/");
  std::string c_path("/c_vecs/");

  log_info("Saving PWA model with", pwa.size(), "regions");

  for (unsigned int i = 0; i < pwa.size(); i++) {
    std::vector<Vector2d> poly_vec;
    for (auto it = pwa[i].first.vertices_begin(); it < pwa[i].first.vertices_end(); it++) {
      Vector2d vec = Vector2d::Zero();
      vec[0] = CGAL::to_double(it->x());
      vec[1] = CGAL::to_double(it->y());

      poly_vec.push_back(vec);
    }


    H5Easy::dump(file, polygon_path + std::to_string(i), poly_vec);
    H5Easy::dump(file, A_path + std::to_string(i), pwa[i].second.A_);
    H5Easy::dump(file, c_path + std::to_string(i), pwa[i].second.c_);
  }

  file.flush();
}

PWAFunc cannon::research::parl::load_pwa(const std::string& path) {
  PWAFunc ret_pwa;

  H5Easy::File file(path, H5Easy::File::ReadOnly);

  auto polygon_group = file.getGroup("/polygons");
  auto A_group = file.getGroup("/A_mats");
  auto c_group = file.getGroup("/c_vecs");
  log_info("/polygons group has", polygon_group.getNumberObjects(), "objects");

  assert(polygon_group.getNumberObjects() == A_group.getNumberObjects());
  assert(polygon_group.getNumberObjects() == c_group.getNumberObjects());

  std::string polygon_path("/polygons/");
  std::string A_path("/A_mats/");
  std::string c_path("/c_vecs/");

  for (unsigned int i = 0; i < polygon_group.getNumberObjects(); i++) {

    // Read polygon for region i
    std::vector<Vector2d> poly_vec = H5Easy::load<std::vector<Vector2d>>(file, polygon_path + std::to_string(i));
    Polygon_2 poly;
    for (unsigned int i = 0; i < poly_vec.size(); i++) {
      Vector2d eig_p = poly_vec[i];
      K::Point_2 p(eig_p[0], eig_p[1]);
      poly.push_back(p);
    }

    // Read autonomous dynamics for region i
    MatrixXd A = H5Easy::load<MatrixXd>(file, A_path + std::to_string(i));
    VectorXd c = H5Easy::load<VectorXd>(file, c_path + std::to_string(i));
    AutonomousLinearParams dynamics(A, c, 0);

    ret_pwa.push_back(std::make_pair(poly, dynamics));
  }

  return ret_pwa;
}

PWAFunc cannon::research::parl::restrict_pwa(const PWAFunc& pwa, double radius) {
  PWAFunc ret_pwa;

  for (auto& pair : pwa) {
    bool all_in = true;
    for (auto it = pair.first.vertices_begin(); it < pair.first.vertices_end(); it++) {
      Vector2d vert_vec;
      vert_vec << CGAL::to_double(it->x()),
                  CGAL::to_double(it->y());

      //if (vert_vec.norm() > radius) {
      if (std::fabs(vert_vec[0]) > radius || std::fabs(vert_vec[1]) > 8) {
        all_in = false;
        break;
      }
    }

    if (all_in) {
      ret_pwa.push_back(pair);
    }
  }

  return ret_pwa;
}

Vector2d cannon::research::parl::evaluate_pwa(const PWAFunc& pwa, const Vector2d& query) {
  for (unsigned int i = 0; i < pwa.size(); i++) {
    auto pair = pwa[i];
    if (is_inside(query, pair.first)) {
      return pair.second.A_ * query + pair.second.c_;
    }
  }

  throw std::runtime_error("Query state was not in domain of PWA function");
}
