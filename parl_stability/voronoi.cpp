#include <cannon/research/parl_stability/voronoi.hpp>

using namespace cannon::research::parl;

VD cannon::research::parl::compute_voronoi_diagram(const MatrixXd& refs) {
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

  assert(diagram.is_valid());

  return diagram;
}

std::map<unsigned int, Polygon_2>
cannon::research::parl::create_bounded_voronoi_polygons(const MatrixXd& refs, VD
    diagram) {

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
      if (hit != end) { 
        // We should not have any holes for Voronoi regions under affine maps
        throw std::runtime_error("Expected a simple face, but found a hole.");
      }

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
    default:
      return false;
  }  
}
