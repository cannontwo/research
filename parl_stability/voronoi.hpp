#ifndef CANNON_RESEARCH_PARL_VORONOI
#define CANNON_RESEARCH_PARL_VORONOI

#include <Eigen/Dense>

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Nef_polyhedron_2.h>
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/Extended_cartesian.h>
#include <CGAL/Polygon_set_2.h>

#include <cannon/log/registry.hpp>

using K = CGAL::Exact_predicates_exact_constructions_kernel;
using DT = CGAL::Delaunay_triangulation_2<K>;
using AT = CGAL::Delaunay_triangulation_adaptation_traits_2<DT>;
using AP = CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<DT>;
using VD = CGAL::Voronoi_diagram_2<DT, AT, AP>;

using Polygon_2 = CGAL::Polygon_2<K>;
using Polygon_with_holes_2 = CGAL::Polygon_with_holes_2<K>;
using Polygon_set_2 = CGAL::Polygon_set_2<K>;
using Transformation = CGAL::Aff_transformation_2<K>;

using Extended_kernel = CGAL::Extended_cartesian<K::FT>;
using Nef_polyhedron = CGAL::Nef_polyhedron_2<Extended_kernel>;
using Halfedge_around_face_const_circulator = Nef_polyhedron::Explorer::Halfedge_around_face_const_circulator;
using Hole_const_iterator = Nef_polyhedron::Explorer::Hole_const_iterator;
using Vertex_const_handle = Nef_polyhedron::Explorer::Vertex_const_handle;

using namespace Eigen;

using namespace cannon::log;

namespace cannon {
  namespace research {
    namespace parl {

      VD compute_voronoi_diagram(const MatrixXd& refs);

      /*!
       * Create bounded polygons from the input voronoi diagram corresponding
       * to the references for the input PARL agent.
       *
       * \param refs Reference points for the PARL agent whose reference points
       * generated the input Voronoi diagram.
       * \param diagram The Voronoi diagram.
       *
       * \returns A map from reference index to Voronoi polygon, for only the
       * bounded polygons in the Voronoi diagram.
       */
      std::map<unsigned int, Polygon_2> create_bounded_voronoi_polygons(const
          MatrixXd& refs, VD diagram);

      /*!
       * Extracts a polygon from a Nef_polyhedron explorer which is assumed to
       * contain a single finite face. This is specifically used for extracting
       * controller saturation regions on top of the Parl Voronoi diagram, so
       * its usage in other scenarios is probably sketchy.
       *
       * \param e Explorer corresponding to a planar Nef_polyhedron map.
       *
       * \returns A polygon representing the single finite face.
       */
      Polygon_2 extract_finite_face_polygon(const Nef_polyhedron::Explorer& e);

      /*!
       * Check whether input state is in input polygon.
       * 
       * \param state The state the check
       * \param poly The polygon to check for containment
       *
       * \return Whether the polygon contains the input state.
       */
      bool is_inside(const Vector2d& state, const Polygon_2& poly);

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_VORONOI */
