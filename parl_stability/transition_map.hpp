#ifndef CANNON_RESEARCH_PARL_STABILITY_TRANSITION_MAP_H
#define CANNON_RESEARCH_PARL_STABILITY_TRANSITION_MAP_H 

#include <vector>

#include <CGAL/Polygon_2.h>
#include <CGAL/Aff_transformation_2.h>
#include <CGAL/Boolean_set_operations_2.h>

#include <CGAL/Nef_polyhedron_2.h>
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/Extended_cartesian.h>

#include <cannon/research/parl_stability/voronoi.hpp>

using Polygon_2 = CGAL::Polygon_2<K>;
using Polygon_with_holes_2 = CGAL::Polygon_with_holes_2<K>;
using Transformation = CGAL::Aff_transformation_2<K>;

using Extended_kernel = CGAL::Extended_cartesian<K::FT>;
using Nef_polyhedron = CGAL::Nef_polyhedron_2<Extended_kernel>;
using Halfedge_around_face_const_circulator = Nef_polyhedron::Explorer::Halfedge_around_face_const_circulator;
using Hole_const_iterator = Nef_polyhedron::Explorer::Hole_const_iterator;
using Vertex_const_handle = Nef_polyhedron::Explorer::Vertex_const_handle;

namespace cannon {
  namespace research {
    namespace parl {

      // Used to convert otherwise infinite rays into long segments
      const static int RAY_LENGTH = 1000;

      /*!
       * Create the transition map corresponding to the controlled system
       * learned by the input PARL agent on the input Voronoi diagram.
       *
       * \param parl The PARL agent.
       * \param diagram The PARL reference point Voronoi diagram.
       *
       * \returns A map from pairs of indices (i, j) to polygons representing
       * regions mapped from Voronoi region i to Voronoi region j by the PARL
       * controlled system.
       */
      std::map<std::pair<unsigned int, unsigned int>,
        Polygon_2> compute_transition_map(std::shared_ptr<Parl> parl, VD
            diagram);

      /*!
       * Create bounded polygons from the input voronoi diagram corresponding
       * to the references for the input PARL agent.
       *
       * \param parl The PARL agent whose reference points generated the input
       * Voronoi diagram.
       * \param diagram The Voronoi diagram.
       *
       * \returns A map from reference index to Voronoi polygon, for only the
       * bounded polygons in the Voronoi diagram.
       */
      std::map<unsigned int, Polygon_2>
        create_bounded_voronoi_polygons(std::shared_ptr<Parl> parl, VD
            diagram);

      /*!
       * Map the input polygon using the input affine map, and return the
       * mapped polygon and inverse affine transformation.
       *
       * \param p The polygon to map.
       * \param map The affine map to apply.
       *
       * \returns A pair composed of the mapped polygon and the inverse affine
       * transformation.
       */
      std::pair<Polygon_2, Transformation> affine_map_polygon(const Polygon_2& p, 
          const AutonomousLinearParams& map);

      /*!
       * Get polygons representing areas where control is linear and min/max
       * saturated. This is specifically for the R^2 state space, R^1 control
       * space case (i.e. inverted pendulum). For more general cases, more work
       * is needed.
       *
       * \returns Tuple of polygonal regions. First return value is the linear
       * region, second is the min-control saturated region, and third is the
       * max-control saturated region.
       */
      std::tuple<Polygon_2, Polygon_2, Polygon_2> get_saturated_polygons(const
          Polygon_2& voronoi_polygon, const RowVector2d& K, double k, double
          lower=-2.0, double upper=2.0);

      /*!
       * TODO
       */
      Polygon_2 extract_finite_face_polygon(const Nef_polyhedron::Explorer& e);

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_STABILITY_TRANSITION_MAP_H */

