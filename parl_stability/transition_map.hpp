#ifndef CANNON_RESEARCH_PARL_STABILITY_TRANSITION_MAP_H
#define CANNON_RESEARCH_PARL_STABILITY_TRANSITION_MAP_H 

#include <vector>
#include <filesystem>

#include <CGAL/Polygon_2.h>
#include <CGAL/Aff_transformation_2.h>
#include <CGAL/Boolean_set_operations_2.h>

#include <CGAL/Nef_polyhedron_2.h>
#include <CGAL/Lazy_exact_nt.h>
#include <CGAL/Extended_cartesian.h>
#include <CGAL/Polygon_set_2.h>

#include <Eigen/Dense>
#include <thirdparty/HighFive/include/highfive/H5Easy.hpp>

#include <cannon/research/parl_stability/voronoi.hpp>
#include <cannon/research/parl/linear_params.hpp>
#include <cannon/research/parl/parl.hpp>
#include <cannon/plot/plotter.hpp>
#include <cannon/graphics/random_color.hpp>

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

using namespace cannon::plot;
using namespace cannon::graphics;

namespace cannon {
  namespace research {
    namespace parl {

      using PWAFunc = std::vector<std::pair<Polygon_2, AutonomousLinearParams>>;
      using TransitionMap = std::map<std::pair<unsigned int, unsigned int>, std::vector<Polygon_2>>;
      using OutMap = std::map<unsigned int, std::vector<Polygon_2>>;

      /*!
       * Compute transition map of the input PWA system.
       */
      std::pair<TransitionMap, OutMap> compute_transition_map(const
          std::vector<std::pair<Polygon_2, AutonomousLinearParams>>& pwa_func);

      /*!
       * Compute transition map of the input PWA system, using the transition
       * information of an old PWA map and the correspondence induced by the
       * input multimap. This is more efficient than the version that can't use
       * the previous transition map.
       */
      std::pair<TransitionMap, OutMap> compute_transition_map(const PWAFunc&
          pwa_func, const TransitionMap& old_transition_map, const
          std::multimap<unsigned int, unsigned int>& correspondence);

      /*!
       * Compute the effective PWA controlled system represented by a PARL
       * controller over estimated dynamics, taking into account controller
       * saturation.
       *
       * \param parl The PARL agent.
       * \param diagram The PARL reference point Voronoi diagram.
       *
       * \returns A vector of Polygons and associated dynamics, extracted from
       * the Parl-controlled system.
       */
      std::vector<std::pair<Polygon_2, AutonomousLinearParams>>
        compute_parl_pwa_func(std::shared_ptr<Parl> parl, VD diagram);

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
       * Compute the preimage of the intersection of map_poly transformed by
       * the input affine map and test_poly. This function is specifically
       * designed for well-behaved Voronoi polygons, but should be
       * generalizable.
       *
       * \param map_poly The polygon to be affine mapped.
       * \param test_poly The polygon to test intersection against.
       * \param map The affine map to apply to map_poly.
       *
       * \returns A vector of preimage polygons of the intersection of map(map_poly) and test_poly.
       */
      std::vector<Polygon_2> compute_premap_set(const Polygon_2& map_poly, const Polygon_2&
          test_poly, const AutonomousLinearParams& map);

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
       * Plot transition map and display plot.
       *
       * \param transition_map_pair The transition map to plot.
       * \param pwa_func The PWA function over which the transition map is defined
       */
      void plot_transition_map(const std::pair<TransitionMap, OutMap>&
          transition_map_pair, const PWAFunc& pwa_func, bool save=true);

      /*!
       * Save a PWA function to an HDF5 file. 
       *
       * The structure of the stored data is as follows:
       *
       * /polygons/i - Polygon information, stored as a vector of points
       *               assumed to be in CCW order, with an entry for each region.
       * /A_mats/i - Linear portions of autonomous system dynamics.
       * /c_vecs/i - Offset portions of autonomous system dynamics.
       *
       * \param pwa The PWA function to save.
       * \param path The file path to save it to.
       */
      void save_pwa(const PWAFunc& pwa, const std::string& path);

      /*!
       * Load a PWA function from an HDF5 file. 
       *
       * The structure of the stored data is as follows:
       *
       * /polygons/i - Polygon information, stored as a vector of points
       *               assumed to be in CCW order, with an entry for each region.
       * /A_mats/i - Linear portions of autonomous system dynamics.
       * /c_vecs/i - Offset portions of autonomous system dynamics.
       *
       * \param path The file path to save it to.
       *
       * \return The loaded PWA function.
       */
      PWAFunc load_pwa(const std::string& path);

      /*!
       * Restrict input PWA by only including regions entirely within the input radius.
       *
       * \param pwa The PWA to restrict.
       * \param radius The around 0 to restrict the PWA to.
       *
       * \returns The restricted PWA.
       */
      PWAFunc restrict_pwa(const PWAFunc& pwa, double radius);

      Vector2d evaluate_pwa(const PWAFunc& pwa, const Vector2d& query);

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_STABILITY_TRANSITION_MAP_H */

