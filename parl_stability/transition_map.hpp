#ifndef CANNON_RESEARCH_PARL_STABILITY_TRANSITION_MAP_H
#define CANNON_RESEARCH_PARL_STABILITY_TRANSITION_MAP_H 

#include <vector>

#include <CGAL/Polygon_2.h>
#include <CGAL/Aff_transformation_2.h>
#include <CGAL/Boolean_set_operations_2.h>

#include <cannon/research/parl_stability/voronoi.hpp>

using Polygon_2 = CGAL::Polygon_2<K>;
using Polygon_with_holes_2 = CGAL::Polygon_with_holes_2<K>;
using Transformation = CGAL::Aff_transformation_2<K>;

namespace cannon {
  namespace research {
    namespace parl {

      // Used to convert otherwise infinite rays into long segments
      const static int RAY_LENGTH = 1000;

      std::map<std::pair<unsigned int, unsigned int>,
        Polygon_2> compute_transition_map(std::shared_ptr<Parl> parl, VD
            diagram);

      std::map<unsigned int, Polygon_2> create_bounded_voronoi_polygons(std::shared_ptr<Parl> parl, VD diagram);

      std::pair<Polygon_2, Transformation> affine_map_polygon(const Polygon_2& p, 
          const AutonomousLinearParams& map);

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_STABILITY_TRANSITION_MAP_H */

