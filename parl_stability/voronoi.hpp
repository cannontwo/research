#ifndef CANNON_RESEARCH_PARL_VORONOI
#define CANNON_RESEARCH_PARL_VORONOI

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>

#include <cannon/research/parl/parl.hpp>

using K = CGAL::Exact_predicates_exact_constructions_kernel;
using DT = CGAL::Delaunay_triangulation_2<K>;
using AT = CGAL::Delaunay_triangulation_adaptation_traits_2<DT>;
using AP = CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<DT>;
using VD = CGAL::Voronoi_diagram_2<DT, AT, AP>;

namespace cannon {
  namespace research {
    namespace parl {

      VD compute_voronoi_diagram(std::shared_ptr<Parl> parl);

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_VORONOI */
