#ifndef CANNON_RESEARCH_PARL_STABILITY_LYAPUNOV_FINDING_H
#define CANNON_RESEARCH_PARL_STABILITY_LYAPUNOV_FINDING_H 

#include <Eigen/Dense>

#include <cannon/research/parl_stability/transition_map.hpp>
#include <cannon/ml/linear_programming.hpp>

using namespace Eigen;

using namespace cannon::ml;

namespace cannon {
  namespace research {
    namespace parl {
      struct LyapunovComponent {

        LyapunovComponent() = delete;

        LyapunovComponent(const Polygon_2& poly, const RowVectorXd& linear, double affine) :
          poly_(poly), linear_part_(linear), affine_part_(affine) {}

        LyapunovComponent(const LyapunovComponent &o) : poly_(o.poly_),
        linear_part_(o.linear_part_), affine_part_(o.affine_part_) {}

        LyapunovComponent(LyapunovComponent &&o) :
          poly_(std::move(o.poly_)),
          linear_part_(std::move(o.linear_part_)),
          affine_part_(std::move(o.affine_part_)) {}

        ~LyapunovComponent() {}

        double evaluate(const VectorXd& query);

        Polygon_2 poly_;
        RowVectorXd linear_part_;
        double affine_part_;
      };

      std::pair<std::vector<LyapunovComponent>, double> attempt_lp_solve(const PWAFunc& pwa, const
          TransitionMap& transition_map, const OutMap& out_map, double M=100, double eps=1e-6);

      std::tuple<std::vector<LyapunovComponent>, PWAFunc, double> find_lyapunov(const PWAFunc& pwa, const
          TransitionMap& initial_transition_map, const OutMap& initial_out_map,
          unsigned int max_iters=10);

      std::tuple<PWAFunc, TransitionMap, OutMap> refine_pwa(const PWAFunc& pwa,
          const TransitionMap& transition_map, const OutMap& out_map);

      double evaluate_lyap(std::vector<LyapunovComponent> lyap, const Vector2d& query);

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_STABILITY_LYAPUNOV_FINDING_H */
