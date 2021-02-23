#include <cannon/research/parl_stability/lyapunov_finding.hpp>

using namespace cannon::research::parl;

std::vector<LyapunovComponent> cannon::research::parl::attempt_lp_solve(const
    PWAFunc& pwa, const TransitionMap& transition_map, const OutMap& out_map,
    double M, double eps) {

  // var[0] is \alpha_1, var[1] is \alpha_3, var[2] is \theta
  // var[3*i + 3] is F_i[0], var[3*i + 4] is F_i[1], var[3*i + 5] is g_i
  unsigned int num_vars = 3 * pwa.size() + 3; 

  VectorXd lower = VectorXd::Ones(num_vars) * -1e6;
  lower[0] = eps; // \alpha_1 > 0
  lower[1] = eps; // \alpha_3 > 0

  VectorXd upper = VectorXd::Ones(num_vars) * 1e6;

  // Find polygon closures containing 0 and add upper and lower constraints = 0
  for (unsigned int i = 0; i < pwa.size(); i++) {
    if (is_inside(Vector2d::Zero(), pwa[i].first)) {
      lower[3*i + 5] = 0.0; 
      upper[3*i + 5] = 0.0; 
    }
  }

  // Objective is to maximize \theta, and LPOptimizer minimizes by default
  VectorXd objective = VectorXd::Zero(num_vars);
  objective[2] = -1.0; 

  LPOptimizer opt(lower, upper, objective);

  // Add constraints to optimizer
  
  // Constraint (8a)
  for (unsigned int i = 0; i < pwa.size(); i++) {
    for (auto it = pwa[i].first.vertices_begin(); it < pwa[i].first.vertices_end(); it++) {
      RowVectorXd lhs_mat = VectorXd::Zero(num_vars);
      double x = CGAL::to_double(it->x());
      double y = CGAL::to_double(it->y());

      // alpha_1 term
      lhs_mat[0] = std::sqrt(std::pow(x, 2) + std::pow(y, 2));

      // -F_i * v_{i, h} term
      lhs_mat[3*i + 3] = -x;
      lhs_mat[3*i + 4] = -y;

      // -g_i term
      lhs_mat[3*i + 5] = -1.0;

      VectorXd rhs = VectorXd::Zero(1);

      opt.add_constraint(lhs_mat, rhs);
    }
  }

  // Constraint (8c)
  for (unsigned int i = 0; i < pwa.size(); i++) {
    for (auto it = pwa[i].first.vertices_begin(); it < pwa[i].first.vertices_end(); it++) {
      RowVectorXd lhs_mat = VectorXd::Zero(num_vars);
      double x = CGAL::to_double(it->x());
      double y = CGAL::to_double(it->y());

      // F_i * v_{i, h} term
      lhs_mat[3*i + 3] = x;
      lhs_mat[3*i + 4] = y;

      // g_i term
      lhs_mat[3*i + 5] = 1.0;

      VectorXd rhs = VectorXd::Ones(1) * M;

      opt.add_constraint(lhs_mat, rhs);
    }
  }
   
  // Constraint (8e)
  for (auto& pair : transition_map) {
    unsigned int i, j;
    std::tie(i, j) = pair.first;
    for (auto it = pair.second.vertices_begin(); it < pair.second.vertices_end(); it++) {
      RowVectorXd lhs_mat = VectorXd::Zero(num_vars);
      Vector2d vertex = Vector2d::Zero();
      vertex[0] = CGAL::to_double(it->x());
      vertex[1] = CGAL::to_double(it->y());

      // F_j * (A_i v_{ij, h} + a_i) term
      Vector2d interior = pwa[i].second.A_ * vertex + pwa[i].second.c_;
      lhs_mat[3*j + 3] = interior[0];
      lhs_mat[3*j + 4] = interior[1];

      // g_j term
      lhs_mat[3*j + 5] = 1.0;

      // -F_i v_{ij,h} term
      lhs_mat[3*i + 3] = -vertex[0];
      lhs_mat[3*i + 4] = -vertex[1];

      // -g_i term
      lhs_mat[3*i + 5] = -1.0;

      // \alpha_3 term
      lhs_mat[1] = std::sqrt(std::pow(vertex[0], 2) + std::pow(vertex[1], 2));

      VectorXd rhs = VectorXd::Zero(1);

      opt.add_constraint(lhs_mat, rhs);
    }
  }
  
  // Constraint (8g)
  for (auto& pair : out_map) {
    unsigned int i = pair.first;
    for (auto& poly : pair.second) {
      for (auto it = poly.vertices_begin(); it < poly.vertices_end(); it++) {
        RowVectorXd lhs_mat = VectorXd::Zero(num_vars);
        double x = CGAL::to_double(it->x());
        double y = CGAL::to_double(it->y());

        // \theta term
        lhs_mat[2] = 1.0;

        // -F_i v_{ip,h}
        lhs_mat[3*i + 3] = -x;
        lhs_mat[3*i + 4] = -y;

        // -g_i
        lhs_mat[3*i + 5] = -1.0;

        VectorXd rhs = VectorXd::Zero(1);

        opt.add_constraint(lhs_mat, rhs);
      }
    }
  }
 
  log_info("Attempting to solve PWA Lyapunov LP");

  // This will throw a std::runtime_error if the LP is infeasible
  auto result = opt.optimize();
  log_info("Found solution to Lyapunov LP with objective", result.objective);

  std::vector<LyapunovComponent> lyap;
  for (unsigned int i = 0; i < pwa.size(); i++) {
    RowVector2d F = RowVector2d::Zero();
    F[0] = result.solution[3*i + 3];
    F[1] = result.solution[3*i + 4];

    LyapunovComponent component(pwa[i].first, F, result.solution[3*i + 5]);
    lyap.push_back(component);
  }

  return lyap;
    
}

std::vector<LyapunovComponent> cannon::research::parl::find_lyapunov(const
    PWAFunc& pwa, const TransitionMap& initial_transition_map, const OutMap&
    initial_out_map, unsigned int max_iters) {

  auto current_pwa = pwa;
  auto current_transition_map = initial_transition_map;
  auto current_out_map = initial_out_map;

  for (unsigned int i = 0; i < max_iters; i++) {
    try {
      log_info("On iteration", i, "attempting to solve LP for PWA func with", pwa.size(), "regions");
      auto lyap = attempt_lp_solve(current_pwa, current_transition_map, current_out_map);

      // If we got here, solve was successful!
      return lyap;

    } catch (...) {
      log_info("LP infeasible, refining polygon map");

      // TODO Adjust current_pwa, current_transition_map, current_out_map. New
      // PWA is same affine maps over polygons of current_transition_map and
      // current_out_map, new transition/out map computed by
      // compute_transition_map
      
      // Construct new PWA using polygons of transition map
      PWAFunc new_pwa;
      for (auto& pair : current_transition_map) {
        unsigned int i = pair.first.first;  

        // X_{ij} experiences X_i dynamics
        new_pwa.push_back(std::make_pair(pair.second, current_pwa[i].second));
      }

      for (auto& pair : current_out_map) {
        unsigned int i = pair.first;

        for (auto& poly : pair.second) {
          // \Omega_{ip} experiences X_i dynamics
          new_pwa.push_back(std::make_pair(poly, current_pwa[i].second));
        }
      }

      auto new_transition_map_pair = compute_transition_map(new_pwa);
      plot_transition_map(new_transition_map_pair, new_pwa);

      current_pwa = new_pwa;
      current_transition_map = new_transition_map_pair.first;
      current_out_map = new_transition_map_pair.second;
    }
  }

  throw std::runtime_error("Could not solve Lyapunov LP within maximum iterations");
}

double cannon::research::parl::evaluate_lyap(std::vector<LyapunovComponent> lyap,
    const Vector2d& query) {

  bool contained = false;
  double value = 0.0;
  for (auto& component : lyap) {
    if (is_inside(query, component.poly_)) {
      value = std::max(value, component.linear_part_ * query + component.affine_part_);
      contained = true;
    }
  }

  if (!contained) {
    throw std::runtime_error("Query point not in domain of Lyapunov function"); 
  }

  return value;
}
