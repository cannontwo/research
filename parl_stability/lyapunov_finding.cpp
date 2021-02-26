#include <cannon/research/parl_stability/lyapunov_finding.hpp>

using namespace cannon::research::parl;

std::pair<std::vector<LyapunovComponent>, double>
cannon::research::parl::attempt_lp_solve(const PWAFunc& pwa, const
    TransitionMap& transition_map, const OutMap& out_map, double M, double eps) {

  // var[0] is \alpha_1, var[1] is \alpha_3, var[2] is \theta
  // var[3*i + 3] is F_i[0], var[3*i + 4] is F_i[1], var[3*i + 5] is g_i
  unsigned int num_vars = 3 * pwa.size() + 3; 

  VectorXd lower = VectorXd::Ones(num_vars) * -1e3;
  lower[0] = eps; // \alpha_1 > 0
  lower[1] = eps; // \alpha_3 > 0
  lower[2] = eps; // \theta > 0 is implied

  VectorXd upper = VectorXd::Ones(num_vars) * 1e3;

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
    for (auto& poly : pair.second) {
      for (auto it = poly.vertices_begin(); it < poly.vertices_end(); it++) {
        RowVectorXd lhs_mat = VectorXd::Zero(num_vars);
        Vector2d vertex = Vector2d::Zero();
        vertex[0] = CGAL::to_double(it->x());
        vertex[1] = CGAL::to_double(it->y());

        // F_j * (A_i v_{ij, h} + a_i) term
        Vector2d interior = pwa[i].second.A_ * vertex + pwa[i].second.c_;
        lhs_mat[3*j + 3] = interior[0];
        lhs_mat[3*j + 4] = interior[1];

        // TODO This might be incorrect, but it also might be necessary to
        // avoid numerical issues with the LP solving
        if (std::fabs(lhs_mat[3*j + 3]) < 1e-6) {
          log_info("Very small interior x constraint coeff, adding as zero");
          lhs_mat[3*j + 3] = 0.0;
        }
        if (std::fabs(lhs_mat[3*j + 4]) < 1e-6) {
          log_info("Very small interior y constraint coeff, adding as zero");
          lhs_mat[3*j + 4] = 0.0;
        }

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

  log_info("Retrieved alpha_1=", result.solution[0], ", alpha_3=",
      result.solution[1], ", theta=", result.solution[2]);

  std::vector<LyapunovComponent> lyap;
  for (unsigned int i = 0; i < pwa.size(); i++) {
    RowVector2d F = RowVector2d::Zero();
    F[0] = result.solution[3*i + 3];
    F[1] = result.solution[3*i + 4];

    LyapunovComponent component(pwa[i].first, F, result.solution[3*i + 5]);
    lyap.push_back(component);
  }

  return std::make_pair(lyap, result.solution[2]);
    
}

std::tuple<std::vector<LyapunovComponent>, PWAFunc, double>
cannon::research::parl::find_lyapunov(const PWAFunc& pwa, const TransitionMap&
    initial_transition_map, const OutMap& initial_out_map, unsigned int
    max_iters) {

  auto current_pwa = pwa;
  auto current_transition_map = initial_transition_map;
  auto current_out_map = initial_out_map;

  for (unsigned int i = 0; i < max_iters; i++) {
    try {
      log_info("On iteration", i, "attempting to solve LP for PWA func with", current_pwa.size(), "regions");

      std::vector<LyapunovComponent> lyap;
      double theta;
      std::tie(lyap, theta) = attempt_lp_solve(current_pwa, current_transition_map, current_out_map);

      log_info("LP solved with theta=", theta);

      // If we got here, solve was successful!

      // Store found Lyapunov function
      std::filesystem::create_directory("models");
      save_lyap(lyap, theta, std::string("models/lyap_") +
          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
          + std::string(".h5"));

      return std::make_tuple(lyap, current_pwa, theta);

    } catch (...) {
      log_info("LP infeasible, refining polygon map");
    }

    std::tie(current_pwa, current_transition_map, current_out_map) =
      refine_pwa(current_pwa, current_transition_map, current_out_map);

    log_info("After refinement, transition map has",
        current_transition_map.size(), "elements and out map has",
        current_out_map.size(), "elements");
  }

  throw std::runtime_error("Could not solve Lyapunov LP within maximum iterations");
}


std::tuple<PWAFunc, TransitionMap, OutMap>
cannon::research::parl::refine_pwa(const PWAFunc& pwa, const TransitionMap&
    transition_map, const OutMap& out_map) {

    PWAFunc new_pwa;
    for (auto& pair : transition_map) {
      unsigned int i = pair.first.first;  

      for (auto& poly : pair.second) {
        // X_{ij} experiences X_i dynamics
        new_pwa.push_back(std::make_pair(poly, pwa[i].second));
      }
    }

    // TODO I'm testing this; I think it should be alright to remove polygons
    // that get mapped out of bounds from consideration, since they cannot be
    // in the PI set by definition.
    //
    for (auto& pair : out_map) {
      unsigned int i = pair.first;

      for (auto& poly : pair.second) {
        // \Omega_{ip} experiences X_i dynamics
        new_pwa.push_back(std::make_pair(poly, pwa[i].second));
      }
    }
    //
    // TODO End testing
    

    // Store refined PWA function
    std::filesystem::create_directory("models");
    save_pwa(new_pwa, std::string("models/pwa_") +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
        + std::string(".h5"));

    // TODO Make more efficient transition map computation that uses previous
    // transition map information
    auto new_transition_map_pair = compute_transition_map(new_pwa);
    plot_transition_map(new_transition_map_pair, new_pwa);

    return std::make_tuple(new_pwa, new_transition_map_pair.first, new_transition_map_pair.second);
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

void cannon::research::parl::save_lyap(std::vector<LyapunovComponent>& lyap, double theta,
    const std::string& path) {
  H5Easy::File file(path, H5Easy::File::Overwrite);
  std::string polygon_path("/polygons/");
  std::string linear_path("/linear/");
  std::string affine_path("/affine/");

  log_info("Saving Lyapunov function with", lyap.size(), "components");

  for (unsigned int i = 0; i < lyap.size(); i++) {
    std::vector<Vector2d> poly_vec;
    for (auto it = lyap[i].poly_.vertices_begin(); it < lyap[i].poly_.vertices_end(); it++) {
      Vector2d vec = Vector2d::Zero();
      vec[0] = CGAL::to_double(it->x());
      vec[1] = CGAL::to_double(it->y());

      poly_vec.push_back(vec);
    }

    H5Easy::dump(file, polygon_path + std::to_string(i), poly_vec);
    H5Easy::dump(file, linear_path + std::to_string(i), lyap[i].linear_part_);
    H5Easy::dump(file, affine_path + std::to_string(i), lyap[i].affine_part_);
  }

  H5Easy::dump(file, "/theta", theta);

  file.flush();
}

std::pair<std::vector<LyapunovComponent>, double>
cannon::research::parl::load_lyap(const std::string& path) {
  std::vector<LyapunovComponent> ret_lyap;

  H5Easy::File file(path, H5Easy::File::ReadOnly);

  auto polygon_group = file.getGroup("/polygons");
  auto linear_group = file.getGroup("/linear");
  auto affine_group = file.getGroup("/affine");

  log_info("/polygons group has", polygon_group.getNumberObjects(), "objects");

  assert(polygon_group.getNumberObjects() == linear_group.getNumberObjects());
  assert(polygon_group.getNumberObjects() == affine_group.getNumberObjects());

  std::string polygon_path("/polygons/");
  std::string linear_path("/linear/");
  std::string affine_path("/affine/");

  for (unsigned int i = 0; i < polygon_group.getNumberObjects(); i++) {

    // Read polygon for region i
    std::vector<Vector2d> poly_vec = H5Easy::load<std::vector<Vector2d>>(file, polygon_path + std::to_string(i));
    Polygon_2 poly;
    for (unsigned int i = 0; i < poly_vec.size(); i++) {
      Vector2d eig_p = poly_vec[i];
      K::Point_2 p(eig_p[0], eig_p[1]);
      poly.push_back(p);
    }

    // Read Lyapunov function parameters for region i
    RowVectorXd linear = H5Easy::load<RowVectorXd>(file, linear_path + std::to_string(i));
    double affine = H5Easy::load<double>(file, affine_path + std::to_string(i));

    LyapunovComponent tmp(poly, linear, affine);
    ret_lyap.push_back(tmp);
  }

  double theta = H5Easy::load<double>(file, "/theta");

  return std::make_pair(ret_lyap, theta);
}
