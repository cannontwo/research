#ifndef CANNON_RESEARCH_PARL_HYPERPARAMS_H
#define CANNON_RESEARCH_PARL_HYPERPARAMS_H 

#include <valarray>
#include <string>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace cannon {
  namespace research {
    namespace parl {

      struct Hyperparams {

          Hyperparams() {}

          void load_config(const std::string& filename);

          double discount_factor = 0.95;
          double alpha = 1e4;
          double forgetting_factor = 1.0;
          double controller_learning_rate = 1.0;

          bool use_clipping = true;
          bool use_line_search = true;
          bool clip_line_search = false;
          bool use_adam = true;

          std::valarray<bool> states_periodic;
      };

    } // namespace parl
  } // namespace research
} // namespace cannon

#endif /* ifndef CANNON_RESEARCH_PARL_HYPERPARAMS_H */
