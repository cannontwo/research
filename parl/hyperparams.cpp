#include <cannon/research/parl/hyperparams.hpp>

using namespace cannon::research::parl;

void Hyperparams::load_config(const std::string& filename) {
  YAML::Node config = YAML::LoadFile(filename);

  if (!config["agent"])
    throw std::runtime_error("Config file did not have an 'agent' map.");

  YAML::Node agent_params = config["agent"];

  if (agent_params["discount_factor"])
    discount_factor = agent_params["discount_factor"].as<double>();
  if (agent_params["alpha"])
    alpha = agent_params["alpha"].as<double>();
  if (agent_params["forgetting_factor"])
    forgetting_factor = agent_params["forgetting_factor"].as<double>();
  if (agent_params["controller_learning_rate"])
    controller_learning_rate = agent_params["controller_learning_rate"].as<double>();

  if (agent_params["use_clipping"])
    use_clipping = agent_params["use_clipping"].as<bool>();
  if (agent_params["use_line_search"])
    use_line_search = agent_params["use_line_search"].as<bool>();
  if (agent_params["clip_line_search"])
    clip_line_search = agent_params["clip_line_search"].as<bool>();
  if (agent_params["use_adam"])
    use_adam = agent_params["use_adam"].as<bool>();

  if (agent_params["states_periodic"]) {
    YAML::Node sp = agent_params["states_periodic"];
    if (sp.Type() != YAML::NodeType::Sequence)
      throw std::runtime_error("states_periodic in YAML config was not a sequence");

    states_periodic.resize(sp.size());
    for (size_t i = 0; i < sp.size(); i++) {
      states_periodic[i] = sp[i].as<bool>();
    } 
  }
}
