#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <vector>
#include <sstream>

#include <Eigen/Dense>

#include <cannon/research/parl/parl.hpp>
#include <cannon/research/parl/envs/inverted_pendulum.hpp>

using namespace Eigen;

using namespace cannon::research::parl;

struct TrajElem {
  VectorXd state;
  VectorXd action;
  double reward;
  bool done;
};

std::vector<TrajElem> read_traj_csv(const std::string& filename) {
  std::ifstream traj_file(filename);

  std::vector<TrajElem> traj;

  std::string line, word;
  std::vector<std::string> row;
  for (int i = 0; i < 201; i++) {

    row.clear();
    std::getline(traj_file, line);
    std::stringstream ss(line);

    if (i == 0)
      continue;

    while (std::getline(ss, word, ',')) {
      row.push_back(word);
    }

    assert(row.size() == 5);

    TrajElem tmp;

    tmp.state = VectorXd::Zero(2);
    tmp.state[0] = std::stod(row[0]);
    tmp.state[1] = std::stod(row[1]);

    tmp.action = VectorXd::Zero(1);
    tmp.action[0] = std::stod(row[2]);

    tmp.reward = std::stod(row[3]);
    tmp.done = row[4] != "0";

    traj.push_back(tmp);

  }

  return traj;
}

int main(int argc, char** argv) {
  assert(argc >= 2);

  auto traj = read_traj_csv(argv[1]);
  assert(traj.size() == 200);

  InvertedPendulumEnvironment env;
  Hyperparams params;
  params.load_config("/home/cannon/Documents/cannon/cannon/research/experiments/parl_configs/r10c10_short.yaml");
  Parl model(env.get_state_space(), env.get_action_space(), env.sample_grid_refs({10, 10}), params, 0, false);

  for (int i = 0; i < 199; i++) {
    model.process_datum(traj[i].state, traj[i].action, traj[i].reward, traj[i+1].state);
  }

  for (int i = 0; i < 199; i++) {
    log_info("");
    model.value_grad_update_controller(traj[i].state);
  }

  for (int i = 0; i < 199; i++) {
    log_info("At timestep", i);
    log_info("\tValue model predicts value", model.predict_value(traj[i].state));
    log_info("\tDynamics model predicts value", model.predict_next_state(traj[i].state, traj[i].action).transpose());
    log_info("\tControl after update is", model.get_unconstrained_action(traj[i].state));
    log_info("");
  }
}
