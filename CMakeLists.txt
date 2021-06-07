list(APPEND RESEARCH_SOURCES
  parl/parl.cpp
  parl/hyperparams.cpp
  parl/runner.cpp
  parl_planning/executor.cpp
  parl_planning/aggregate_model.cpp
  parl_planning/ompl_utils.cpp
  parl_planning/lqr_control_space.cpp
  parl_stability/voronoi.cpp
  parl_stability/transition_map.cpp
  parl_stability/lyapunov_finding.cpp
  )

add_library(research OBJECT ${RESEARCH_SOURCES})
