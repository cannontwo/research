list(APPEND RESEARCH_SOURCES
  parl/parl.cpp
  parl/hyperparams.cpp
  parl/runner.cpp
  parl_planning/error_space_executor.cpp
  parl_planning/lqr_executor.cpp
  parl_planning/aggregate_model.cpp
  parl_planning/ompl_utils.cpp
  parl_planning/lqr_control_space.cpp
  parl_planning/lqr_state_propagator.cpp
  parl_stability/voronoi.cpp
  parl_stability/transition_map.cpp
  parl_stability/lyapunov_finding.cpp
  )

add_library(research OBJECT ${RESEARCH_SOURCES})
target_compile_options(research PUBLIC -Wextra -Werror -Wno-error=deprecated-copy)
