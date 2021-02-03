list(APPEND RESEARCH_SOURCES
  parl/parl.cpp
  parl/hyperparams.cpp
  parl/runner.cpp
  parl_planning/executor.cpp
  parl_planning/parl_planner.cpp
  parl_planning/aggregate_model.cpp
  parl_planning/ompl_utils.cpp
  parl_stability/voronoi.cpp
  )

add_library(research OBJECT ${RESEARCH_SOURCES})
