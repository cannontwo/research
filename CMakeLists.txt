list(APPEND RESEARCH_SOURCES
  parl/parl.cpp
  parl/hyperparams.cpp
  parl/runner.cpp
  parl_planning/executor.cpp
  parl_planning/parl_planner.cpp
  parl_planning/aggregate_model.cpp
  )

add_library(research OBJECT ${RESEARCH_SOURCES})
