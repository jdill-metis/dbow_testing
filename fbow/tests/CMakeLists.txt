# CMake file for sba's demo program
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/tests/dbow2 ${PROJECT_SOURCE_DIR}/src/tests/dbow3)
LINK_LIBRARIES(${PROJECT_NAME})
 

# ADD_EXECUTABLE(fbow_test_dbow2VSfbow dbow2fbow.cpp dbow2/ScoringObject.cpp dbow2/BowVector.h  dbow2/FClass.h  dbow2/FeatureVector.h   dbow2/FORB.h  dbow2/ScoringObject.h  dbow2/TemplatedVocabulary.h)
ADD_EXECUTABLE(fbow_test_dbow2VSfbow  dbow2fbow.cpp  dbow3/BowVector.h  dbow3/BowVector.cpp  dbow3/Database.cpp  dbow3/Database.h  dbow3/DBoW3.h  dbow3/DescManip.cpp  dbow3/DescManip.h  dbow3/exports.h  dbow3/FeatureVector.cpp  dbow3/FeatureVector.h  dbow3/QueryResults.cpp  dbow3/QueryResults.h  dbow3/quicklz.c  dbow3/quicklz.h  dbow3/ScoringObject.cpp  dbow3/ScoringObject.h  dbow3/timers.h  dbow3/Vocabulary.cpp  dbow3/Vocabulary.h)

# ADD_EXECUTABLE(comparison_tests comparison_tests.cpp dbow2/ScoringObject.cpp dbow2/BowVector.h  dbow2/FClass.h  dbow2/FeatureVector.h   dbow2/FORB.h  dbow2/ScoringObject.h  dbow2/TemplatedVocabulary.h  dbow3/BowVector.h  dbow3/BowVector.cpp  dbow3/Database.cpp  dbow3/Database.h  dbow3/DBoW3.h  dbow3/DescManip.cpp  dbow3/DescManip.h  dbow3/exports.h  dbow3/FeatureVector.cpp  dbow3/FeatureVector.h  dbow3/QueryResults.cpp  dbow3/QueryResults.h  dbow3/quicklz.c  dbow3/quicklz.h  dbow3/ScoringObject.cpp  dbow3/ScoringObject.h  dbow3/timers.h  dbow3/Vocabulary.cpp  dbow3/Vocabulary.h)

# ADD_EXECUTABLE(fbow_test_cpu_x86   test_cpu_x86.cpp   )
#   fbow_test_cpu_x86
INSTALL(TARGETS fbow_test_dbow2VSfbow   RUNTIME DESTINATION bin)

