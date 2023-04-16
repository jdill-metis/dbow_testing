 INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src)
LINK_LIBRARIES(${PROJECT_NAME})


ADD_EXECUTABLE(test_iobinary test_iobinary.cpp  )
ADD_EXECUTABLE(test_bigvoc test_bigvoc.cpp  )
ADD_EXECUTABLE(test_flann test_flann.cpp  )
ADD_EXECUTABLE(test_fbow test_fbow.cpp  )
ADD_EXECUTABLE(time_metrics time_metrics.cpp)

INSTALL(TARGETS  time_metrics  RUNTIME DESTINATION bin)