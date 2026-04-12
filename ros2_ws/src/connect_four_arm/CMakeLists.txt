cmake_minimum_required(VERSION 3.8)
project(connect_four_arm)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(PROGRAMS scripts/column_mover.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
