cmake_minimum_required(VERSION 3.8)
project(connect_four_arm)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

install(DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)

install(PROGRAMS
  scripts/column_mover.py
  scripts/piece_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
