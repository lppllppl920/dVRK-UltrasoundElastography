cmake_minimum_required(VERSION 2.8.3)
project(my_socket)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

set(CMAKE_CXX_FLAGS -std=c++11)

set(CMAKE_MODULE_PATH /home/xingtong/Downloads/libevent-2.1.8-stable)
find_package(Event)
include_directories(${EVENT_INCLUDE_DIR})

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_PATH})

set(CMAKE_PREFIX_PATH /usr/local/lib/igtl)
find_package(OpenIGTLink)
include_directories(${OpenIGTLink_INCLUDE_DIRS})

set(CMAKE_PREFIX_PATH /usr/local/lib/igtl)
find_package(OpenIGTLinkMUSiiC)
include_directories(${OpenIGTLinkMUSiiC_INCLUDE_DIRS})
link_directories(/usr/local/lib/igtl)

find_package(Boost REQUIRED COMPONENTS thread system)
include_directories(${Boost_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES my_socket
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#find_library (OpenIGTLinkMUSiiC_LIBRARIES_FULL_PATH NAMES ${OpenIGTLinkMUSiiC_LIBRARIES} PATHS /usr/local/lib/igtl)
#add_executable(MySocketClient 
#	src/main_IGTLclient.cpp 
#	src/IGTLclient.cpp )
#target_link_libraries( MySocketClient 
#	pthread
#	${EVENT_LIBRARY}
#	${OpenIGTLinkMUSiiC_LIBRARIES}
#	${OpenIGTLink_LIBRARIES}
#	${Boost_LIBRARIES}
#	${catkin_LIBRARIES}
#	${OpenCV_LIBRARIES}
#)

add_executable( MySocketServer
	src/image_server.cpp )
		
target_link_libraries( MySocketServer 
	${OpenIGTLinkMUSiiC_LIBRARIES}
	${OpenIGTLink_LIBRARIES}
	${Boost_LIBRARIES}
	${CUDA_LIBRARIES}
	${OpenCV_LIBRARIES}
	pthread
)

cuda_add_executable( MySocketClient
	src/main_IGTLclient.cpp 
	src/IGTLclient.cpp 
	src/Elastography.cpp
	src/elastography/matrix_io.cu
	src/elastography/ncc.cu
	src/elastography/scan_conversion.cu
	
	src/IGTLclient.h
    src/Elastography.h
	src/elastography/concurrent_queue.h
	src/elastography/ncc.h
	src/elastography/scan_conversion.h
	src/elastography/matrix_io.h
)
	
	
target_link_libraries( MySocketClient 
	${CUDA_LIBRARIES}
	${EVENT_LIBRARY}
	${OpenIGTLinkMUSiiC_LIBRARIES}
	${OpenIGTLink_LIBRARIES}
	${Boost_LIBRARIES}
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	pthread
	X11
)
