cmake_minimum_required(VERSION 3.15)

# 交叉编译：export GCC_COMPILER=.../bin/aarch64-linux-gnu
if(NOT DEFINED CMAKE_C_COMPILER AND DEFINED ENV{GCC_COMPILER})
    set(CMAKE_C_COMPILER "$ENV{GCC_COMPILER}-gcc")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER AND DEFINED ENV{GCC_COMPILER})
    set(CMAKE_CXX_COMPILER "$ENV{GCC_COMPILER}-g++")
endif()

project(camera_rknn LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- RKNN YOLOv8 + 摄像头（唯一目标）----------
set(RKNN_MODEL_ZOO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/rknn_model_zoo"
    CACHE PATH "rknn_model_zoo 根目录（含 3rdparty、utils）")

option(BUILD_CAM_YOLO "编译 cam_yolo（需 rknn_model_zoo + aarch64 librknnrt）" ON)

# rknn_model_zoo 的 3rdparty 在 64 位下固定选 Linux/aarch64 的 librknnrt，必须在 aarch64 板子上
# 或使用 aarch64-linux-gnu-g++ 交叉链接，否则 x86 主机会报 “file in wrong format”。
set(_cam_yolo_ok OFF)
if(CMAKE_C_COMPILER MATCHES "aarch64-linux-gnu" OR CMAKE_CXX_COMPILER MATCHES "aarch64-linux-gnu")
    set(_cam_yolo_ok ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(_cam_yolo_ok ON)
endif()

if(BUILD_CAM_YOLO AND NOT _cam_yolo_ok)
    message(WARNING "当前不是 aarch64 工具链/主机，跳过 cam_yolo。交叉编译示例: export GCC_COMPILER=.../aarch64-linux-gnu && cmake ..")
endif()

if(BUILD_CAM_YOLO AND _cam_yolo_ok)
    if(NOT EXISTS "${RKNN_MODEL_ZOO_ROOT}/utils/CMakeLists.txt")
        message(WARNING "未找到 rknn_model_zoo，跳过 cam_yolo。请设置 -DRKNN_MODEL_ZOO_ROOT=...")
    else()
        # 与官方例程一致：RK3566 使用 rk356x
        set(TARGET_SOC "rk356x" CACHE STRING "Rockchip SoC" FORCE)

        add_subdirectory("${RKNN_MODEL_ZOO_ROOT}/3rdparty" rknn_3rdparty.out)
        add_subdirectory("${RKNN_MODEL_ZOO_ROOT}/utils" rknn_utils.out)

        find_package(Threads REQUIRED)

        set(_yolov8_cpp "${RKNN_MODEL_ZOO_ROOT}/examples/yolov8/cpp")
        add_executable(cam_yolo
            cam_yolo.cpp
            v4l2_capture.cpp
            ${_yolov8_cpp}/postprocess.cc
            ${_yolov8_cpp}/rknpu2/yolov8.cc
        )

        # 头文件与官方 yolov8 demo 一致；标签路径见 postprocess.cc 默认 ../model/...
        target_include_directories(cam_yolo PRIVATE
            ${_yolov8_cpp}
            ${LIBRKNNRT_INCLUDES}
            ${CMAKE_SOURCE_DIR}
        )

        target_link_libraries(cam_yolo PRIVATE
            imageutils
            fileutils
            imagedrawing
            ${LIBRKNNRT}
            dl
            Threads::Threads
        )

        set_target_properties(cam_yolo PROPERTIES
            INSTALL_RPATH "$ORIGIN/lib:$ORIGIN/../lib"
        )

        # 部署到板子时，可将 SDK 里的 librknnrt.so 拷到可执行文件旁 lib/ 或设置 LD_LIBRARY_PATH
        # 例: cp rknn_model_zoo/3rdparty/rknpu2/Linux/aarch64/librknnrt.so ./lib/
    endif()
endif()