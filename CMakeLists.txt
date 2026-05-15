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
set(RKNN_MODEL_ZOO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/rknn_3566_model/rknn_model_zoo"
    CACHE PATH "rknn_model_zoo 根目录（含 3rdparty、utils）")

option(BUILD_CAM_YOLO "编译 cam_yolo（需 rknn_model_zoo + aarch64 librknnrt）" ON)

# rknn_model_zoo 的 3rdparty 在 64 位下固定选 Linux/aarch64 的 librknnrt，必须在 aarch64 板子上
# 或使用 aarch64-linux-gnu-g++ 交叉链接，否则 x86 主机会报 “file in wrong format”。
set(_cam_yolo_ok OFF)
# aarch64-linux-gnu、Yocto aarch64-poky-linux 等
if(CMAKE_C_COMPILER MATCHES "aarch64.*-linux" OR CMAKE_CXX_COMPILER MATCHES "aarch64.*-linux")
    set(_cam_yolo_ok ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(_cam_yolo_ok ON)
endif()

if(BUILD_CAM_YOLO AND NOT _cam_yolo_ok)
    message(WARNING "当前不是 aarch64 工具链/主机，跳过 cam_yolo。交叉编译示例: export GCC_COMPILER=.../aarch64-linux-gnu && cmake ..")
endif()

if(BUILD_CAM_YOLO AND _cam_yolo_ok)
    if(NOT EXISTS "${RKNN_MODEL_ZOO_ROOT}/utils/CMakeLists.txt")
        set(RKNN_MODEL_ZOO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/rknn_model_zoo" CACHE PATH "Path to rknn_model_zoo" FORCE)
    endif()
    if(NOT EXISTS "${RKNN_MODEL_ZOO_ROOT}/utils/CMakeLists.txt")
        message(WARNING "未找到 rknn_model_zoo，跳过 cam_yolo。请设置 -DRKNN_MODEL_ZOO_ROOT=...")
    else()
        # 与官方例程一致：RK3566 使用 rk356x
        set(TARGET_SOC "rk356x" CACHE STRING "Rockchip SoC" FORCE)

        add_subdirectory("${RKNN_MODEL_ZOO_ROOT}/3rdparty" rknn_3rdparty.out)
        add_subdirectory("${RKNN_MODEL_ZOO_ROOT}/utils" rknn_utils.out)

        find_package(Threads REQUIRED)

        set(_yolov8_cpp "${RKNN_MODEL_ZOO_ROOT}/examples/yolov8/cpp")
        set(_cam_yolo_sources
            cam_yolo.cpp
            v4l2_capture.cpp
            ${_yolov8_cpp}/postprocess.cc
            ${_yolov8_cpp}/rknpu2/yolov8.cc
        )

        add_library(cam_yolo_lib SHARED ${_cam_yolo_sources})
        set_target_properties(cam_yolo_lib PROPERTIES
            OUTPUT_NAME cam_yolo
            INSTALL_RPATH "$ORIGIN/lib:$ORIGIN/../lib"
        )

        add_executable(cam_yolo cam_yolo_exe.cpp)
        # 可执行文件最终链接阶段需解析 libcam_yolo.so -> librknnrt.so；
        # Poky 等 sysroot 中通常没有 librknnrt，须指明 rknn 预编译库目录（-rpath-link + 直接链接）。
        get_filename_component(_LIBRKNNRT_DIR "${LIBRKNNRT}" DIRECTORY)
        target_link_libraries(cam_yolo PRIVATE cam_yolo_lib ${LIBRKNNRT})
        if(EXISTS "${LIBRKNNRT}")
            target_link_options(cam_yolo_lib PRIVATE "LINKER:-rpath-link=${_LIBRKNNRT_DIR}")
            target_link_options(cam_yolo PRIVATE "LINKER:-rpath-link=${_LIBRKNNRT_DIR}")
        endif()

        # 头文件与官方 yolov8 demo 一致；标签路径见 postprocess.cc 默认 ../model/...
        foreach(_t cam_yolo_lib cam_yolo)
            target_include_directories(${_t} PRIVATE
                ${_yolov8_cpp}
                ${LIBRKNNRT_INCLUDES}
                ${RKNN_MODEL_ZOO_ROOT}/utils
                ${CMAKE_SOURCE_DIR}
            )
        endforeach()

        target_link_libraries(cam_yolo_lib PRIVATE
            imageutils
            fileutils
            imagedrawing
            ${LIBRKNNRT}
            dl
            Threads::Threads
        )

        # 部署到板子时，可将 SDK 里的 librknnrt.so 拷到可执行文件旁 lib/ 或设置 LD_LIBRARY_PATH
        # 例: cp rknn_model_zoo/3rdparty/rknpu2/Linux/aarch64/librknnrt.so ./lib/
    endif()
endif()