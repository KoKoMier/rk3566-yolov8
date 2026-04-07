#include "v4l2_capture.hpp"

#include <linux/videodev2.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

static int xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

bool V4l2Nv12Capture::open_device(const std::string& device, int req_w, int req_h) {
    cleanup();
    fd_ = open(device.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
        std::cerr << "打开失败 " << device << "：" << std::strerror(errno) << "\n";
        return false;
    }

    v4l2_capability cap {};
    if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) != 0) {
        std::cerr << "VIDIOC_QUERYCAP 失败：" << std::strerror(errno) << "\n";
        cleanup();
        return false;
    }

    is_mplane_ = (cap.device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) != 0;
    const bool can_stream = (cap.device_caps & V4L2_CAP_STREAMING) != 0;
    if (!can_stream) {
        std::cerr << "设备不支持 Streaming\n";
        cleanup();
        return false;
    }

    v4l2_format fmt {};
    fmt.type = is_mplane_ ? V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE : V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (req_w > 0 && req_h > 0) {
        if (is_mplane_) {
            fmt.fmt.pix_mp.width = static_cast<__u32>(req_w);
            fmt.fmt.pix_mp.height = static_cast<__u32>(req_h);
            fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
            fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
        } else {
            fmt.fmt.pix.width = static_cast<__u32>(req_w);
            fmt.fmt.pix.height = static_cast<__u32>(req_h);
            fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
            fmt.fmt.pix.field = V4L2_FIELD_NONE;
        }
        if (xioctl(fd_, VIDIOC_S_FMT, &fmt) != 0) {
            std::cerr << "VIDIOC_S_FMT 失败（将按驱动当前格式继续）：" << std::strerror(errno) << "\n";
        }
    }

    std::memset(&fmt, 0, sizeof(fmt));
    fmt.type = is_mplane_ ? V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE : V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_G_FMT, &fmt) != 0) {
        std::cerr << "VIDIOC_G_FMT 失败：" << std::strerror(errno) << "\n";
        cleanup();
        return false;
    }

    if (is_mplane_) {
        width_ = fmt.fmt.pix_mp.width;
        height_ = fmt.fmt.pix_mp.height;
        fourcc_ = fmt.fmt.pix_mp.pixelformat;
        bytesperline_y_ = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;
        if (bytesperline_y_ == 0) bytesperline_y_ = width_;
        /* 连续 NV12 mmap：UV 紧跟 Y 平面，偏移为 bytesperline×height。
         * 勿用 plane_fmt[0].sizeimage：那是缓冲对齐后的大小，常被填成页对齐值，
         * 误作 UV 偏移会导致色度读错、画面发绿。 */
        uv_plane_offset_ = bytesperline_y_ * height_;
    } else {
        width_ = fmt.fmt.pix.width;
        height_ = fmt.fmt.pix.height;
        fourcc_ = fmt.fmt.pix.pixelformat;
        bytesperline_y_ = fmt.fmt.pix.bytesperline;
        if (bytesperline_y_ == 0) bytesperline_y_ = width_;
        uv_plane_offset_ = bytesperline_y_ * height_;
    }

    v4l2_requestbuffers req {};
    req.count = 4;
    req.type = is_mplane_ ? V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE : V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    req_type_ = req.type;

    if (xioctl(fd_, VIDIOC_REQBUFS, &req) != 0) {
        std::cerr << "VIDIOC_REQBUFS 失败：" << std::strerror(errno) << "\n";
        cleanup();
        return false;
    }
    if (req.count < 2) {
        std::cerr << "可用 buffer 太少\n";
        cleanup();
        return false;
    }

    buffers_.resize(req.count);
    for (uint32_t i = 0; i < req.count; i++) {
        v4l2_buffer buf {};
        v4l2_plane planes[VIDEO_MAX_PLANES] {};
        buf.type = req.type;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (is_mplane_) {
            buf.length = VIDEO_MAX_PLANES;
            buf.m.planes = planes;
        }
        if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) != 0) {
            std::cerr << "VIDIOC_QUERYBUF 失败：" << std::strerror(errno) << "\n";
            cleanup();
            return false;
        }

        size_t length = 0;
        size_t offset = 0;
        if (is_mplane_) {
            length = buf.m.planes[0].length;
            offset = buf.m.planes[0].m.mem_offset;
        } else {
            length = buf.length;
            offset = buf.m.offset;
        }

        void* start = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, static_cast<off_t>(offset));
        if (start == MAP_FAILED) {
            std::cerr << "mmap 失败：" << std::strerror(errno) << "\n";
            cleanup();
            return false;
        }
        buffers_[i].start = start;
        buffers_[i].length = length;
    }

    for (uint32_t i = 0; i < req.count; i++) {
        v4l2_buffer buf {};
        v4l2_plane planes[VIDEO_MAX_PLANES] {};
        buf.type = req.type;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (is_mplane_) {
            buf.length = VIDEO_MAX_PLANES;
            buf.m.planes = planes;
        }
        if (xioctl(fd_, VIDIOC_QBUF, &buf) != 0) {
            std::cerr << "VIDIOC_QBUF 失败：" << std::strerror(errno) << "\n";
            cleanup();
            return false;
        }
    }

    char fcc[5] = {0};
    std::memcpy(fcc, &fourcc_, 4);
    std::cerr << "V4L2 实际格式：" << width_ << "x" << height_ << " fourcc=" << fcc
              << (is_mplane_ ? " (mplane)" : "") << " bytesperline=" << bytesperline_y_
              << " uv_offset=" << uv_plane_offset_ << "\n";
    return true;
}

bool V4l2Nv12Capture::start() {
    v4l2_buf_type type = static_cast<v4l2_buf_type>(req_type_);
    if (xioctl(fd_, VIDIOC_STREAMON, &type) != 0) {
        std::cerr << "VIDIOC_STREAMON 失败：" << std::strerror(errno) << "\n";
        return false;
    }
    return true;
}

bool V4l2Nv12Capture::stop() {
    if (fd_ < 0) return true;
    v4l2_buf_type type = static_cast<v4l2_buf_type>(req_type_);
    xioctl(fd_, VIDIOC_STREAMOFF, &type);
    return true;
}

bool V4l2Nv12Capture::dequeue_frame(uint8_t** data, size_t* used, uint32_t* buf_index) {
    v4l2_buffer buf {};
    v4l2_plane planes[VIDEO_MAX_PLANES] {};
    buf.type = static_cast<v4l2_buf_type>(req_type_);
    buf.memory = V4L2_MEMORY_MMAP;
    if (is_mplane_) {
        buf.length = VIDEO_MAX_PLANES;
        buf.m.planes = planes;
    }
    if (xioctl(fd_, VIDIOC_DQBUF, &buf) != 0) {
        return false;
    }
    size_t u = 0;
    if (is_mplane_) {
        u = buf.m.planes[0].bytesused;
    } else {
        u = buf.bytesused;
    }
    *buf_index = buf.index;
    if (buf.index >= buffers_.size() || u == 0 || u > buffers_[buf.index].length) {
        return false;
    }
    *data = reinterpret_cast<uint8_t*>(buffers_[buf.index].start);
    *used = u;
    return true;
}

bool V4l2Nv12Capture::queue_buffer(uint32_t buf_index) {
    v4l2_buffer buf {};
    v4l2_plane planes[VIDEO_MAX_PLANES] {};
    buf.type = static_cast<v4l2_buf_type>(req_type_);
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = buf_index;
    if (is_mplane_) {
        buf.length = VIDEO_MAX_PLANES;
        buf.m.planes = planes;
    }
    return xioctl(fd_, VIDIOC_QBUF, &buf) == 0;
}

void V4l2Nv12Capture::cleanup() {
    stop();
    for (auto& b : buffers_) {
        if (b.start && b.length) {
            munmap(b.start, b.length);
            b.start = nullptr;
            b.length = 0;
        }
    }
    buffers_.clear();
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    width_ = height_ = fourcc_ = req_type_ = 0;
    bytesperline_y_ = uv_plane_offset_ = 0;
    is_mplane_ = false;
}
