#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct MMapBuf {
    void* start = nullptr;
    size_t length = 0;
};

/** V4L2 NV12（含 MPLANE）mmap 采集，供 cam_yolo 使用 */
class V4l2Nv12Capture {
public:
    ~V4l2Nv12Capture() { cleanup(); }

    bool open_device(const std::string& device, int req_w, int req_h);
    void cleanup();

    bool start();
    bool stop();

    /** 取一帧；返回 mmap 指针与 bytesused，成功后必须 queue_buffer */
    bool dequeue_frame(uint8_t** data, size_t* used, uint32_t* buf_index);
    bool queue_buffer(uint32_t buf_index);

    int fd() const { return fd_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint32_t fourcc() const { return fourcc_; }
    bool is_mplane() const { return is_mplane_; }
    /** Y 平面每行字节数（常 ≥ width，对齐后更大）；错误地按 width 取 UV 会偏色（偏绿） */
    uint32_t bytesperline_y() const { return bytesperline_y_; }
    /** UV 平面相对缓冲区起始的字节偏移（单缓冲连续 NV12） */
    uint32_t uv_plane_offset() const { return uv_plane_offset_; }

private:
    int fd_ = -1;
    bool is_mplane_ = false;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t fourcc_ = 0;
    uint32_t bytesperline_y_ = 0;
    uint32_t uv_plane_offset_ = 0;
    uint32_t req_type_ = 0;
    std::vector<MMapBuf> buffers_;
};
