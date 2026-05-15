// 摄像头 NV12 -> RKNN YOLOv8 推理 -> NV12 上画框；可选 framebuffer 预览（无 OpenCV）

#include "v4l2_capture.hpp"

#include <linux/fb.h>
#include <linux/videodev2.h>

#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "yolov8.h"
#include "postprocess.h"

#if defined(_WIN32)
#define CAM_YOLO_API __declspec(dllexport)
#else
#define CAM_YOLO_API __attribute__((visibility("default")))
#endif

extern "C" {
#include "image_drawing.h"
#include "image_utils.h"
}

struct Framebuffer {
    int fd = -1;
    uint8_t* mem = nullptr;
    size_t mem_size = 0;
    int xres = 0;
    int yres = 0;
    int bpp = 0;
    int line_length = 0;
};

static int xioctl_fb(int fd, unsigned long req, void* arg) {
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

static bool fb_open(Framebuffer& fb, const std::string& path) {
    fb.fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fb.fd < 0) {
        std::cerr << "打开 framebuffer 失败 " << path << "\n";
        return false;
    }
    fb_fix_screeninfo finfo {};
    fb_var_screeninfo vinfo {};
    if (xioctl_fb(fb.fd, FBIOGET_FSCREENINFO, &finfo) != 0 ||
        xioctl_fb(fb.fd, FBIOGET_VSCREENINFO, &vinfo) != 0) {
        close(fb.fd);
        fb.fd = -1;
        return false;
    }
    fb.xres = static_cast<int>(vinfo.xres);
    fb.yres = static_cast<int>(vinfo.yres);
    fb.bpp = static_cast<int>(vinfo.bits_per_pixel);
    fb.line_length = static_cast<int>(finfo.line_length);
    fb.mem_size = static_cast<size_t>(finfo.smem_len);
    fb.mem = static_cast<uint8_t*>(
        mmap(nullptr, fb.mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, fb.fd, 0));
    if (fb.mem == MAP_FAILED) {
        fb.mem = nullptr;
        close(fb.fd);
        fb.fd = -1;
        return false;
    }
    std::cerr << "Framebuffer: " << fb.xres << "x" << fb.yres << " bpp=" << fb.bpp << "\n";
    return true;
}

static void fb_close(Framebuffer& fb) {
    if (fb.mem) {
        munmap(fb.mem, fb.mem_size);
        fb.mem = nullptr;
    }
    if (fb.fd >= 0) {
        close(fb.fd);
        fb.fd = -1;
    }
}

static inline uint8_t clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<uint8_t>(v);
}

/** BT.601 限幅（TV range） */
static inline void yuv_to_rgb_limited(uint8_t y, uint8_t u, uint8_t v, uint8_t& r, uint8_t& g, uint8_t& b) {
    int C = static_cast<int>(y) - 16;
    int D = static_cast<int>(u) - 128;
    int E = static_cast<int>(v) - 128;
    int rr = (298 * C + 409 * E + 128) >> 8;
    int gg = (298 * C - 100 * D - 208 * E + 128) >> 8;
    int bb = (298 * C + 516 * D + 128) >> 8;
    r = clamp_u8(rr);
    g = clamp_u8(gg);
    b = clamp_u8(bb);
}

/** BT.709 限幅（常见 720p/1080p 链路；与 601 混用会偏色） */
static inline void yuv_to_rgb_limited_bt709(uint8_t y, uint8_t u, uint8_t v, uint8_t& r, uint8_t& g, uint8_t& b) {
    int C = static_cast<int>(y) - 16;
    int D = static_cast<int>(u) - 128;
    int E = static_cast<int>(v) - 128;
    int rr = (298 * C + 459 * E + 128) >> 8;
    int gg = (298 * C - 55 * D - 137 * E + 128) >> 8;
    int bb = (298 * C + 541 * D + 128) >> 8;
    r = clamp_u8(rr);
    g = clamp_u8(gg);
    b = clamp_u8(bb);
}

/** 近似 JPEG full range，部分传感器/链路偏绿时可试 */
static inline void yuv_to_rgb_full(uint8_t y, uint8_t u, uint8_t v, uint8_t& r, uint8_t& g, uint8_t& b) {
    int U = static_cast<int>(u) - 128;
    int V = static_cast<int>(v) - 128;
    int Y = static_cast<int>(y);
    int rr = Y + ((1436 * V + 512) >> 10);
    int gg = Y - ((352 * U + 731 * V + 512) >> 10);
    int bb = Y + ((1814 * U + 512) >> 10);
    r = clamp_u8(rr);
    g = clamp_u8(gg);
    b = clamp_u8(bb);
}

/** 从带 stride 的 V4L2 NV12 拷成紧密排列（w*h*3/2），供推理/画框/ffmpeg */
static void nv12_copy_tight(const uint8_t* src, int w, int h, uint32_t y_stride, uint32_t uv_off, uint8_t* dst) {
    for (int y = 0; y < h; y++) {
        std::memcpy(dst + static_cast<size_t>(y) * w, src + static_cast<size_t>(y) * y_stride, static_cast<size_t>(w));
    }
    const uint8_t* src_uv = src + uv_off;
    uint8_t* dst_uv = dst + static_cast<size_t>(w) * h;
    for (int y = 0; y < h / 2; y++) {
        std::memcpy(dst_uv + static_cast<size_t>(y) * w, src_uv + static_cast<size_t>(y) * y_stride, static_cast<size_t>(w));
    }
}

/** 不缩放：左上角对齐，超出部分裁剪；y_stride/uv_off 与 v4l2 bytesperline 一致 */
static void blit_nv12_to_fb(const uint8_t* nv12, int src_w, int src_h, uint32_t y_stride, uint32_t uv_off, Framebuffer& fb,
                            bool full_range_rgb, bool chroma_vu, bool bt709) {
    const uint8_t* y_plane = nv12;
    const uint8_t* uv_plane = nv12 + uv_off;
    const int copy_w = (src_w < fb.xres) ? src_w : fb.xres;
    const int copy_h = (src_h < fb.yres) ? src_h : fb.yres;
    for (int dy = 0; dy < copy_h; dy++) {
        const int sy = dy;
        uint8_t* dst_line = fb.mem + static_cast<size_t>(dy) * static_cast<size_t>(fb.line_length);
        for (int dx = 0; dx < copy_w; dx++) {
            const int sx = dx;
            uint8_t Y = y_plane[static_cast<size_t>(sy) * y_stride + sx];
            size_t uv_index = (static_cast<size_t>(sy / 2) * y_stride) + static_cast<size_t>(sx & ~1);
            uint8_t U = uv_plane[uv_index + 0];
            uint8_t V = uv_plane[uv_index + 1];
            if (chroma_vu) {
                uint8_t t = U;
                U = V;
                V = t;
            }
            uint8_t r, g, b;
            if (full_range_rgb) {
                yuv_to_rgb_full(Y, U, V, r, g, b);
            } else if (bt709) {
                yuv_to_rgb_limited_bt709(Y, U, V, r, g, b);
            } else {
                yuv_to_rgb_limited(Y, U, V, r, g, b);
            }
            if (fb.bpp == 16) {
                uint16_t rgb565 = static_cast<uint16_t>(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
                reinterpret_cast<uint16_t*>(dst_line)[dx] = rgb565;
            } else if (fb.bpp == 32) {
                uint32_t xrgb = (0xFFu << 24) | (static_cast<uint32_t>(r) << 16) |
                                (static_cast<uint32_t>(g) << 8) | static_cast<uint32_t>(b);
                reinterpret_cast<uint32_t*>(dst_line)[dx] = xrgb;
            }
        }
    }
}

/** 保存 NV12 裸数据为 .yuv（内容与常见 NV12/YUV420SP 流一致，供 ffmpeg 等使用） */
static bool save_nv12_as_yuv(const char* path_yuv, image_buffer_t* nv12_img) {
    FILE* fp = std::fopen(path_yuv, "wb");
    if (!fp) return false;
    int w = nv12_img->width;
    int h = nv12_img->height;
    size_t n = static_cast<size_t>(w) * static_cast<size_t>(h) * 3 / 2;
    size_t nw = std::fwrite(nv12_img->virt_addr, 1, n, fp);
    std::fclose(fp);
    return nw == n;
}

/** NV12（已画框）转 JPG；失败则退化为同序号 .yuv 裸流 */
static bool save_frame_image(const char* path_jpg, const char* path_yuv_fallback, image_buffer_t* nv12_img,
                             bool yuv_only) {
    if (yuv_only) {
        return save_nv12_as_yuv(path_yuv_fallback, nv12_img);
    }
    int w = nv12_img->width;
    int h = nv12_img->height;
    image_buffer_t rgb {};
    rgb.width = w;
    rgb.height = h;
    rgb.width_stride = w;
    rgb.height_stride = h;
    rgb.format = IMAGE_FORMAT_RGB888;
    rgb.size = w * h * 3;
    rgb.virt_addr = static_cast<unsigned char*>(std::malloc(static_cast<size_t>(rgb.size)));
    rgb.fd = -1;
    if (!rgb.virt_addr) return false;

    int ret = convert_image(nv12_img, &rgb, nullptr, nullptr, 0);
    if (ret == 0) {
        ret = write_image(path_jpg, &rgb);
    }
    std::free(rgb.virt_addr);
    if (ret == 0) return true;

    if (save_nv12_as_yuv(path_yuv_fallback, nv12_img)) {
        std::cerr << "保存 JPG 失败，已写入原始 NV12 流: " << path_yuv_fallback << "\n";
        return true;
    }
    return false;
}

/** --input 为 V4L2 节点（如 /dev/video0）时与 --device 等价，走 NV12 mmap 采集流。 */
static bool input_path_is_v4l2_device(const std::string& p) {
    if (p.size() >= 10 && p.compare(0, 10, "/dev/video") == 0) {
        return true;
    }
    struct stat st {};
    if (stat(p.c_str(), &st) != 0 || !S_ISCHR(st.st_mode)) {
        return false;
    }
    return major(st.st_rdev) == 81u;
}

/** 输出路径是否为「原始 NV12 码流文件」（逐帧紧密拼接，非 MP4 容器）。 */
static bool path_is_nv12_stream_file(const std::string& p) {
    auto ends_icase = [](const std::string& s, const char* suf) {
        const size_t n = std::strlen(suf);
        if (s.size() < n) {
            return false;
        }
        for (size_t i = 0; i < n; i++) {
            if (std::tolower(static_cast<unsigned char>(s[s.size() - n + i])) !=
                std::tolower(static_cast<unsigned char>(suf[i]))) {
                return false;
            }
        }
        return true;
    };
    return ends_icase(p, ".yuv") || ends_icase(p, ".nv12");
}

/**
 * 如 ./out.yuvclear：文件名里带有 ".yuv" 但整条路径不以 .yuv/.nv12 结尾，用户常误以为可写 NV12 码流，
 * 实际会落入「当作输出目录」分支。
 */
static bool path_looks_like_nonstandard_yuv_name(const std::string& p) {
    if (path_is_nv12_stream_file(p)) {
        return false;
    }
    std::string lower = p;
    for (char& c : lower) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    const size_t dot_yuv = lower.rfind(".yuv");
    if (dot_yuv == std::string::npos) {
        return false;
    }
    /* 以 .yuv 结尾但 path_is_nv12_stream_file 为假：理论上不应发生 */
    if (dot_yuv + 4 == lower.size()) {
        return false;
    }
    /* 存在形如 xxx.yuvxxx 的中间段且后面还有字符 */
    return true;
}

/** 流模式下，由 --save-dir 或 --output 推导保存带框 JPEG 的目录（空表示不保存）。 */
static std::string stream_nv12_save_directory(const std::string& save_dir, const std::string& output_path) {
    if (!save_dir.empty()) {
        return save_dir;
    }
    if (output_path == "out.jpg") {
        return {};
    }
    if (path_is_nv12_stream_file(output_path)) {
        return {};
    }
    auto ends_icase = [](const std::string& s, const char* suf) {
        const size_t n = std::strlen(suf);
        if (s.size() < n) {
            return false;
        }
        for (size_t i = 0; i < n; i++) {
            if (std::tolower(static_cast<unsigned char>(s[s.size() - n + i])) !=
                std::tolower(static_cast<unsigned char>(suf[i]))) {
                return false;
            }
        }
        return true;
    };
    if (ends_icase(output_path, ".jpg") || ends_icase(output_path, ".jpeg") || ends_icase(output_path, ".png")) {
        const size_t dot = output_path.find_last_of('.');
        const std::string base =
            (dot == std::string::npos) ? output_path : output_path.substr(0, dot);
        return base + "_frames";
    }
    if (path_looks_like_nonstandard_yuv_name(output_path)) {
        std::cerr << "警告: --output 「" << output_path
                  << "」不以 .yuv 或 .nv12 结尾，不会当作「单文件 NV12 码流」写出；\n"
                  << "      当前将其视为「目录」仅用于按间隔保存 frame_*.jpg（或 .yuv）。\n"
                  << "      若要一整条带检测框的 NV12 流，请使用例如: --output ./out.yuv\n";
    }
    return output_path;
}

static bool mkdir_p(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    std::string cur;
    cur.reserve(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
        cur.push_back(path[i]);
        if (path[i] == '/' && cur.size() > 1) {
            if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
    }
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

/**
 * 从已采集的 NV12 字节流（紧密排列：Y 宽×高 + UV 交错）逐帧处理。
 * stream_path 为 "-" 时读 stdin。不包含 V4L2 打开，仅 fread。
 */
static int run_nv12_stream_input(const std::string& stream_path, int stream_w, int stream_h, int max_frames,
                                 rknn_app_context_t* rknn_ctx, bool display, const std::string& fb_path, bool timing,
                                 const std::string& save_dir, int save_interval, bool save_yuv_only,
                                 const std::string& yuv_stream_path, int yuv_stream_interval, bool yuv_full_range,
                                 bool chroma_vu, bool bt709) {
    FILE* in_fp = nullptr;
    bool close_in = false;
    if (stream_path == "-") {
        in_fp = stdin;
    } else {
        in_fp = std::fopen(stream_path.c_str(), "rb");
        if (!in_fp) {
            std::cerr << "无法打开 NV12 流: " << stream_path << " (" << std::strerror(errno) << ")\n";
            return 1;
        }
        close_in = true;
    }

    const uint32_t cap_w = static_cast<uint32_t>(stream_w);
    const uint32_t cap_h = static_cast<uint32_t>(stream_h);
    if (cap_w < 2 || cap_h < 2) {
        std::cerr << "NV12 流需要有效的 --width / --height\n";
        if (close_in) std::fclose(in_fp);
        return 1;
    }

    if (!save_dir.empty() && !mkdir_p(save_dir)) {
        std::cerr << "无法创建输出目录: " << save_dir << " (" << std::strerror(errno) << ")\n";
        if (close_in) std::fclose(in_fp);
        return 1;
    }

    const size_t frame_bytes = static_cast<size_t>(cap_w) * cap_h * 3 / 2;
    std::vector<uint8_t> frame_buf(frame_bytes);

    Framebuffer fb;
    if (display) {
        if (!fb_open(fb, fb_path) || (fb.bpp != 16 && fb.bpp != 32)) {
            std::cerr << "framebuffer 不可用或 bpp 非 16/32，关闭显示\n";
            if (fb.fd >= 0) fb_close(fb);
            display = false;
        }
    }

    FILE* yuv_stream_fp = nullptr;
    if (!yuv_stream_path.empty()) {
        yuv_stream_fp = std::fopen(yuv_stream_path.c_str(), "wb");
        if (!yuv_stream_fp) {
            std::cerr << "无法打开 YUV 流文件 " << yuv_stream_path << "：" << std::strerror(errno) << "\n";
            if (close_in) std::fclose(in_fp);
            if (display) fb_close(fb);
            return 1;
        }
        std::cerr << "NV12 视频流写入: " << yuv_stream_path << " （" << cap_w << "x" << cap_h << ", 每 "
                  << yuv_stream_interval << " 帧追加一次）\n";
    }

    const uint32_t bpl = cap_w;
    const uint32_t uv_off = cap_w * cap_h;

    if (display) {
        std::cerr << "预览色彩: "
                  << (yuv_full_range ? "full" : (bt709 ? "BT.709" : "BT.601")) << " 限幅"
                  << (chroma_vu ? " + NV21(VU) 色度" : " + NV12(UV) 色度") << "\n";
    }

    int processed = 0;
    double total_ms = 0;
    while (max_frames == 0 || processed < max_frames) {
        auto t_loop0 = std::chrono::steady_clock::now();

        const size_t got = std::fread(frame_buf.data(), 1, frame_bytes, in_fp);
        if (got != frame_bytes) {
            if (got > 0) {
                std::cerr << "NV12 流末尾非整帧（缺 " << (frame_bytes - got) << " 字节），已停止\n";
            }
            break;
        }

        image_buffer_t img {};
        img.width = static_cast<int>(cap_w);
        img.height = static_cast<int>(cap_h);
        img.height_stride = img.height;
        img.format = IMAGE_FORMAT_YUV420SP_NV12;
        img.size = static_cast<int>(frame_bytes);
        img.fd = -1;
        img.virt_addr = frame_buf.data();
        img.width_stride = static_cast<int>(cap_w);

        object_detect_result_list od_results {};
        if (inference_yolov8_model(rknn_ctx, &img, &od_results) == 0) {
            char text[256];
            for (int i = 0; i < od_results.count; i++) {
                object_detect_result* det = &od_results.results[i];
                int x1 = det->box.left;
                int y1 = det->box.top;
                int x2 = det->box.right;
                int y2 = det->box.bottom;
                draw_rectangle(&img, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 2);
                std::snprintf(text, sizeof(text), "%s %.0f%%", coco_cls_to_name(det->cls_id), det->prop * 100.f);
                draw_text(&img, text, x1, y1 - 12, COLOR_RED, 10);
            }
        }

        if (display) {
            blit_nv12_to_fb(frame_buf.data(), img.width, img.height, bpl, uv_off, fb, yuv_full_range, chroma_vu,
                            bt709);
        }

        if (!save_dir.empty() && (processed % save_interval) == 0) {
            char path_jpg[512];
            char path_yuv[512];
            std::snprintf(path_jpg, sizeof(path_jpg), "%s/frame_%06d.jpg", save_dir.c_str(), processed);
            std::snprintf(path_yuv, sizeof(path_yuv), "%s/frame_%06d.yuv", save_dir.c_str(), processed);
            save_frame_image(path_jpg, path_yuv, &img, save_yuv_only);
        }

        if (yuv_stream_fp && (processed % yuv_stream_interval) == 0) {
            size_t nw = std::fwrite(img.virt_addr, 1, frame_bytes, yuv_stream_fp);
            if (nw != frame_bytes) {
                std::cerr << "写入 YUV 流失败\n";
            }
        }

        processed++;

        auto t_loop1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_loop1 - t_loop0).count();
        total_ms += ms;
        if (timing) {
            std::cerr << "frame " << (processed - 1) << "  loop_time=" << ms << " ms\n";
        } else if ((processed % 30) == 0) {
            double avg = total_ms / static_cast<double>(processed);
            std::cerr << "已处理 " << processed << " 帧, 平均循环 " << avg << " ms/帧\r" << std::flush;
        }
    }

    std::cerr << "\n";
    if (yuv_stream_fp) {
        std::fclose(yuv_stream_fp);
        if (!yuv_stream_path.empty()) {
            std::cout << "YUV 流已关闭: " << yuv_stream_path << "\n";
        }
    }
    if (close_in) {
        std::fclose(in_fp);
    }
    if (display) {
        fb_close(fb);
    }

    if (processed > 0) {
        std::cout << "结束，共 " << processed << " 帧，平均循环时间 "
                  << (total_ms / static_cast<double>(processed)) << " ms/帧\n";
    } else {
        std::cout << "结束，共 0 帧（请确认 NV12 尺寸与 --width/--height 一致）\n";
    }
    return 0;
}

struct CamYoloHandle {
    rknn_app_context_t rknn_ctx;
};

extern "C" CAM_YOLO_API void* cam_yolo_create(const char* model_path, const char* labels_path) {
    if (!model_path) {
        return nullptr;
    }
    if (init_post_process(labels_path) != 0) {
        std::cerr << "cam_yolo_create: init_post_process 失败\n";
        return nullptr;
    }
    auto* h = new CamYoloHandle();
    std::memset(&h->rknn_ctx, 0, sizeof(h->rknn_ctx));
    if (init_yolov8_model(model_path, &h->rknn_ctx) != 0) {
        std::cerr << "cam_yolo_create: init_yolov8_model 失败\n";
        deinit_post_process();
        delete h;
        return nullptr;
    }
    return h;
}

extern "C" CAM_YOLO_API void cam_yolo_destroy(void* handle) {
    if (!handle) {
        return;
    }
    auto* h = static_cast<CamYoloHandle*>(handle);
    release_yolov8_model(&h->rknn_ctx);
    deinit_post_process();
    delete h;
}

extern "C" CAM_YOLO_API int cam_yolo_infer_nv12(void* handle, uint8_t* nv12_data, int width, int height) {
    if (!handle || !nv12_data || width < 2 || height < 2) {
        return -1;
    }
    auto* h = static_cast<CamYoloHandle*>(handle);
    const uint32_t cap_w = static_cast<uint32_t>(width);
    const uint32_t cap_h = static_cast<uint32_t>(height);
    const size_t frame_bytes = static_cast<size_t>(cap_w) * cap_h * 3 / 2;

    image_buffer_t img {};
    img.width = static_cast<int>(cap_w);
    img.height = static_cast<int>(cap_h);
    img.height_stride = img.height;
    img.format = IMAGE_FORMAT_YUV420SP_NV12;
    img.size = static_cast<int>(frame_bytes);
    img.fd = -1;
    img.virt_addr = nv12_data;
    img.width_stride = static_cast<int>(cap_w);

    object_detect_result_list od_results {};
    if (inference_yolov8_model(&h->rknn_ctx, &img, &od_results) != 0) {
        return -1;
    }
    char text[256];
    for (int i = 0; i < od_results.count; i++) {
        object_detect_result* det = &od_results.results[i];
        int x1 = det->box.left;
        int y1 = det->box.top;
        int x2 = det->box.right;
        int y2 = det->box.bottom;
        draw_rectangle(&img, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 2);
        std::snprintf(text, sizeof(text), "%s %.0f%%", coco_cls_to_name(det->cls_id), det->prop * 100.f);
        draw_text(&img, text, x1, y1 - 12, COLOR_RED, 10);
    }
    return 0;
}

extern "C" CAM_YOLO_API int cam_yolo_run_buffer(const char* model_path, uint8_t* data, size_t size, int width, int height, const char* out_path, int append) {
    if (!model_path || !data || width < 2 || height < 2) {
        return -1;
    }
    void* handle = cam_yolo_create(model_path, nullptr);
    if (!handle) {
        return -1;
    }

    FILE* fp = nullptr;
    if (out_path) {
        fp = std::fopen(out_path, append ? "ab" : "wb");
        if (!fp) {
            cam_yolo_destroy(handle);
            return -1;
        }
    }

    const size_t frame_bytes = static_cast<size_t>(width) * height * 3 / 2;
    const size_t n_frames = size / frame_bytes;
    int ret = 0;

    for (size_t i = 0; i < n_frames; i++) {
        uint8_t* frame_ptr = data + i * frame_bytes;
        if (cam_yolo_infer_nv12(handle, frame_ptr, width, height) != 0) {
            ret = -1;
            break;
        }
        if (fp) {
            if (std::fwrite(frame_ptr, 1, frame_bytes, fp) != frame_bytes) {
                ret = -1;
                break;
            }
        }
    }

    if (fp) std::fclose(fp);
    cam_yolo_destroy(handle);
    return ret;
}

static void usage(const char* p) {
    std::cerr << "用法: " << p << " [选项]   或   " << p << " --model <yolov8.rknn> [选项]\n"
              << "  --input <path>         图片 jpg/png；摄像头 /dev/video*；外部 NV12 流见 --stream-nv12\n"
              << "  --stream-nv12          与 --input 合用：从文件/管道/stdin(-) 读紧密 NV12，帧大小=宽×高×3/2\n"
              << "  --output <path>     图片：单张 jpg。流：须以 .yuv 或 .nv12 结尾才写「单文件带框 NV12 流」；\n"
              << "                      其它路径名（如 out.yuvclear）会被当成「目录」只保存周期性抽帧图\n"
              << "  --labels <path.txt>    指定标签文件（默认 ./model/coco_80_labels_list.txt）\n"
              << "  --model <yolov8.rknn>   指定模型\n"
              << "  --device /dev/video0   摄像头节点（若未用 --input 指定视频设备则用此项）\n"
              << "  --width 640 --height 480   采集分辨率（默认 640x480）\n"
              << "  --frames N   摄像头流：处理 N 帧后退出；0 表示一直采集\n"
              << "  --display    可选：/dev/fb0 预览\n"
              << "  --fb /dev/fb0\n"
              << "  --timing     每帧打印本循环耗时（ms）\n"
              << "  --save-dir <目录>   摄像头/NV12 流：周期性保存带框画面\n"
              << "  --save-yuv-stream <文件>  带框 NV12 连续追加（原始视频流；--output *.yuv 可代替）\n"
              << "  --yuv-stream-interval N   写 NV12 流时隔 N 帧一次（默认 1）\n";
}

extern "C" CAM_YOLO_API int cam_yolo_main(int argc, char** argv) {
    std::string model_path;
    std::string input_path;
    std::string output_path = "out.jpg";
    std::string labels_path;
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    int max_frames = 0;
    bool display = false;
    std::string fb_path = "/dev/fb0";
    bool timing = false;
    std::string save_dir;
    int save_interval = 30;
    bool save_yuv_only = false;
    std::string yuv_stream_path;
    int yuv_stream_interval = 1;
    bool yuv_full_range = false;
    bool chroma_vu = false;
    bool bt709 = true;
    bool input_nv12_stream = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (a == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (a == "--labels" && i + 1 < argc) {
            labels_path = argv[++i];
        } else if (a == "--timing") {
            timing = true;
        } else if (a == "--save-yuv-only") {
            save_yuv_only = true;
        } else if (a == "--save-dir" && i + 1 < argc) {
            save_dir = argv[++i];
        } else if (a == "--save-interval" && i + 1 < argc) {
            save_interval = std::stoi(argv[++i]);
        } else if (a == "--save-yuv-stream" && i + 1 < argc) {
            yuv_stream_path = argv[++i];
        } else if (a == "--yuv-stream-interval" && i + 1 < argc) {
            yuv_stream_interval = std::stoi(argv[++i]);
        } else if (a == "--yuv-full-range") {
            yuv_full_range = true;
        } else if (a == "--chroma-vu") {
            chroma_vu = true;
        } else if (a == "--bt601") {
            bt709 = false;
        } else if (a == "--stream-nv12") {
            input_nv12_stream = true;
        } else if (a == "--device" && i + 1 < argc) {
            device = argv[++i];
        } else if (a == "--width" && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        } else if (a == "--height" && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        } else if (a == "--frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (a == "--display") {
            display = true;
        } else if (a == "--fb" && i + 1 < argc) {
            fb_path = argv[++i];
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (input_nv12_stream && input_path.empty()) {
        std::cerr << "--stream-nv12 需要同时指定 --input <文件路径或 ->\n";
        return 1;
    }

    if (!input_path.empty() && !input_nv12_stream && input_path_is_v4l2_device(input_path)) {
        device = input_path;
        input_path.clear();
    }

    if (argc == 1) {
        model_path = "../model/yolov8n.rknn";
        yuv_stream_path = "./out/stream.yuv";
        max_frames = 300;
        std::cerr << "使用默认参数：--model " << model_path << " --save-yuv-stream " << yuv_stream_path
                  << " --frames " << max_frames << "\n";
    }

    if (model_path.empty()) {
        model_path = "../model/yolov8n.rknn";
    }

    /* 流/摄像头：--output foo.yuv 等价于向该文件写带框 NV12 视频流（不占用 --save-yuv-stream） */
    if (yuv_stream_path.empty() && path_is_nv12_stream_file(output_path)) {
        if (input_nv12_stream || input_path.empty()) {
            yuv_stream_path = output_path;
        }
    }

    if (init_post_process(labels_path.empty() ? nullptr : labels_path.c_str()) != 0) {
        std::cerr << "init_post_process 失败\n";
    }

    rknn_app_context_t rknn_ctx;
    std::memset(&rknn_ctx, 0, sizeof(rknn_ctx));
    if (init_yolov8_model(model_path.c_str(), &rknn_ctx) != 0) {
        std::cerr << "init_yolov8_model 失败\n";
        deinit_post_process();
        return 1;
    }

    if (!input_path.empty() && input_nv12_stream) {
        const std::string stream_save_dir = stream_nv12_save_directory(save_dir, output_path);
        if (stream_save_dir.empty() && yuv_stream_path.empty() && !display) {
            std::cerr << "提示: 未保存画面；可加 --output out.yuv（带框NV12流）、--save-dir、--save-yuv-stream 或 --display\n";
        }
        const int sret =
            run_nv12_stream_input(input_path, width, height, max_frames, &rknn_ctx, display, fb_path, timing,
                                  stream_save_dir, save_interval, save_yuv_only, yuv_stream_path, yuv_stream_interval,
                                  yuv_full_range, chroma_vu, bt709);
        release_yolov8_model(&rknn_ctx);
        deinit_post_process();
        return sret;
    }

    if (!input_path.empty()) {
        // 图片模式：预防性检查文件是否存在，避免 read_image 内部 crash
        if (access(input_path.c_str(), R_OK) != 0) {
            std::cerr << "无法读取输入文件: " << input_path << " (" << strerror(errno) << ")\n";
            release_yolov8_model(&rknn_ctx);
            deinit_post_process();
            return 1;
        }

        image_buffer_t src_img;
        memset(&src_img, 0, sizeof(image_buffer_t));
        if (read_image(input_path.c_str(), &src_img) != 0) {
            std::cerr << "加载图片失败: " << input_path << "\n";
            release_yolov8_model(&rknn_ctx);
            deinit_post_process();
            return 1;
        }

        object_detect_result_list od_results;
        auto t_start = std::chrono::steady_clock::now();
        int ret = inference_yolov8_model(&rknn_ctx, &src_img, &od_results);
        auto t_end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        if (ret == 0) {
            printf("推理耗时: %.2f ms\n", ms);
            printf("检测到对象数量: %d\n", od_results.count);
            for (int i = 0; i < od_results.count; i++) {
                object_detect_result* det = &od_results.results[i];
                printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det->cls_id),
                       det->box.left, det->box.top, det->box.right, det->box.bottom,
                       det->prop);
                
                draw_rectangle(&src_img, det->box.left, det->box.top, 
                               det->box.right - det->box.left, 
                               det->box.bottom - det->box.top, COLOR_BLUE, 3);
                char text[256];
                sprintf(text, "%s %.1f%%", coco_cls_to_name(det->cls_id), det->prop * 100);
                draw_text(&src_img, text, det->box.left, det->box.top - 20, COLOR_RED, 10);
            }
            if (write_image(output_path.c_str(), &src_img) == 0) {
                printf("结果已保存至: %s\n", output_path.c_str());
            } else {
                std::cerr << "保存图片失败: " << output_path << "\n";
            }
        } else {
            std::cerr << "推理失败\n";
        }

        free(src_img.virt_addr);
        release_yolov8_model(&rknn_ctx);
        deinit_post_process();
        return 0;
    }

    V4l2Nv12Capture cap;
    if (!cap.open_device(device, width, height)) {
        release_yolov8_model(&rknn_ctx);
        deinit_post_process();
        return 1;
    }
    if (cap.fourcc() != V4L2_PIX_FMT_NV12) {
        std::cerr << "当前设备输出不是 NV12，本示例仅支持 NV12\n";
        release_yolov8_model(&rknn_ctx);
        deinit_post_process();
        return 1;
    }

    Framebuffer fb;
    if (display) {
        if (!fb_open(fb, fb_path) || (fb.bpp != 16 && fb.bpp != 32)) {
            std::cerr << "framebuffer 不可用或 bpp 非 16/32，关闭显示\n";
            if (fb.fd >= 0) fb_close(fb);
            display = false;
        }
    }

    if (!cap.start()) {
        release_yolov8_model(&rknn_ctx);
        deinit_post_process();
        return 1;
    }

    FILE* yuv_stream_fp = nullptr;
    if (!yuv_stream_path.empty()) {
        yuv_stream_fp = std::fopen(yuv_stream_path.c_str(), "wb");
        if (!yuv_stream_fp) {
            std::cerr << "无法打开 YUV 流文件 " << yuv_stream_path << "：" << std::strerror(errno) << "\n";
            cap.stop();
            cap.cleanup();
            release_yolov8_model(&rknn_ctx);
            deinit_post_process();
            return 1;
        }
        std::cerr << "NV12 视频流写入: " << yuv_stream_path
                  << " （" << cap.width() << "x" << cap.height() << ", 每 " << yuv_stream_interval << " 帧追加一次）\n";
    }

    const uint32_t cap_w = cap.width();
    const uint32_t cap_h = cap.height();
    const uint32_t bpl = cap.bytesperline_y();
    const uint32_t uv_off = cap.uv_plane_offset();
    const bool need_tight = (bpl != cap_w) || (uv_off != bpl * cap_h);
    std::vector<uint8_t> tight_nv12;
    if (need_tight) {
        tight_nv12.resize(static_cast<size_t>(cap_w) * cap_h * 3 / 2);
        std::cerr << "NV12 bytesperline=" << bpl << " uv_offset=" << uv_off << "，使用紧密缓冲（推理/画框/显示/流）\n";
    }
    if (display) {
        std::cerr << "预览色彩: "
                  << (yuv_full_range ? "full" : (bt709 ? "BT.709" : "BT.601")) << " 限幅"
                  << (chroma_vu ? " + NV21(VU) 色度" : " + NV12(UV) 色度") << "\n";
    }

    int processed = 0;
    double total_ms = 0;
    while (max_frames == 0 || processed < max_frames) {
        auto t_loop0 = std::chrono::steady_clock::now();

        uint8_t* frame_ptr = nullptr;
        size_t used = 0;
        uint32_t buf_idx = 0;
        if (!cap.dequeue_frame(&frame_ptr, &used, &buf_idx)) {
            std::cerr << "VIDIOC_DQBUF 失败\n";
            break;
        }

        image_buffer_t img {};
        img.width = static_cast<int>(cap_w);
        img.height = static_cast<int>(cap_h);
        img.height_stride = img.height;
        img.format = IMAGE_FORMAT_YUV420SP_NV12;
        img.size = static_cast<int>(img.width * img.height * 3 / 2);
        img.fd = -1;

        uint8_t* work_ptr = frame_ptr;
        if (need_tight) {
            nv12_copy_tight(frame_ptr, img.width, img.height, bpl, uv_off, tight_nv12.data());
            work_ptr = tight_nv12.data();
        }
        img.virt_addr = work_ptr;
        img.width_stride = img.width;

        object_detect_result_list od_results {};
        if (inference_yolov8_model(&rknn_ctx, &img, &od_results) == 0) {
            char text[256];
            for (int i = 0; i < od_results.count; i++) {
                object_detect_result* det = &od_results.results[i];
                int x1 = det->box.left;
                int y1 = det->box.top;
                int x2 = det->box.right;
                int y2 = det->box.bottom;
                draw_rectangle(&img, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 2);
                std::snprintf(text, sizeof(text), "%s %.0f%%", coco_cls_to_name(det->cls_id),
                              det->prop * 100.f);
                draw_text(&img, text, x1, y1 - 12, COLOR_RED, 10);
            }
        }

        if (display) {
            const uint32_t blit_stride = need_tight ? cap_w : bpl;
            const uint32_t blit_uv = need_tight ? (cap_w * cap_h) : uv_off;
            blit_nv12_to_fb(work_ptr, img.width, img.height, blit_stride, blit_uv, fb, yuv_full_range, chroma_vu,
                            bt709);
        }

        if (!save_dir.empty() && (processed % save_interval) == 0) {
            char path_jpg[512];
            char path_yuv[512];
            std::snprintf(path_jpg, sizeof(path_jpg), "%s/frame_%06d.jpg", save_dir.c_str(), processed);
            std::snprintf(path_yuv, sizeof(path_yuv), "%s/frame_%06d.yuv", save_dir.c_str(), processed);
            save_frame_image(path_jpg, path_yuv, &img, save_yuv_only);
        }

        if (yuv_stream_fp && (processed % yuv_stream_interval) == 0) {
            size_t frame_bytes = static_cast<size_t>(img.width) * static_cast<size_t>(img.height) * 3 / 2;
            size_t nw = std::fwrite(img.virt_addr, 1, frame_bytes, yuv_stream_fp);
            if (nw != frame_bytes) {
                std::cerr << "写入 YUV 流失败\n";
            }
        }

        cap.queue_buffer(buf_idx);
        processed++;

        auto t_loop1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_loop1 - t_loop0).count();
        total_ms += ms;
        if (timing) {
            std::cerr << "frame " << (processed - 1) << "  loop_time=" << ms << " ms\n";
        } else if ((processed % 30) == 0) {
            double avg = total_ms / static_cast<double>(processed);
            std::cerr << "已处理 " << processed << " 帧, 平均循环 " << avg << " ms/帧\r" << std::flush;
        }
    }

    std::cerr << "\n";
    if (yuv_stream_fp) {
        std::fclose(yuv_stream_fp);
        yuv_stream_fp = nullptr;
        if (!yuv_stream_path.empty()) {
            std::cout << "YUV 流已关闭: " << yuv_stream_path << "\n";
        }
    }
    cap.stop();
    cap.cleanup();
    if (display) fb_close(fb);

    release_yolov8_model(&rknn_ctx);
    deinit_post_process();
    if (processed > 0) {
        std::cout << "结束，共 " << processed << " 帧，平均循环时间 "
                  << (total_ms / static_cast<double>(processed)) << " ms/帧\n";
    } else {
        std::cout << "结束，共 0 帧\n";
    }
    return 0;
}
