// 摄像头 NV12 -> RKNN YOLOv8 推理 -> NV12 上画框；可选 framebuffer 预览（无 OpenCV）

#include "v4l2_capture.hpp"

#include <linux/fb.h>
#include <linux/videodev2.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "yolov8.h"
#include "postprocess.h"

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

static void usage(const char* p) {
    std::cerr << "用法: " << p << " [选项]   或   " << p << " --model <yolov8.rknn> [选项]\n"
              << "  无参数时默认：--model ../model/yolov8n.rknn --save-yuv-stream ./out/stream.yuv --frames 300\n"
              << "  --model <yolov8.rknn>   指定模型（有其它参数时必须带；单独运行可无参使用上述默认）\n"
              << "  --device /dev/video0   默认 /dev/video0\n"
              << "  --width 640 --height 480   采集分辨率（默认 640x480）\n"
              << "  --frames N   处理 N 帧后退出；0 表示一直运行\n"
              << "  （类别名自 ./model/coco_80_labels_list.txt，请在运行目录下放置 model/）\n"
              << "  --display    可选：/dev/fb0 预览（NV12->RGB，不缩放仅裁剪）\n"
              << "  --fb /dev/fb0\n"
              << "  --timing     每帧打印本循环耗时（ms）\n"
              << "  --save-dir <目录>   保存带框画面：优先 JPG，失败则同序号 .yuv（NV12 裸流）\n"
              << "  --save-yuv-only     仅保存 .yuv，不做 JPG（数据流为 NV12，扩展名 .yuv）\n"
              << "  --save-interval N   每 N 帧保存一次（默认 30，仅 --save-dir 时有效）\n"
              << "  --save-yuv-stream <文件.yuv>  连续写入 NV12 视频流（每帧一帧，与 ffmpeg -f rawvideo -pix_fmt nv12 兼容）\n"
              << "  --yuv-stream-interval N   流式写入时每 N 帧写 1 帧（默认 1=每帧都写）\n"
              << "  --yuv-full-range   预览用 full range YUV→RGB（与限幅矩阵互斥优先项）\n"
              << "  --chroma-vu        预览按 NV21(VU) 解释色度（偏青绿可试）\n"
              << "  --bt601            预览用 BT.601 限幅矩阵（默认 BT.709）\n";
}

int main(int argc, char** argv) {
    std::string model_path;
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

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model_path = argv[++i];
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

    if (argc == 1) {
        model_path = "../model/yolov8n.rknn";
        yuv_stream_path = "./out/stream.yuv";
        max_frames = 300;
        std::cerr << "使用默认参数：--model " << model_path << " --save-yuv-stream " << yuv_stream_path
                  << " --frames " << max_frames << "\n";
    }

    if (model_path.empty()) {
        std::cerr << "必须指定 --model <xxx.rknn>（或无任何参数以使用内置默认）\n";
        usage(argv[0]);
        return 1;
    }
    if (save_interval <= 0) save_interval = 1;
    if (yuv_stream_interval <= 0) yuv_stream_interval = 1;
    if (!save_dir.empty()) {
        if (mkdir(save_dir.c_str(), 0755) != 0 && errno != EEXIST) {
            std::cerr << "无法创建目录 " << save_dir << "：" << std::strerror(errno) << "\n";
            return 1;
        }
    }
    if (!yuv_stream_path.empty()) {
        size_t slash = yuv_stream_path.find_last_of('/');
        if (slash != std::string::npos) {
            std::string dir = yuv_stream_path.substr(0, slash);
            if (!dir.empty() && dir != ".") {
                if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
                    std::cerr << "无法创建目录 " << dir << "：" << std::strerror(errno) << "\n";
                    return 1;
                }
            }
        }
    }

    V4l2Nv12Capture cap;
    if (!cap.open_device(device, width, height)) return 1;
    if (cap.fourcc() != V4L2_PIX_FMT_NV12) {
        std::cerr << "当前设备输出不是 NV12，本示例仅支持 NV12\n";
        return 1;
    }

    rknn_app_context_t rknn_ctx;
    std::memset(&rknn_ctx, 0, sizeof(rknn_ctx));
    if (init_post_process() != 0) {
        std::cerr << "init_post_process 失败（需当前工作目录下存在 ./model/coco_80_labels_list.txt）\n";
        return 1;
    }
    if (init_yolov8_model(model_path.c_str(), &rknn_ctx) != 0) {
        std::cerr << "init_yolov8_model 失败\n";
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
