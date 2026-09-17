#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/command.h>
#include <ncnn/pipeline.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <algorithm>

static double now_ms() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static bool g_warp_gpu = true;

static const char* warp_glsl = R"GLSL(
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(binding = 0) buffer b0 { float x[]; };
layout(binding = 1) buffer b1 { float flow[]; };
layout(binding = 2) buffer b2 { float top[]; };
layout(push_constant) uniform PC {
    int w; int h; int fw; int fh; int ox; int oy; int ch;
} p;
void main() {
    int xi = int(gl_GlobalInvocationID.x);
    int y  = int(gl_GlobalInvocationID.y);
    int c  = int(gl_GlobalInvocationID.z);
    if (xi >= p.w || y >= p.h || c >= p.ch) return;
    int fidx = ((y + p.oy) * p.fw + (xi + p.ox)) * 2;
    float sx = float(xi) + flow[fidx];
    float sy = float(y)  + flow[fidx + 1];
    float v = 0.0;
    if (sx >= 0.0 && sy >= 0.0 && sx <= float(p.w - 1) && sy <= float(p.h - 1)) {
        int x0 = int(sx); int y0 = int(sy);
        int x1 = min(x0 + 1, p.w - 1); int y1 = min(y0 + 1, p.h - 1);
        float ax = sx - float(x0); float ay = sy - float(y0);
        int base = c * p.w * p.h;
        v = (1.0 - ax) * (1.0 - ay) * x[base + y0 * p.w + x0]
          + ax * (1.0 - ay) * x[base + y0 * p.w + x1]
          + (1.0 - ax) * ay * x[base + y1 * p.w + x0]
          + ax * ay * x[base + y1 * p.w + x1];
    }
    top[c * p.w * p.h + y * p.w + xi] = v;
}
)GLSL";

class Warp_layer : public ncnn::Layer {
public:
    ncnn::Pipeline* pipeline;
    Warp_layer() {
        pipeline = 0;
        support_vulkan = g_warp_gpu;
        support_fp16_storage = false;
        support_packing = false;
    }
    virtual int create_pipeline(const ncnn::Option& opt) {
        if (!opt.use_vulkan_compute || !support_vulkan) return 0;
        std::vector<uint32_t> spirv;
        int ret = ncnn::compile_spirv_module(warp_glsl, (int)strlen(warp_glsl), opt, spirv);
        if (ret != 0) {
            fprintf(stderr, "WARP_VULKAN_OFF compile=%d\n", ret);
            support_vulkan = false;
            return 0;
        }
        pipeline = new ncnn::Pipeline(ncnn::get_gpu_device(0));
        pipeline->set_optimal_local_size_xyz(8, 8, 1);
        ret = pipeline->create(spirv.data(), spirv.size() * 4,
                               std::vector<ncnn::vk_specialization_type>());
        if (ret != 0) {
            fprintf(stderr, "WARP_VULKAN_OFF create=%d\n", ret);
            support_vulkan = false;
            delete pipeline;
            pipeline = 0;
            return 0;
        }
        fprintf(stderr, "WARP_VULKAN_ON\n");
        return 0;
    }
    virtual int destroy_pipeline(const ncnn::Option& opt) {
        delete pipeline;
        pipeline = 0;
        return 0;
    }
    virtual int forward_gpu(const std::vector<ncnn::VkMat>& bottom_blobs,
                            std::vector<ncnn::VkMat>& top_blobs,
                            ncnn::VkCompute& cmd,
                            const ncnn::Option& opt) const {
        const ncnn::VkMat& x = bottom_blobs[0];
        const ncnn::VkMat& flow = bottom_blobs[1];
        const int w = x.w, h = x.h, ch = x.c;
        if (flow.c != 2 || flow.w < w || flow.h < h) {
            fprintf(stderr, "WARP_GPU_GUARD1 w=%d h=%d ch=%d fw=%d fh=%d fc=%d\n",
                    w, h, ch, flow.w, flow.h, flow.c);
            return -100;
        }
        if (x.elemsize != 4u || flow.elemsize != 4u ||
            x.elempack != 1 || flow.elempack != 1) {
            fprintf(stderr, "WARP_GPU_GUARD2 xs=%zu xe=%d fs=%zu fe=%d\n",
                    x.elemsize, x.elempack, flow.elemsize, flow.elempack);
            return -101;
        }
        ncnn::VkMat& top = top_blobs[0];
        top.create(w, h, ch, 4u, 1, opt.blob_vkallocator);
        if (top.empty()) return -100;
        std::vector<ncnn::VkMat> bindings(3);
        bindings[0] = x;
        bindings[1] = flow;
        bindings[2] = top;
        std::vector<ncnn::vk_constant_type> k(7);
        k[0].i = w;
        k[1].i = h;
        k[2].i = flow.w;
        k[3].i = flow.h;
        k[4].i = (flow.w - w) / 2;
        k[5].i = (flow.h - h) / 2;
        k[6].i = ch;
        cmd.record_pipeline(pipeline, bindings, k, top);
        return 0;
    }
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const {
        const ncnn::Mat& x = bottom_blobs[0];
        const ncnn::Mat& flow = bottom_blobs[1];
        const int w = x.w, h = x.h, ch = x.c;
        if (flow.c != 2 || flow.w < w || flow.h < h) {
            fprintf(stderr, "WARP_GUARD1 w=%d h=%d ch=%d fw=%d fh=%d fc=%d\n",
                    w, h, ch, flow.w, flow.h, flow.c);
            return -100;
        }
        if (x.elemsize != 4u || flow.elemsize != 4u ||
            x.elempack != 1 || flow.elempack != 1) {
            fprintf(stderr, "WARP_GUARD2 xs=%zu xe=%d fs=%zu fe=%d\n",
                    x.elemsize, x.elempack, flow.elemsize, flow.elempack);
            return -101;
        }
        const int oy = (flow.h - h) / 2;
        const int ox = (flow.w - w) / 2;
        ncnn::Mat& top = top_blobs[0];
        top.create(w, h, ch, 4u, opt.blob_allocator);
        if (top.empty()) return -100;
        const float* f0 = flow.channel(0);
        const float* f1 = flow.channel(1);
        const int fstride = flow.w;
        const int nthreads = 4;
        auto body = [&](int y0, int y1) {
            for (int c = 0; c < ch; c++) {
                const float* xp = x.channel(c);
                float* tp = top.channel(c);
                for (int y = y0; y < y1; y++) {
                    const int fy_row = (y + oy) * fstride + ox;
                    for (int xi = 0; xi < w; xi++) {
                        const float sx = xi + f0[fy_row + xi];
                        const float sy = y  + f1[fy_row + xi];
                        if (sx < 0.f || sx > (float)(w - 1) ||
                            sy < 0.f || sy > (float)(h - 1)) {
                            tp[y * w + xi] = 0.f;
                            continue;
                        }
                        const int x0 = (int)sx, y0i = (int)sy;
                        const int x1 = std::min(x0 + 1, w - 1);
                        const int y1i = std::min(y0i + 1, h - 1);
                        const float ax = sx - (float)x0, ay = sy - (float)y0i;
                        tp[y * w + xi] =
                            (1 - ax) * (1 - ay) * xp[y0i * w + x0] +
                            ax * (1 - ay) * xp[y0i * w + x1] +
                            (1 - ax) * ay * xp[y1i * w + x0] +
                            ax * ay * xp[y1i * w + x1];
                    }
                }
            }
        };
        const int per = (h + nthreads - 1) / nthreads;
        std::vector<std::thread> ths;
        for (int t = 1; t < nthreads; t++) {
            int y0 = t * per;
            int y1 = std::min(h, y0 + per);
            if (y0 < y1) ths.push_back(std::thread(body, y0, y1));
        }
        body(0, std::min(h, per));
        for (std::thread& th : ths) th.join();
        return 0;
    }
};

static ncnn::Layer* Warp_layer_creator(void*) { return new Warp_layer; }

static int run_net(bool warp_gpu, const char* param, const char* bin,
                   int W, int H, int runs, const char* label) {
    g_warp_gpu = warp_gpu;
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.set_vulkan_device(0);
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_packing_layout = true;
    net.opt.num_threads = 4;
    net.register_custom_layer("rife.Warp", Warp_layer_creator);
    if (net.load_param(param) != 0) {
        fprintf(stderr, "ERR: load_param failed\n"); fflush(NULL);
        return 2;
    }
    if (net.load_model(bin) != 0) {
        fprintf(stderr, "ERR: load_model failed\n"); fflush(NULL);
        return 2;
    }
    ncnn::Mat a(W, H, 3), b(W, H, 3);
    for (int c = 0; c < 3; c++) {
        float* pa = a.channel(c); float* pb = b.channel(c);
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            pa[y*W+x] = fmodf((x + c * 37.0f) / 255.0f, 1.0f);
            pb[y*W+x] = fmodf((x + y + c * 37.0f) / 255.0f, 1.0f);
        }
    }
    ncnn::Mat six(W, H, 6);
    for (int c = 0; c < 3; c++) {
        memcpy(six.channel(c), a.channel(c), (size_t)W * H * 4);
        memcpy(six.channel(3 + c), b.channel(c), (size_t)W * H * 4);
    }
    ncnn::Mat ts(1, 1, 1); ts.channel(0)[0] = 0.5f;
    const std::vector<int>& ins = net.input_indexes();
    std::vector<ncnn::Mat> shp = net.input_shapes();
    int c_in = ((int)shp.size() == 1 && shp[0].dims == 3) ? shp[0].c : 0;
    printf("inputs=%d c_in=%d\n", (int)ins.size(), c_in);
    auto feed = [&](ncnn::Extractor& ex) {
        if ((int)ins.size() >= 2) {
            ex.input(ins[0], a); ex.input(ins[1], b);
            if ((int)ins.size() >= 3) ex.input(ins[2], ts);
        } else if (c_in >= 7) {
            ncnn::Mat in(W, H, c_in);
            for (int c = 0; c < c_in; c++) {
                float* p = in.channel(c);
                if (c < 3) memcpy(p, a.channel(c), (size_t)W * H * 4);
                else if (c < 6) memcpy(p, b.channel(c - 3), (size_t)W * H * 4);
                else if (c == 6) { for (int i = 0; i < W * H; i++) p[i] = 0.5f; }
                else memset(p, 0, (size_t)W * H * 4);
            }
            ex.input(ins[0], in);
        } else if (c_in == 6) {
            ex.input(ins[0], six);
        } else {
            ex.input(ins[0], six);
        }
    };
    ncnn::Mat out;
    {
        ncnn::Extractor ex = net.create_extractor();
        feed(ex);
        int ret = ex.extract(net.output_indexes()[0], out);
        if (ret != 0) {
            fprintf(stderr, "EXTRACT_RET=%d (%s)\n", ret, label);
            fflush(NULL);
            return 3;
        }
    }
    double mn = 1e9, mx = -1e9, sum = 0;
    int n = out.w * out.h * out.c;
    const float* p = out;
    for (int i = 0; i < n; i++) { double v = p[i]; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
    printf("out w=%d h=%d c=%d min=%.3f max=%.3f mean=%.3f (%s)\n",
           out.w, out.h, out.c, mn, mx, sum / n, label);
    double t0 = now_ms();
    for (int r = 0; r < runs; r++) {
        ncnn::Extractor e2 = net.create_extractor();
        feed(e2);
        ncnn::Mat o2;
        e2.extract(net.output_indexes()[0], o2);
    }
    double dt = now_ms() - t0;
    printf("time: %d runs, %.1f ms/run => %.2f fps at %dx%d (%s)\n",
           runs, dt / runs, runs * 1000.0 / dt, W, H, label);
    return 0;
}

int main(int argc, char** argv) {
    const char* param = argc > 1 ? argv[1] : "flownet.param";
    const char* bin   = argc > 2 ? argv[2] : "flownet.bin";
    int W = argc > 3 ? atoi(argv[3]) : 512;
    int H = argc > 4 ? atoi(argv[4]) : 288;
    int runs = argc > 5 ? atoi(argv[5]) : 5;
    const char* mode = argc > 6 ? argv[6] : "gpu";
    if (ncnn::get_gpu_count() == 0) {
        fprintf(stderr, "ERR: no vulkan gpu\n"); fflush(NULL); _exit(1);
    }
    if (strcmp(mode, "cpufb") == 0) {
        return run_net(false, param, bin, W, H, runs, "cpufb") == 0 ? 0 : 3;
    }
    int rc = run_net(true, param, bin, W, H, runs, "gpu");
    if (rc != 0) {
        fprintf(stderr, "GPU_PATH_FAILED rc=%d; auto cpufb control\n", rc);
        fflush(NULL);
        rc = run_net(false, param, bin, W, H, runs, "cpufb-auto");
        fflush(NULL);
        return rc != 0 ? 3 : 4;
    }
    return 0;
}
