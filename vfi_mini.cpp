#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <algorithm>

static double now_ms() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static bool g_identity = false;

class Warp_layer : public ncnn::Layer {
public:
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const {
        const ncnn::Mat& x = bottom_blobs[0];
        const int w = x.w, h = x.h, ch = x.c;
        if (g_identity) {
            ncnn::Mat& top = top_blobs[0];
            top.create(w, h, ch, 4u, opt.blob_allocator);
            if (top.empty()) return -100;
            memcpy(top.data, x.data, (size_t)w * h * ch * 4u);
            return 0;
        }
        const ncnn::Mat& flow = bottom_blobs[1];
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
        for (int c = 0; c < ch; c++) {
            const float* xp = x.channel(c);
            float* tp = top.channel(c);
            for (int y = 0; y < h; y++) {
                const int fy_row = (y + oy) * fstride + ox;
                for (int xi = 0; xi < w; xi++) {
                    const float sx = xi + f0[fy_row + xi];
                    const float sy = y  + f1[fy_row + xi];
                    if (sx < 0.f || sx > (float)(w - 1) ||
                        sy < 0.f || sy > (float)(h - 1)) {
                        tp[y * w + xi] = 0.f;
                        continue;
                    }
                    const int x0 = (int)sx, y0 = (int)sy;
                    const int x1 = std::min(x0 + 1, w - 1);
                    const int y1 = std::min(y0 + 1, h - 1);
                    const float ax = sx - (float)x0, ay = sy - (float)y0;
                    tp[y * w + xi] =
                        (1 - ax) * (1 - ay) * xp[y0 * w + x0] +
                        ax * (1 - ay) * xp[y0 * w + x1] +
                        (1 - ax) * ay * xp[y1 * w + x0] +
                        ax * ay * xp[y1 * w + x1];
                }
            }
        }
        return 0;
    }
};

static ncnn::Layer* Warp_layer_creator(void*) { return new Warp_layer; }

int main(int argc, char** argv) {
    const char* param = argc > 1 ? argv[1] : "flownet.param";
    const char* bin   = argc > 2 ? argv[2] : "flownet.bin";
    int W = argc > 3 ? atoi(argv[3]) : 512;
    int H = argc > 4 ? atoi(argv[4]) : 288;
    int runs = argc > 5 ? atoi(argv[5]) : 5;
    const char* mode = argc > 6 ? argv[6] : "gpu";
    g_identity = (strcmp(mode, "gpuid") == 0);
    const bool use_gpu = (strcmp(mode, "cpu") != 0);
    if (use_gpu && ncnn::get_gpu_count() == 0) {
        fprintf(stderr, "ERR: no vulkan gpu\n"); fflush(NULL); _exit(1);
    }
    ncnn::Net net;
    net.opt.use_vulkan_compute = use_gpu;
    if (use_gpu) net.set_vulkan_device(0);
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_packing_layout = true;
    net.opt.num_threads = 4;
    net.register_custom_layer("rife.Warp", Warp_layer_creator);
    if (net.load_param(param) != 0) {
        fprintf(stderr, "ERR: load_param failed\n"); fflush(NULL); _exit(2);
    }
    if (net.load_model(bin) != 0) {
        fprintf(stderr, "ERR: load_model failed\n"); fflush(NULL); _exit(2);
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
    ncnn::Mat out;
    {
        ncnn::Extractor ex = net.create_extractor();
        if ((int)ins.size() >= 2) {
            ex.input(ins[0], a); ex.input(ins[1], b);
            if ((int)ins.size() >= 3) ex.input(ins[2], ts);
        } else {
            ex.input(ins[0], six);
        }
        if (ex.extract(net.output_indexes()[0], out) != 0) {
            fprintf(stderr, "ERR: extract failed\n"); fflush(NULL); _exit(3);
        }
    }
    double mn = 1e9, mx = -1e9, sum = 0;
    int n = out.w * out.h * out.c;
    const float* p = out;
    for (int i = 0; i < n; i++) { double v = p[i]; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
    printf("out w=%d h=%d c=%d min=%.3f max=%.3f mean=%.3f\n",
           out.w, out.h, out.c, mn, mx, sum / n);
    double t0 = now_ms();
    for (int r = 0; r < runs; r++) {
        ncnn::Extractor e2 = net.create_extractor();
        if ((int)ins.size() >= 2) {
            e2.input(ins[0], a); e2.input(ins[1], b);
            if ((int)ins.size() >= 3) e2.input(ins[2], ts);
        } else {
            e2.input(ins[0], six);
        }
        ncnn::Mat o2;
        e2.extract(net.output_indexes()[0], o2);
    }
    double dt = now_ms() - t0;
    printf("time: %d runs, %.1f ms/run => %.2f fps at %dx%d (%s)\n",
           runs, dt / runs, runs * 1000.0 / dt, W, H, mode);
    return 0;
}
