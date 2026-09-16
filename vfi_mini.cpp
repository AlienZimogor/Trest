#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <vector>

static double now_ms() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    const char* param = argc > 1 ? argv[1] : "flownet.param";
    const char* bin   = argc > 2 ? argv[2] : "flownet.bin";
    int W = argc > 3 ? atoi(argv[3]) : 512;
    int H = argc > 4 ? atoi(argv[4]) : 288;
    int runs = argc > 5 ? atoi(argv[5]) : 5;
    if (ncnn::get_gpu_count() == 0) { fprintf(stderr, "ERR: no vulkan gpu\n"); return 1; }
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.set_vulkan_device(0);
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = false;
    net.opt.num_threads = 4;
    if (net.load_param(param) != 0) { fprintf(stderr, "ERR: load_param failed\n"); return 2; }
    if (net.load_model(bin) != 0) { fprintf(stderr, "ERR: load_model failed\n"); return 2; }
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
            fprintf(stderr, "ERR: extract failed\n"); return 3;
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
    printf("time: %d runs, %.1f ms/run => %.2f fps at %dx%d (vulkan)\n",
           runs, dt / runs, runs * 1000.0 / dt, W, H);
    return 0;
}
