#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/command.h>
#include <ncnn/pipeline.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/wait.h>
#include <thread>
#include <atomic>
#include <string>
#include <vector>
#include <algorithm>

static double now_ms() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static bool g_warp_gpu = false;
static int g_reshmode = 1;

static std::atomic<int> g_frame{0};
static std::atomic<int> g_frames{0};
static volatile const char* g_phase = "init";

static void start_heartbeat() {
    std::thread([]() {
        double t0 = now_ms();
        for (;;) {
            usleep(1000 * 1000);
            fprintf(stderr, "ALIVE t=%.1fs phase=%s frame=%d/%d\n",
                    (now_ms() - t0) / 1000.0, g_phase,
                    g_frame.load(), g_frames.load());
            fflush(stderr);
        }
    }).detach();
}

static void dump_param_head(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "DUMP: cannot open %s\n", path); fflush(stderr); return; }
    unsigned char buf[32];
    size_t n = fread(buf, 1, sizeof(buf), fp);
    fclose(fp);
    fprintf(stderr, "DUMP param head (%zu bytes):", n);
    for (size_t i = 0; i < n; i++) fprintf(stderr, " %02x", buf[i]);
    fprintf(stderr, "\nDUMP ascii: |");
    for (size_t i = 0; i < n; i++) fputc(isprint(buf[i]) ? buf[i] : '.', stderr);
    fprintf(stderr, "|\n");
    fflush(stderr);
}

static void parse_opt(const char* e, int& threads, bool& fp16) {
    threads = 8; fp16 = false;
    if (!e) return;
    std::string s(e);
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t col = s.find(':', pos);
        std::string tok = s.substr(pos,
            col == std::string::npos ? std::string::npos : col - pos);
        if (tok == "thr=1") threads = 1;
        else if (tok == "thr=4") threads = 4;
        else if (tok == "fp16=1") fp16 = true;
        if (col == std::string::npos) break;
        pos = col + 1;
    }
}

class Reshape_fix : public ncnn::Layer {
public:
    int w, h, c11, c2;
    Reshape_fix() {
        w = h = -1; c11 = c2 = 0;
        support_vulkan = false;
        support_fp16_storage = false;
        support_packing = false;
    }
    virtual int load_param(const ncnn::ParamDict& pd) {
        w = pd.get(0, -1); h = pd.get(1, -1);
        c11 = pd.get(11, 0); c2 = pd.get(2, 0);
        return 0;
    }
    virtual int forward(const std::vector<ncnn::Mat>& bb,
                        std::vector<ncnn::Mat>& tb,
                        const ncnn::Option& opt) const {
        const ncnn::Mat& b = bb[0];
        const int total = (int)b.total();
        ncnn::Mat& top = tb[0];
        if (name == "reshape_133") {
            top.create(b.w, b.h, b.c, b.elemsize, b.elempack, opt.blob_allocator);
            if (top.empty()) return -100;
            memcpy(top.data, b.data, (size_t)total * b.elemsize);
            return 0;
        }
        int tw = w > 0 ? w : b.w;
        int th = h > 0 ? h : b.h;
        if (c11 > 0 && tw == b.w && th == b.h && c11 < b.c) {
            top.create(tw, th, c11, b.elemsize, b.elempack, opt.blob_allocator);
            if (top.empty()) return -100;
            for (int c = 0; c < c11; c++)
                memcpy(top.channel(c), b.channel(c), (size_t)tw * th * b.elemsize);
            return 0;
        }
        if (c2 > 0 && tw == b.w && th == b.h && c2 == 2 * b.c) {
            top.create(tw, th, c2, b.elemsize, b.elempack, opt.blob_allocator);
            if (top.empty()) return -100;
            for (int c = 0; c < c2; c++)
                memcpy(top.channel(c), b.channel(c % b.c), (size_t)tw * th * b.elemsize);
            return 0;
        }
        int tc = 0;
        if (tw > 0 && th > 0 && total % (tw * th) == 0) tc = total / (tw * th);
        if (tc == 0) {
            tc = c11 > 0 ? c11 : (c2 > 0 ? c2 : b.c);
            if (tw <= 0 || th <= 0 || tw * th * tc != total) {
                fprintf(stderr, "RESHAPE_FAIL %s in=%dx%dx%d total=%d tw=%d th=%d c11=%d c2=%d\n",
                        name.c_str(), b.w, b.h, b.c, total, tw, th, c11, c2);
                return -100;
            }
        }
        top.create(tw, th, tc, b.elemsize, b.elempack, opt.blob_allocator);
        if (top.empty()) return -100;
        memcpy(top.data, b.data, (size_t)total * b.elemsize);
        return 0;
    }
};

class Slice_fix : public ncnn::Layer {
public:
    ncnn::Mat slices;
    int axis;
    Slice_fix() {
        axis = 0;
        support_vulkan = false;
        support_fp16_storage = false;
        support_packing = false;
    }
    virtual int load_param(const ncnn::ParamDict& pd) {
        slices = pd.get(0, ncnn::Mat());
        axis = pd.get(1, 0);
        return 0;
    }
    virtual int forward(const std::vector<ncnn::Mat>& bb,
                        std::vector<ncnn::Mat>& tb,
                        const ncnn::Option& opt) const {
        const ncnn::Mat& b = bb[0];
        const int C = b.c;
        const int n = (int)tb.size();
        if (C <= 0 || n <= 0) {
            for (int i = 0; i < n; i++) tb[i] = ncnn::Mat();
            return 0;
        }
        int pos_sum = 0, n_auto = 0;
        for (int i = 0; i < n; i++) {
            int s = (i < slices.w) ? (int)slices[i] : -233;
            if (s > 0) pos_sum += s; else n_auto++;
        }
        int auto_each = n_auto > 0 ? std::max(1, (C - pos_sum) / n_auto) : 0;
        int off = 0;
        for (int i = 0; i < n; i++) {
            int s = (i < slices.w) ? (int)slices[i] : -233;
            int want = (s > 0) ? s : auto_each;
            int take = (i == n - 1) ? (C - off) : std::min(want, C - off);
            if (take <= 0) { tb[i] = ncnn::Mat(); continue; }
            ncnn::Mat& t = tb[i];
            t.create(b.w, b.h, take, b.elemsize, b.elempack, opt.blob_allocator);
            if (t.empty()) return -100;
            for (int c = 0; c < take; c++)
                memcpy(t.channel(c), b.channel(off + c), (size_t)b.w * b.h * b.elemsize);
            off += take;
        }
        return 0;
    }
};

class GridSample_fix : public ncnn::Layer {
public:
    int align_corners;
    mutable bool logged = false;
    GridSample_fix() {
        align_corners = 1;
        support_vulkan = false;
        support_fp16_storage = false;
        support_packing = false;
    }
    virtual int load_param(const ncnn::ParamDict& pd) {
        align_corners = pd.get(2, 1);
        return 0;
    }
    virtual int forward(const std::vector<ncnn::Mat>& bb,
                        std::vector<ncnn::Mat>& tb,
                        const ncnn::Option& opt) const {
        const ncnn::Mat& x = bb[0];
        const ncnn::Mat& grid = bb[1];
        const int IW = x.w, IH = x.h, C = x.c;
        const int GW = grid.w, GH = grid.h;
        if (!logged) {
            fprintf(stderr, "GRIDSAMPLE %s x=%dx%dx%d grid=%dx%dx%d\n",
                    name.c_str(), IW, IH, C, GW, GH, grid.c);
            fflush(stderr);
            logged = true;
        }
        if (x.empty() || grid.empty() || grid.elemsize != 4u || grid.elempack != 1 ||
            (int)grid.total() < GW * GH * 2) {
            fprintf(stderr, "GRIDSAMPLE_GUARD %s x=%dx%dx%d grid=%dx%dx%d total=%d\n",
                    name.c_str(), IW, IH, C, GW, GH, grid.c, (int)grid.total());
            return -100;
        }
        ncnn::Mat& top = tb[0];
        top.create(GW, GH, C, 4u, opt.blob_allocator);
        if (top.empty()) return -100;
        const float* gp = (const float*)grid.data;
        for (int c = 0; c < C; c++) {
            const float* xp = x.channel(c);
            float* tp = top.channel(c);
            for (int y = 0; y < GH; y++) {
                for (int xi = 0; xi < GW; xi++) {
                    const float g0 = gp[(y * GW + xi) * 2];
                    const float g1 = gp[(y * GW + xi) * 2 + 1];
                    float sx, sy;
                    if (align_corners) {
                        sx = (g0 + 1.f) * (IW - 1) * 0.5f;
                        sy = (g1 + 1.f) * (IH - 1) * 0.5f;
                    } else {
                        sx = ((g0 + 1.f) * IW - 1.f) * 0.5f;
                        sy = ((g1 + 1.f) * IH - 1.f) * 0.5f;
                    }
                    if (sx < 0.f || sx > (float)(IW - 1) ||
                        sy < 0.f || sy > (float)(IH - 1)) { tp[y * GW + xi] = 0.f; continue; }
                    const int x0 = (int)sx, y0 = (int)sy;
                    const int x1 = std::min(x0 + 1, IW - 1);
                    const int y1 = std::min(y0 + 1, IH - 1);
                    const float ax = sx - (float)x0, ay = sy - (float)y0;
                    tp[y * GW + xi] =
                        (1 - ax) * (1 - ay) * xp[y0 * IW + x0] +
                        ax * (1 - ay) * xp[y0 * IW + x1] +
                        (1 - ax) * ay * xp[y1 * IW + x0] +
                        ax * ay * xp[y1 * IW + x1];
                }
            }
        }
        return 0;
    }
};

static ncnn::Layer* Reshape_fix_creator(void*) { return new Reshape_fix; }
static ncnn::Layer* Slice_fix_creator(void*) { return new Slice_fix; }
static ncnn::Layer* GridSample_fix_creator(void*) { return new GridSample_fix; }

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
    mutable bool logged = false;
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
        if (ret != 0) { support_vulkan = false; return 0; }
        pipeline = new ncnn::Pipeline(ncnn::get_gpu_device(0));
        pipeline->set_optimal_local_size_xyz(8, 8, 1);
        ret = pipeline->create(spirv.data(), spirv.size() * 4,
                               std::vector<ncnn::vk_specialization_type>());
        if (ret != 0) {
            support_vulkan = false;
            delete pipeline; pipeline = 0;
            return 0;
        }
        fprintf(stderr, "WARP_VULKAN_ON\n");
        return 0;
    }
    virtual int destroy_pipeline(const ncnn::Option& opt) {
        delete pipeline; pipeline = 0; return 0;
    }
    virtual int forward_gpu(const std::vector<ncnn::VkMat>&, std::vector<ncnn::VkMat>&,
                            ncnn::VkCompute&, const ncnn::Option&) const { return -100; }
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const {
        const ncnn::Mat& x = bottom_blobs[0];
        const ncnn::Mat& flow = bottom_blobs[1];
        const int w = x.w, h = x.h, ch = x.c;
        if (!logged) {
            fprintf(stderr, "WARP %s x=%dx%dx%d flow=%dx%dx%d\n",
                    name.c_str(), w, h, ch, flow.w, flow.h, flow.c);
            fflush(stderr);
            logged = true;
        }
        if (flow.c != 2 || flow.w < w || flow.h < h) return -100;
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
                            sy < 0.f || sy > (float)(h - 1)) { tp[y*w+xi] = 0.f; continue; }
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
            int y0 = t * per, y1 = std::min(h, y0 + per);
            if (y0 < y1) ths.push_back(std::thread(body, y0, y1));
        }
        body(0, std::min(h, per));
        for (std::thread& th : ths) th.join();
        return 0;
    }
};

static ncnn::Layer* Warp_layer_creator(void*) { return new Warp_layer; }

static void register_all(ncnn::Net& net) {
    net.register_custom_layer(ncnn::layer_to_index("Reshape"), Reshape_fix_creator);
    net.register_custom_layer(ncnn::layer_to_index("Slice"), Slice_fix_creator);
    net.register_custom_layer(ncnn::layer_to_index("GridSample"), GridSample_fix_creator);
    net.register_custom_layer("rife.Warp", Warp_layer_creator);
}

static int run_net(bool warp_gpu, bool use_vulkan,
                   const char* param, const char* bin,
                   int W, int H, int runs, const char* label) {
    g_warp_gpu = warp_gpu;
    int threads; bool fp16;
    parse_opt(getenv("VFI_OPT"), threads, fp16);
    printf("opt: thr=%d fp16=%d vulkan=%d reshmode=%d\n",
           threads, fp16 ? 1 : 0, use_vulkan ? 1 : 0, g_reshmode);
    ncnn::Net net;
    net.opt.use_vulkan_compute = use_vulkan;
    if (use_vulkan) net.set_vulkan_device(0);
    net.opt.use_fp16_packed = fp16;
    net.opt.use_fp16_storage = fp16;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = threads;
    register_all(net);
    g_phase = "load";
    fprintf(stderr, "LOAD: param %s\n", param); fflush(stderr);
    if (net.load_param(param) != 0) {
        fprintf(stderr, "ERR: load_param failed\n");
        dump_param_head(param);
        fflush(NULL); return 2;
    }
    fprintf(stderr, "LOAD: param ok layers=%d\n", (int)net.layers().size()); fflush(stderr);
    fprintf(stderr, "LOAD: model %s\n", bin); fflush(stderr);
    if (net.load_model(bin) != 0) { fprintf(stderr, "ERR: load_model failed\n"); fflush(NULL); return 2; }
    fprintf(stderr, "LOAD: model ok\n"); fflush(stderr);
    ncnn::Mat a(W, H, 3), b(W, H, 3);
    for (int c = 0; c < 3; c++) {
        float* pa = a.channel(c); float* pb = b.channel(c);
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            pa[y*W+x] = fmodf((x + c * 37.0f) / 255.0f, 1.0f);
            pb[y*W+x] = fmodf((x + y + c * 37.0f) / 255.0f, 1.0f);
        }
    }
    ncnn::Mat ts(1, 1, 1); ts.channel(0)[0] = 0.5f;
    const std::vector<int>& ins = net.input_indexes();
    printf("inputs=%d\n", (int)ins.size());
    auto make_input = [&](int ci) {
        ncnn::Mat in(W, H, ci);
        for (int c = 0; c < ci; c++) {
            float* p = in.channel(c);
            if (c < 3) memcpy(p, a.channel(c), (size_t)W * H * 4);
            else if (c < 6) memcpy(p, b.channel(c - 3), (size_t)W * H * 4);
            else if (c == 6) { for (int i = 0; i < W * H; i++) p[i] = 0.5f; }
            else memset(p, 0, (size_t)W * H * 4);
        }
        return in;
    };
    g_phase = "probe";
    int c_in = 0;
    if ((int)ins.size() == 1) {
        const int cands[4] = {7, 6, 11, 4};
        for (int k = 0; k < 4; k++) {
            pid_t pid = fork();
            if (pid == 0) {
                ncnn::Extractor ex2 = net.create_extractor();
                ex2.input(ins[0], make_input(cands[k]));
                ncnn::Mat o;
                int r = ex2.extract(net.output_indexes()[0], o);
                fflush(NULL);
                _exit(r == 0 ? 0 : 1);
            }
            int st = 0; waitpid(pid, &st, 0);
            if (WIFEXITED(st) && WEXITSTATUS(st) == 0) { c_in = cands[k]; break; }
        }
        if (c_in == 0) { fprintf(stderr, "ERR: no feed contract worked\n"); fflush(NULL); return 3; }
        printf("single-input contract c_in=%d\n", c_in);
    }
    auto feed = [&](ncnn::Extractor& ex) {
        if ((int)ins.size() >= 2) {
            ex.input(ins[0], a); ex.input(ins[1], b);
            if ((int)ins.size() >= 3) ex.input(ins[2], ts);
        } else {
            ex.input(ins[0], make_input(c_in));
        }
    };
    g_phase = "extract";
    ncnn::Mat out;
    {
        ncnn::Extractor ex = net.create_extractor();
        feed(ex);
        int ret = ex.extract(net.output_indexes()[0], out);
        if (ret != 0) { fprintf(stderr, "EXTRACT_RET=%d (%s)\n", ret, label); fflush(NULL); return 3; }
    }
    double mn = 1e9, mx = -1e9, sum = 0;
    int n = out.w * out.h * out.c;
    const float* p = out;
    for (int i = 0; i < n; i++) { double v = p[i]; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
    printf("out w=%d h=%d c=%d min=%.3f max=%.3f mean=%.3f (%s)\n",
           out.w, out.h, out.c, mn, mx, sum / n, label);
    if (!(sum == sum)) fprintf(stderr, "WARN: NaN in output (%s)\n", label);
    g_phase = "timing";
    g_frames = runs;
    double t0 = now_ms();
    for (int r = 0; r < runs; r++) {
        g_frame = r + 1;
        printf("FRAME %d/%d\n", r + 1, runs);
        fflush(NULL);
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

static int bisect(const char* param, const char* bin, int W, int H) {
    int threads; bool fp16;
    parse_opt(getenv("VFI_OPT"), threads, fp16);
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.set_vulkan_device(0);
    net.opt.use_fp16_packed = fp16;
    net.opt.use_fp16_storage = fp16;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_packing_layout = false;
    net.opt.num_threads = threads;
    register_all(net);
    g_phase = "load";
    fprintf(stderr, "LOAD: param %s\n", param); fflush(stderr);
    if (net.load_param(param) != 0) {
        fprintf(stderr, "ERR: load_param failed in bisect\n");
        dump_param_head(param);
        fflush(NULL); return 2;
    }
    fprintf(stderr, "LOAD: param ok layers=%d\n", (int)net.layers().size()); fflush(stderr);
    fprintf(stderr, "LOAD: model %s\n", bin); fflush(stderr);
    if (net.load_model(bin) != 0) { fprintf(stderr, "ERR: load_model failed in bisect\n"); fflush(NULL); return 2; }
    fprintf(stderr, "LOAD: model ok\n"); fflush(stderr);
    const std::vector<ncnn::Layer*>& lrs = net.layers();
    const int n = (int)lrs.size();
    printf("bisect: layers=%d (single-pass, vulkan)\n", n);
    fflush(NULL);
    ncnn::Mat in(W, H, 7);
    for (int c = 0; c < 7; c++) {
        float* p = in.channel(c);
        if (c < 3) {
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
                p[y*W+x] = fmodf((x + c * 37.0f) / 255.0f, 1.0f);
        } else if (c < 6) {
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
                p[y*W+x] = fmodf((x + y + (c - 3) * 37.0f) / 255.0f, 1.0f);
        } else {
            for (int k = 0; k < W * H; k++) p[k] = 0.5f;
        }
    }
    ncnn::Extractor ex = net.create_extractor();
    ex.input(net.input_indexes()[0], in);
    const double bt0 = now_ms();
    const int bsec = getenv("VFI_BISECT_SEC") ? atoi(getenv("VFI_BISECT_SEC")) : 240;
    g_phase = "bisect";
    g_frames = n;
    for (int i = 0; i < n; i++) {
        if (now_ms() - bt0 > bsec * 1000.0) {
            printf("BISECT_INCOMPLETE stop at %d/%d (budget)\n", i, n);
            fflush(NULL);
            return 3;
        }
        if (lrs[i]->tops.empty()) continue;
        g_frame = i + 1;
        ncnn::Mat o;
        int r = ex.extract(lrs[i]->tops[0], o);
        printf("BISECT %d/%d %s %s rc=%d w=%d h=%d c=%d\n",
               i, n, lrs[i]->type.c_str(), lrs[i]->name.c_str(), r, o.w, o.h, o.c);
        fflush(NULL);
        if (r != 0) {
            printf("BISECT_FAIL at %d %s %s rc=%d\n",
                   i, lrs[i]->type.c_str(), lrs[i]->name.c_str(), r);
            fflush(NULL);
            return 3;
        }
    }
    printf("ALL LAYERS OK\n");
    return 0;
}

int main(int argc, char** argv) {
    start_heartbeat();
    const char* param = argc > 1 ? argv[1] : "flownet.param";
    const char* bin   = argc > 2 ? argv[2] : "flownet.bin";
    int W = argc > 3 ? atoi(argv[3]) : 512;
    int H = argc > 4 ? atoi(argv[4]) : 384;
    int runs = argc > 5 ? atoi(argv[5]) : 5;
    const char* mode = argc > 6 ? argv[6] : "gpu";
    if (strcmp(mode, "bisect") == 0) return bisect(param, bin, W, H);
    if (strcmp(mode, "cpu") == 0)
        return run_net(false, false, param, bin, W, H, runs, "cpu") == 0 ? 0 : 3;
    if (strcmp(mode, "gpuw") == 0) {
        if (ncnn::get_gpu_count() == 0) { fprintf(stderr, "ERR: no vulkan gpu\n"); fflush(NULL); _exit(1); }
        int rc = run_net(true, true, param, bin, W, H, runs, "gpuw");
        if (rc != 0) { fprintf(stderr, "GPUW_FAILED rc=%d\n", rc); fflush(NULL); return 3; }
        return 0;
    }
    if (ncnn::get_gpu_count() == 0) { fprintf(stderr, "ERR: no vulkan gpu\n"); fflush(NULL); _exit(1); }
    return run_net(false, true, param, bin, W, H, runs, "gpu") == 0 ? 0 : 3;
}
