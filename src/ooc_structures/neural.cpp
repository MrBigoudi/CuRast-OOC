#include "neural.h"

#include <sstream>

namespace F = torch::nn::functional;


// =============================================================================
// Helpers
// =============================================================================

// Bilinearly resize source to target's spatial size if they differ
// (same as resize_to_match in the Python script).
static torch::Tensor resize_to_match(const torch::Tensor& source, const torch::Tensor& target)
{
    if (source.size(-2) == target.size(-2) && source.size(-1) == target.size(-1)) {
        return source;
    }
    return F::interpolate(
        source,
        F::InterpolateFuncOptions()
            .size(std::vector<int64_t>{ target.size(-2), target.size(-1) })
            .mode(torch::kBilinear)
            .align_corners(false)
    );
}

// Unconditional bilinear upsample to reference size (decoder path).
static torch::Tensor upsample_to(const torch::Tensor& x, const torch::Tensor& reference)
{
    return F::interpolate(
        x,
        F::InterpolateFuncOptions()
            .size(std::vector<int64_t>{ reference.size(-2), reference.size(-1) })
            .mode(torch::kBilinear)
            .align_corners(false)
    );
}


// =============================================================================
// GatedConvolution
// =============================================================================

GatedConvolution::GatedConvolution(
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t kernel_size,
    uint32_t padding,
    float    negative_slope)
    : out_channels(out_channels)
{
    conv = register_module(
        "conv",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels * 2, kernel_size)
                .padding(padding)
        )
    );

    activation = register_module(
        "activation",
        torch::nn::LeakyReLU(
            torch::nn::LeakyReLUOptions()
                .negative_slope(negative_slope)
                .inplace(false)
        )
    );

    gate_activation = register_module("gate_activation", torch::nn::Sigmoid());
}

torch::Tensor GatedConvolution::forward(torch::Tensor x)
{
    x = conv->forward(x);
    auto parts   = x.split(out_channels, 1);
    auto feature = parts[0];
    auto gate    = parts[1];
    return activation->forward(feature) * gate_activation->forward(gate);
}


// =============================================================================
// GatedDoubleConv
// =============================================================================

GatedDoubleConv::GatedDoubleConv(
    uint32_t in_channels,
    uint32_t out_channels,
    float    negative_slope)
{
    conv1 = register_module(
        "conv1",
        std::make_shared<GatedConvolution>(in_channels, out_channels, 3, 1, negative_slope)
    );
    conv2 = register_module(
        "conv2",
        std::make_shared<GatedConvolution>(out_channels, out_channels, 3, 1, negative_slope)
    );
}

torch::Tensor GatedDoubleConv::forward(torch::Tensor x)
{
    x = conv1->forward(x);
    x = conv2->forward(x);
    return x;
}


// =============================================================================
// PyramidUNet
// =============================================================================

PyramidUNet::PyramidUNet(const PyramidUNetConfig& cfg)
    : config(cfg),
      channels{
          cfg.base_channels,
          cfg.base_channels * 2,
          cfg.base_channels * 4,
          cfg.base_channels * 8
      }
{
    const uint32_t lc    = config.level_channels();
    const float    slope = config.negative_slope;

    pool = register_module("pool", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2)));

    // Encoder (names match Python: enc0..enc3)
    encoders[0] = register_module("enc0", std::make_shared<GatedDoubleConv>(lc,               channels[0], slope));
    encoders[1] = register_module("enc1", std::make_shared<GatedDoubleConv>(channels[0] + lc, channels[1], slope));
    encoders[2] = register_module("enc2", std::make_shared<GatedDoubleConv>(channels[1] + lc, channels[2], slope));
    encoders[3] = register_module("enc3", std::make_shared<GatedDoubleConv>(channels[2] + lc, channels[3], slope));

    // Decoder (names match Python: up0..up2)
    ups[2] = register_module("up2", std::make_shared<GatedDoubleConv>(channels[3] + channels[2], channels[2], slope));
    ups[1] = register_module("up1", std::make_shared<GatedDoubleConv>(channels[2] + channels[1], channels[1], slope));
    ups[0] = register_module("up0", std::make_shared<GatedDoubleConv>(channels[1] + channels[0], channels[0], slope));

    output_conv = register_module(
        "output_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels[0], config.output_channels, 1))
    );

    // Same init as Python for the linear head (only matters when training
    // from scratch; overwritten when weights are loaded).
    if (config.output_activation == OutputActivation::Linear) {
        torch::NoGradGuard no_grad;
        output_conv->bias.fill_(0.5);
    }
}

torch::Tensor PyramidUNet::forward(const std::vector<torch::Tensor>& pyramid)
{
    TORCH_CHECK(pyramid.size() == NB_LEVELS,
                "PyramidUNet expects ", NB_LEVELS, " pyramid levels, got ", pyramid.size());

    const auto& p0 = pyramid[0];
    const auto& p1 = pyramid[1];
    const auto& p2 = pyramid[2];
    const auto& p3 = pyramid[3];

    TORCH_CHECK(p0.size(1) == config.level_channels(),
                "PyramidUNet expects ", config.level_channels(),
                " channels per level, got ", p0.size(1));

    // Encoder
    auto skip0 = encoders[0]->forward(p0);

    auto x = resize_to_match(pool->forward(skip0), p1);
    auto skip1 = encoders[1]->forward(torch::cat({x, p1}, 1));

    x = resize_to_match(pool->forward(skip1), p2);
    auto skip2 = encoders[2]->forward(torch::cat({x, p2}, 1));

    x = resize_to_match(pool->forward(skip2), p3);
    auto bottleneck = encoders[3]->forward(torch::cat({x, p3}, 1));

    // Decoder
    x = ups[2]->forward(torch::cat({upsample_to(bottleneck, skip2), skip2}, 1));
    x = ups[1]->forward(torch::cat({upsample_to(x, skip1), skip1}, 1));
    x = ups[0]->forward(torch::cat({upsample_to(x, skip0), skip0}, 1));

    // Output
    x = output_conv->forward(x);
    if (config.output_activation == OutputActivation::Sigmoid) {
        x = torch::sigmoid(x);
    }
    return x;
}


// =============================================================================
// Weight transfer TorchScript -> native module
// =============================================================================

static void copy_weights_from_jit(const torch::jit::script::Module& src, torch::nn::Module& dst)
{
    torch::NoGradGuard no_grad;

    auto   dst_params = dst.named_parameters(/*recurse=*/true);
    size_t copied     = 0;

    for (const auto& p : src.named_parameters(/*recurse=*/true)) {
        torch::Tensor* t = dst_params.find(p.name);
        if (t == nullptr) {
            throw std::runtime_error(
                "Trained parameter '" + p.name + "' has no match in native PyramidUNet");
        }
        if (t->sizes() != p.value.sizes()) {
            std::ostringstream ss;
            ss << "Shape mismatch for '" << p.name << "': native " << t->sizes()
               << " vs trained " << p.value.sizes()
               << " -> check NeuralNet::config against the Python settings";
            throw std::runtime_error(ss.str());
        }
        t->copy_(p.value);
        ++copied;
    }

    if (copied != dst_params.size()) {
        std::ostringstream ss;
        ss << "Only " << copied << " of " << dst_params.size()
           << " native parameters were found in the trained model";
        throw std::runtime_error(ss.str());
    }

    println("Copied {} parameter tensors from TorchScript model", copied);
}

// Runs both models on the same random pyramid and checks they agree.
static void verify_against_jit(torch::jit::script::Module& jit, PyramidUNet& native)
{
    torch::NoGradGuard no_grad;

    const int64_t W = 480, H = 270;
    const int64_t C = native.config.level_channels();

    std::vector<torch::Tensor> pyramid;
    for (uint32_t level = 0; level < PyramidUNet::NB_LEVELS; level++) {
        pyramid.push_back(torch::rand(
            {1, C, H >> level, W >> level},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)));
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(pyramid);

    auto ref  = jit.forward(inputs).toTensor();
    auto out  = native.forward(pyramid);
    float diff = (ref - out).abs().max().item<float>();

    println("Native vs TorchScript max abs diff: {}", diff);
    if (diff > 1e-3f) {
        throw std::runtime_error("Native PyramidUNet does not match the trained model");
    }
}


// =============================================================================
// NeuralNet
// =============================================================================

void NeuralNet::load(const std::string& model_path)
{
    static_assert(CRenderingSettings::NB_PYRAMID_LEVELS == PyramidUNet::NB_LEVELS,
                  "PyramidUNet is built for exactly 4 pyramid levels");

    println("File {} exists and has size: {}", model_path, std::filesystem::file_size(model_path));

    jit_model = torch::jit::load(model_path, torch::kCUDA);
    jit_model.eval();

    println("Building native PyramidUNet: level_channels={} base_channels={} output={}",
            config.level_channels(), config.base_channels,
            config.output_activation == OutputActivation::Linear ? "linear" : "sigmoid");

    net = std::make_shared<PyramidUNet>(config);
    copy_weights_from_jit(jit_model, *net);
    net->to(torch::kCUDA);
    net->eval();

    verify_against_jit(jit_model, *net);
}


void NeuralNet::infer(RenderTarget& target)
{
    uint32_t NB_LEVELS = CRenderingSettings::NB_PYRAMID_LEVELS;
    uint32_t W0 = target.width;
    uint32_t H0 = target.height;

    // Channels per pyramid level as expected by the trained model
    // (4 for the current checkpoint: RGB + depth).
    const uint32_t C = config.level_channels();

    // ---------------------------------------------------------------
    // 1. Per-level resolve into uint32 RGBA buffers
    // ---------------------------------------------------------------
    static CUdeviceptr d_resolved[CRenderingSettings::NB_PYRAMID_LEVELS] = {};
    static uint64_t    d_resolved_capacity[CRenderingSettings::NB_PYRAMID_LEVELS] = {};

    uint64_t level_cb_offsets_host[CRenderingSettings::NB_PYRAMID_LEVELS];
    uint64_t level_fb_offsets_host[CRenderingSettings::NB_PYRAMID_LEVELS];
    uint64_t level_tensor_offsets_host[CRenderingSettings::NB_PYRAMID_LEVELS];
    {
        uint64_t cb_off     = 0;
        uint64_t tensor_off = 0;
        for (uint32_t level = 0; level < NB_LEVELS; level++) {
            level_cb_offsets_host[level]     = cb_off;
            level_fb_offsets_host[level]     = cb_off;
            level_tensor_offsets_host[level] = tensor_off;
            uint64_t pixels = uint64_t(W0 >> level) * (H0 >> level);
            cb_off     += pixels;
            tensor_off += uint64_t(C) * pixels;
        }
    }

    bool enableEDL = CuRastSettings::enableEDL;
    uint32_t backgroundColor = 0;
    {
        uint8_t* bg = (uint8_t*)&backgroundColor;
        bg[0] = uint8_t(clamp(CuRastSettings::background.x * 256.0f, 0.0f, 255.0f));
        bg[1] = uint8_t(clamp(CuRastSettings::background.y * 256.0f, 0.0f, 255.0f));
        bg[2] = uint8_t(clamp(CuRastSettings::background.z * 256.0f, 0.0f, 255.0f));
        bg[3] = 255;
    }

    for (uint32_t level = 0; level < NB_LEVELS; level++) {
        uint32_t w = W0 >> level;
        uint32_t h = H0 >> level;
        uint64_t pixels = uint64_t(w) * h;

        if (pixels > d_resolved_capacity[level]) {
            if (d_resolved[level]) cuMemFree(d_resolved[level]);
            cuMemAlloc(&d_resolved[level], pixels * sizeof(uint32_t));
            d_resolved_capacity[level] = pixels;
        }

        CRenderTarget level_target = {};
        level_target.colorbuffers[0] = target.colorbuffer + level_cb_offsets_host[level];
        level_target.framebuffers[0] = target.framebuffer + level_cb_offsets_host[level];
        level_target.width  = w;
        level_target.height = h;

        int levelW = (int)w;
        int levelH = (int)h;
        auto d_resolved_level = d_resolved[level];

        // saveScreenshot used the window size at level 0 and the level size at 1..3
        int windowW = (level == 0) ? int(W0 / CuRastSettings::supersamplingFactor) : int(w);
        int windowH = (level == 0) ? int(H0 / CuRastSettings::supersamplingFactor) : int(h);

        void* args[] = { &level_target, &d_resolved_level, &enableEDL, &windowW, &windowH, &backgroundColor };
        GpuVersion::prog->launch2D("kernel_resolve_colorbuffer_to_screenshot", args, w, h);
    }

    // ---------------------------------------------------------------
    // 2. Input / output buffers
    // ---------------------------------------------------------------
    uint64_t total_floats = 0;
    for (uint32_t level = 0; level < NB_LEVELS; level++) {
        total_floats += uint64_t(C) * (W0 >> level) * (H0 >> level);
    }
    uint64_t total_float_bytes = total_floats * sizeof(float);

    static float*   d_input          = nullptr;
    static uint64_t d_input_capacity = 0;
    if (total_float_bytes > d_input_capacity) {
        if (d_input) cudaFree(d_input);
        cudaMalloc(&d_input, total_float_bytes);
        d_input_capacity = total_float_bytes;
    }

    uint64_t output_float_bytes = uint64_t(config.output_channels) * W0 * H0 * sizeof(float);
    static float*   d_output          = nullptr;
    static uint64_t d_output_capacity = 0;
    if (output_float_bytes > d_output_capacity) {
        if (d_output) cudaFree(d_output);
        cudaMalloc(&d_output, output_float_bytes);
        d_output_capacity = output_float_bytes;
    }

    // ---------------------------------------------------------------
    // 3. Upload offset arrays
    // ---------------------------------------------------------------
    static CUdeviceptr d_level_fb_offsets     = 0;
    static CUdeviceptr d_level_tensor_offsets = 0;
    if (d_level_fb_offsets == 0) {
        cuMemAlloc(&d_level_fb_offsets,     NB_LEVELS * sizeof(uint64_t));
        cuMemAlloc(&d_level_tensor_offsets, NB_LEVELS * sizeof(uint64_t));
    }
    cuMemcpyHtoDAsync(d_level_fb_offsets,     level_fb_offsets_host,     NB_LEVELS * sizeof(uint64_t), 0);
    cuMemcpyHtoDAsync(d_level_tensor_offsets, level_tensor_offsets_host, NB_LEVELS * sizeof(uint64_t), 0);

    // ---------------------------------------------------------------
    // 4. Unpack resolved RGBA (+ depth) into float tensor
    //    NOTE: the kernel must write exactly C channels per level,
    //    planar, in the order RGB, [LOD], [depth], i.e. channel c of
    //    pixel (x,y) at  tensor_offset[level] + c*w*h + y*w + x.
    // ---------------------------------------------------------------
    static uint32_t flip_y = 1;      // dataset PNGs were written with stbi vertical flip
    {
        uint32_t block_size = 256;
        uint32_t grid_size  = (uint32_t(W0) * H0 + block_size - 1) / block_size;
        OptionalLaunchSettings launch_settings = { .gridsize = grid_size, .blocksize = block_size };

        auto d_r0 = d_resolved[0];
        auto d_r1 = d_resolved[1];
        auto d_r2 = d_resolved[2];
        auto d_r3 = d_resolved[3];

        uint32_t nb_channels   = C;
        void* unpack_args[] = {
            &d_r0, &d_r1, &d_r2, &d_r3,
            &target.colorbuffer,
            &d_input,
            &W0, &H0,
            &NB_LEVELS,
            &nb_channels,
            &flip_y,
            &d_level_fb_offsets,          // same values as the colorbuffer offsets
            &d_level_tensor_offsets
        };
        GpuVersion::prog->launch("kernel_unpack_resolved_pyramid_to_tensor", unpack_args, launch_settings);
    }

    // ---------------------------------------------------------------
    // 5. Zero-copy per-level tensor views into d_input
    // ---------------------------------------------------------------
    std::vector<torch::Tensor> pyramid;
    pyramid.reserve(NB_LEVELS);
    {
        uint64_t float_offset = 0;
        for (uint32_t level = 0; level < NB_LEVELS; level++) {
            int64_t w = W0 >> level;
            int64_t h = H0 >> level;
            pyramid.push_back(torch::from_blob(
                d_input + float_offset,
                {1, (int64_t)C, h, w},
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)));
            float_offset += uint64_t(C) * w * h;
        }
    }

    // ---------------------------------------------------------------
    // 6. Inference
    // ---------------------------------------------------------------
    torch::NoGradGuard no_grad;

    torch::Tensor output;
    if (use_native) {
        output = net->forward(pyramid);
    } else {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(pyramid);
        output = jit_model.forward(inputs).toTensor();
    }

    // Linear head is unbounded: clamp like eval_model() does in Python
    // (no-op for sigmoid).
    torch::Tensor output_squeezed = output.clamp(0.0, 1.0).squeeze(0).contiguous();  // [3, H0, W0]

    cudaMemcpy(d_output, output_squeezed.data_ptr<float>(), output_float_bytes, cudaMemcpyDeviceToDevice);

    // ---------------------------------------------------------------
    // 7. Pack model output into colorbuffer[level 0]
    // ---------------------------------------------------------------
    {
        uint64_t num_pixels = uint64_t(W0) * H0;
        uint32_t block_size = 256;
        uint32_t grid_size  = (num_pixels + block_size - 1) / block_size;
        OptionalLaunchSettings launch_settings = { .gridsize = grid_size, .blocksize = block_size };

        void* pack_args[] = { &d_output, &target.colorbuffer, &W0, &H0, &flip_y };
        GpuVersion::prog->launch("kernel_pack_tensor_to_colorbuffer", pack_args, launch_settings);
    }
}