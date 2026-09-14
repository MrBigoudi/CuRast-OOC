#include "neural.h"

#include <torch/script.h>

GatedConvolution::GatedConvolution(
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t kernel_size,
    uint32_t padding)
    : out_channels(out_channels)
{
    conv = register_module(
        "conv",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(
                in_channels,
                out_channels * 2,
                kernel_size
            ).padding(padding)
        )
    );

    activation = register_module(
        "activation",
        torch::nn::LeakyReLU(
            torch::nn::LeakyReLUOptions()
                .inplace(false)
                .negative_slope(0.2)
        )
    );

    gate_activation = register_module(
        "gate_activation",
        torch::nn::Sigmoid()
    );
}


torch::Tensor GatedConvolution::forward(torch::Tensor x)
{
    x = conv->forward(x);

    auto output = x.split(out_channels, 1);

    auto feature = output[0];
    auto gate = output[1];

    return activation->forward(feature) *
           gate_activation->forward(gate);
}


GatedDoubleConv::GatedDoubleConv(
    uint32_t in_channels,
    uint32_t out_channels)
{
    conv1 = register_module(
        "conv1",
        std::make_shared<GatedConvolution>(
            in_channels,
            out_channels,
            3,
            1
        )
    );

    conv2 = register_module(
        "conv2",
        std::make_shared<GatedConvolution>(
            out_channels,
            out_channels,
            3,
            1
        )
    );
}


torch::Tensor GatedDoubleConv::forward(torch::Tensor x)
{
    x = conv1->forward(x);
    x = conv2->forward(x);
    return x;
}


PyramidUNet::PyramidUNet(
    uint32_t level_channels,
    uint32_t base_channels,
    uint32_t output_channels)
    : channels{
        base_channels,
        base_channels * 2,
        base_channels * 4,
        base_channels * 8
    }
{
    pool = register_module(
        "pool",
        torch::nn::AvgPool2d(
            torch::nn::AvgPool2dOptions(2)
        )
    );

    encoders[0] = register_module(
        "enc0",
        std::make_shared<GatedDoubleConv>(
            level_channels,
            channels[0]
        )
    );

    encoders[1] = register_module(
        "enc1",
        std::make_shared<GatedDoubleConv>(
            channels[0] + level_channels,
            channels[1]
        )
    );

    encoders[2] = register_module(
        "enc2",
        std::make_shared<GatedDoubleConv>(
            channels[1] + level_channels,
            channels[2]
        )
    );

    encoders[3] = register_module(
        "enc3",
        std::make_shared<GatedDoubleConv>(
            channels[2] + level_channels,
            channels[3]
        )
    );

    ups[0] = register_module(
        "up0",
        std::make_shared<GatedDoubleConv>(
            channels[1] + channels[0],
            channels[0]
        )
    );

    ups[1] = register_module(
        "up1",
        std::make_shared<GatedDoubleConv>(
            channels[2] + channels[1],
            channels[1]
        )
    );

    ups[2] = register_module(
        "up2",
        std::make_shared<GatedDoubleConv>(
            channels[3] + channels[2],
            channels[2]
        )
    );

    output_conv = register_module(
        "output_conv",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(
                channels[0],
                output_channels,
                1
            )
        )
    );

    output_activation = register_module(
        "output_activation",
        torch::nn::Sigmoid()
    );
}


torch::Tensor PyramidUNet::forward(const std::vector<torch::Tensor>& pyramid){
    if (pyramid.size() != 4) {
        throw std::runtime_error("PyramidUNet expects 4 pyramid levels");
    }

    const auto& p0 = pyramid[0];
    const auto& p1 = pyramid[1];
    const auto& p2 = pyramid[2];
    const auto& p3 = pyramid[3];

    // Encoder level 0
    auto skip0 = encoders[0]->forward(p0);

    // Encoder level 1
    torch::Tensor x = pool->forward(skip0);
    NeuralNet::resize_to_match(x, p1);
    x = torch::cat({x, p1}, 1);

    auto skip1 = encoders[1]->forward(x);

    // Encoder level 2
    x = pool->forward(skip1);
    NeuralNet::resize_to_match(x, p2);
    x = torch::cat({x, p2}, 1);

    auto skip2 = encoders[2]->forward(x);

    // Encoder level 3 / bottleneck
    x = pool->forward(skip2);
    NeuralNet::resize_to_match(x, p3);
    x = torch::cat({x, p3}, 1);

    auto bottleneck = encoders[3]->forward(x);

    // Decoder: up2
    x = torch::nn::functional::interpolate(
        bottleneck,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::optional<std::vector<int64_t>>({skip2.size(-2), skip2.size(-1)}))
            .mode(torch::kBilinear)
            .align_corners(false)
    );

    x = torch::cat({x, skip2}, 1);
    x = ups[2]->forward(x);

    // Decoder: up1
    x = torch::nn::functional::interpolate(
        x,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::optional<std::vector<int64_t>>({skip1.size(-2), skip1.size(-1)}))
            .mode(torch::kBilinear)
            .align_corners(false)
    );

    x = torch::cat({x, skip1}, 1);
    x = ups[1]->forward(x);

    // Decoder: up0
    x = torch::nn::functional::interpolate(
        x,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::optional<std::vector<int64_t>>({skip0.size(-2), skip0.size(-1)}))
            .mode(torch::kBilinear)
            .align_corners(false)
    );

    x = torch::cat({x, skip0}, 1);
    x = ups[0]->forward(x);

    // Output
    x = output_conv->forward(x);
    x = output_activation->forward(x);

    return x;
}



void NeuralNet::resize_to_match(torch::Tensor &source, const torch::Tensor &target){
    if(source.size(-2) != target.size(-2) || source.size(-1) != target.size(-1)){
        auto options = torch::nn::functional::InterpolateFuncOptions()
            .mode(torch::kBilinear)
            .align_corners(false)
            .size(std::optional<std::vector<int64_t>>({target.size(-2), target.size(-1)}))
        ;
        source = torch::nn::functional::interpolate(source, options);
    }
}


void NeuralNet::load(const std::string& model_path){
    model = torch::jit::load(model_path);
    model.dump(false, false, false);
    model.eval();
}


void NeuralNet::infer(RenderTarget& target) {
    uint32_t NB_LEVELS = CRenderingSettings::NB_PYRAMID_LEVELS;
    uint32_t W0 = target.width;
    uint32_t H0 = target.height;

    // ---------------------------------------------------------------
    // 1. Per-level resolve: run kernel_resolve_colorbuffer_to_screenshot
    //    on each pyramid level into a uint32 RGBA buffer.
    //    This guarantees byte order matches training (PIL RGB).
    // ---------------------------------------------------------------
    static CUdeviceptr d_resolved[CRenderingSettings::NB_PYRAMID_LEVELS] = {};
    static uint64_t    d_resolved_capacity[CRenderingSettings::NB_PYRAMID_LEVELS] = {};

    // Precompute colorbuffer/framebuffer offsets (same layout as renderOctree)
    uint64_t level_cb_offsets_host[CRenderingSettings::NB_PYRAMID_LEVELS];
    uint64_t level_fb_offsets_host[CRenderingSettings::NB_PYRAMID_LEVELS];
    uint64_t level_tensor_offsets_host[CRenderingSettings::NB_PYRAMID_LEVELS];
    {
        uint64_t cb_off     = 0;
        uint64_t tensor_off = 0;
        for(uint32_t level = 0; level < NB_LEVELS; level++){
            level_cb_offsets_host[level]     = cb_off;
            level_fb_offsets_host[level]     = cb_off;  // framebuffer has same layout
            level_tensor_offsets_host[level] = tensor_off;
            uint32_t w = W0 >> level;
            uint32_t h = H0 >> level;
            uint64_t pixels = uint64_t(w) * h;
            cb_off     += pixels;
            tensor_off += 5 * pixels;
        }
    }

    // Resolve each level
    bool   noEDL  = false;
    bool   noSSAO = false;
    uint32_t backgroundColor = 0;
    {
        uint8_t* bg = (uint8_t*)&backgroundColor;
        bg[0] = uint8_t(clamp(CuRastSettings::background.x * 256.0f, 0.0f, 255.0f));
        bg[1] = uint8_t(clamp(CuRastSettings::background.y * 256.0f, 0.0f, 255.0f));
        bg[2] = uint8_t(clamp(CuRastSettings::background.z * 256.0f, 0.0f, 255.0f));
        bg[3] = 255;
    }

    for(uint32_t level = 0; level < NB_LEVELS; level++){
        uint32_t w = W0 >> level;
        uint32_t h = H0 >> level;
        uint64_t pixels = uint64_t(w) * h;

        // Grow buffer if needed
        if(pixels > d_resolved_capacity[level]){
            if(d_resolved[level]) cuMemFree(d_resolved[level]);
            cuMemAlloc(&d_resolved[level], pixels * sizeof(uint32_t));
            d_resolved_capacity[level] = pixels;
        }

        // Build a RenderTarget slice for this level
        CRenderTarget level_target = {};
        level_target.colorbuffers[0]  = target.colorbuffer + level_cb_offsets_host[level];
        level_target.framebuffers[0]  = target.framebuffer + level_cb_offsets_host[level];
        level_target.width        = w;
        level_target.height       = h;

        int levelW = (int)w;
        int levelH = (int)h;
        auto d_resolved_level = d_resolved[level];

        void* args[] = {
            &level_target,
            &d_resolved_level,
            &noEDL,
            &levelW,
            &levelH,
            &backgroundColor
        };
        GpuVersion::prog->launch2D("kernel_resolve_colorbuffer_to_screenshot",
            args, w, h);
    }

    // ---------------------------------------------------------------
    // 2. Allocate runtime-visible float buffer for the tensor input
    // ---------------------------------------------------------------
    uint64_t total_floats = 0;
    for(uint32_t level = 0; level < NB_LEVELS; level++){
        total_floats += 5 * uint64_t(W0 >> level) * uint64_t(H0 >> level);
    }
    uint64_t total_float_bytes = total_floats * sizeof(float);

    static float*    d_input          = nullptr;
    static uint64_t  d_input_capacity = 0;
    if(total_float_bytes > d_input_capacity){
        if(d_input) cudaFree(d_input);
        cudaMalloc(&d_input, total_float_bytes);
        d_input_capacity = total_float_bytes;
    }

    uint64_t output_float_bytes = 3 * uint64_t(W0) * H0 * sizeof(float);
    static float*    d_output          = nullptr;
    static uint64_t  d_output_capacity = 0;
    if(output_float_bytes > d_output_capacity){
        if(d_output) cudaFree(d_output);
        cudaMalloc(&d_output, output_float_bytes);
        d_output_capacity = output_float_bytes;
    }

    // ---------------------------------------------------------------
    // 3. Upload offset arrays to GPU
    // ---------------------------------------------------------------
    static CUdeviceptr d_level_fb_offsets     = 0;
    static CUdeviceptr d_level_tensor_offsets = 0;
    if(d_level_fb_offsets == 0){
        cuMemAlloc(&d_level_fb_offsets,     NB_LEVELS * sizeof(uint64_t));
        cuMemAlloc(&d_level_tensor_offsets, NB_LEVELS * sizeof(uint64_t));
    }
    cuMemcpyHtoDAsync(d_level_fb_offsets,
                      level_fb_offsets_host,
                      NB_LEVELS * sizeof(uint64_t), 0);
    cuMemcpyHtoDAsync(d_level_tensor_offsets,
                      level_tensor_offsets_host,
                      NB_LEVELS * sizeof(uint64_t), 0);

    // ---------------------------------------------------------------
    // 4. Unpack resolved RGBA + LOD into float tensor
    // ---------------------------------------------------------------
    {
        uint32_t block_size = 256;
        uint32_t grid_size  = (uint32_t(W0) * H0 + block_size - 1) / block_size;
        OptionalLaunchSettings launch_settings = {
            .gridsize  = grid_size,
            .blocksize = block_size
        };

        auto d_r0 = d_resolved[0];
        auto d_r1 = d_resolved[1];
        auto d_r2 = d_resolved[2];
        auto d_r3 = d_resolved[3];

        void* args[] = {
            &d_r0, &d_r1, &d_r2, &d_r3,
            &target.framebuffer,
            &d_input,
            &W0, &H0,
            &NB_LEVELS,
            &d_level_fb_offsets,
            &d_level_tensor_offsets
        };
        GpuVersion::prog->launch("kernel_unpack_resolved_pyramid_to_tensor",
            args, launch_settings);
    }

    // ---------------------------------------------------------------
    // 5. Build per-level tensors (zero-copy views into d_input)
    // ---------------------------------------------------------------
    std::vector<torch::Tensor> pyramid;
    pyramid.reserve(NB_LEVELS);
    {
        uint64_t float_offset = 0;
        for(uint32_t level = 0; level < NB_LEVELS; level++){
            uint32_t w = W0 >> level;
            uint32_t h = H0 >> level;

            torch::Tensor t = torch::from_blob(
                d_input + float_offset,
                {1, 5, (int64_t)h, (int64_t)w},
                torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(torch::kCUDA)
            );
            pyramid.push_back(t);
            float_offset += 5 * uint64_t(w) * h;
        }
    }

    // ---------------------------------------------------------------
    // 6. Run inference
    // ---------------------------------------------------------------
    torch::NoGradGuard no_grad;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(pyramid);

    torch::Tensor output         = model.forward(inputs).toTensor();
    torch::Tensor output_squeezed = output.squeeze(0).contiguous();  // [3, H0, W0]

    cudaMemcpy(d_output,
               output_squeezed.data_ptr<float>(),
               output_float_bytes,
               cudaMemcpyDeviceToDevice);

    // ---------------------------------------------------------------
    // 7. Pack model output into colorbuffer[level 0]
    // ---------------------------------------------------------------
    {
        uint64_t num_pixels = uint64_t(W0) * H0;
        uint32_t block_size = 256;
        uint32_t grid_size  = (num_pixels + block_size - 1) / block_size;
        OptionalLaunchSettings launch_settings = {
            .gridsize  = grid_size,
            .blocksize = block_size
        };

        void* args[] = { &d_output, &target.colorbuffer, &W0, &H0 };
        GpuVersion::prog->launch("kernel_pack_tensor_to_colorbuffer",
            args, launch_settings);
    }
}