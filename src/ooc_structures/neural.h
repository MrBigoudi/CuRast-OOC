#pragma once

#include "gpuVersion.h"
#include <torch/torch.h>
#include <torch/script.h>


// =============================================================================
// Configuration: mirrors the Python training script (test5.py).
// These values MUST match the ones the checkpoint was trained with,
// otherwise weight loading fails with a shape mismatch.
// =============================================================================

enum class OutputActivation {
    Sigmoid,   // OUTPUT_ACTIVATION = "sigmoid"
    Linear     // OUTPUT_ACTIVATION = "linear"  (Identity, clamp at inference)
};

struct PyramidUNetConfig {
    // Per-level input channels, concatenated in this order:
    //   RGB (perturbed), [LOD], [depth]
    uint32_t perturbed_channels = 3;      // PERTURBED_CHANNELS
    bool     use_lod            = false;  // USE_LOD
    bool     lod_rgb            = false;  // LOD_MODE == "RGB"
    bool     use_depth          = true;   // USE_DEPTH
    bool     depth_rgb          = false;  // DEPTH_MODE == "RGB"

    uint32_t base_channels      = 16;     // BASE_CHANNELS
    uint32_t output_channels    = 3;      // N_CHANNELS
    float    negative_slope     = 0.2f;   // LeakyReLU slope in GatedConv2d

    OutputActivation output_activation = OutputActivation::Linear;

    uint32_t lod_channels()   const { return use_lod   ? (lod_rgb   ? 3u : 1u) : 0u; }
    uint32_t depth_channels() const { return use_depth ? (depth_rgb ? 3u : 1u) : 0u; }
    uint32_t level_channels() const {
        return perturbed_channels + lod_channels() + depth_channels();
    }
};


// =============================================================================
// Gated convolution (Yu et al. 2019)
// =============================================================================

struct GatedConvolution : torch::nn::Module {
    uint32_t out_channels;

    torch::nn::Conv2d    conv            = nullptr;
    torch::nn::LeakyReLU activation      = nullptr;
    torch::nn::Sigmoid   gate_activation = nullptr;

    GatedConvolution(
        uint32_t in_channels,
        uint32_t out_channels,
        uint32_t kernel_size    = 3,
        uint32_t padding        = 1,
        float    negative_slope = 0.2f
    );

    torch::Tensor forward(torch::Tensor x);
};


struct GatedDoubleConv : torch::nn::Module {
    std::shared_ptr<GatedConvolution> conv1;
    std::shared_ptr<GatedConvolution> conv2;

    GatedDoubleConv(
        uint32_t in_channels,
        uint32_t out_channels,
        float    negative_slope = 0.2f
    );

    torch::Tensor forward(torch::Tensor x);
};


// =============================================================================
// Pyramid U-Net (4 levels, ADOP-style)
// =============================================================================
//
//   level0 (1920x1080, input)          -> skip0 --------------------+
//        | avgpool, concat pyramid[1]                                |
//   level1 (960x540)                    -> skip1 -----------------+  |
//        | avgpool, concat pyramid[2]                              |  |
//   level2 (480x270)                    -> skip2 --------------+   |  |
//        | avgpool, concat pyramid[3]                           |   |  |
//   level3 / bottleneck (240x135)                                |   |  |
//        | bilinear upsample, concat skip2 -----------------------+   |  |
//   up2 (480x270)                                                     |  |
//        | bilinear upsample, concat skip1 ---------------------------+  |
//   up1 (960x540)                                                        |
//        | bilinear upsample, concat skip0 ------------------------------+
//   up0 (1920x1080) -> 1x1 conv -> output activation -> output (RGB)
//
// =============================================================================

struct PyramidUNet : torch::nn::Module {
    static constexpr uint32_t NB_LEVELS = 4;

    PyramidUNetConfig config;
    uint32_t channels[NB_LEVELS];

    torch::nn::AvgPool2d pool = nullptr;

    std::shared_ptr<GatedDoubleConv> encoders[NB_LEVELS];  // "enc0".."enc3"
    std::shared_ptr<GatedDoubleConv> ups[NB_LEVELS - 1];    // "up0".."up2"

    torch::nn::Conv2d output_conv = nullptr;
    // Output activation has no parameters (Identity / Sigmoid in Python),
    // so it is applied functionally in forward() based on config.

    explicit PyramidUNet(const PyramidUNetConfig& config);

    torch::Tensor forward(const std::vector<torch::Tensor>& pyramid);
};


// =============================================================================
// Runtime wrapper
// =============================================================================

struct NeuralNet {
    static inline PyramidUNetConfig            config;
    static inline std::shared_ptr<PyramidUNet> net;
    static inline torch::jit::script::Module   jit_model;

    // true  -> run the native C++ PyramidUNet (weights copied from the file)
    // false -> run the TorchScript module directly
    static inline bool use_native = true;

    static void load(const std::string& model_path);
    static void infer(RenderTarget& target);
};