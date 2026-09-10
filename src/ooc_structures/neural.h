#pragma once

#include "gpuVersion.h"
#include <torch/torch.h>


struct GatedConvolution : torch::nn::Module {
    uint32_t out_channels;

    torch::nn::Conv2d conv = nullptr;
    torch::nn::LeakyReLU activation = nullptr;
    torch::nn::Sigmoid gate_activation = nullptr;

    GatedConvolution(
        uint32_t in_channels,
        uint32_t out_channels,
        uint32_t kernel_size,
        uint32_t padding
    );

    torch::Tensor forward(torch::Tensor x);
};

struct GatedDoubleConv : torch::nn::Module {
    std::shared_ptr<GatedConvolution> conv1;
    std::shared_ptr<GatedConvolution> conv2;

    GatedDoubleConv(
        uint32_t in_channels,
        uint32_t out_channels
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
//   up0 (1920x1080) -> 1x1 conv -> output (RGB)
//
// =============================================================================



struct PyramidUNet : torch::nn::Module {
    uint32_t channels[4];

    torch::nn::AvgPool2d pool = nullptr;

    std::shared_ptr<GatedDoubleConv> encoders[4];
    std::shared_ptr<GatedDoubleConv> ups[3];

    torch::nn::Conv2d output_conv = nullptr;
    torch::nn::Sigmoid output_activation = nullptr;

    PyramidUNet(
        uint32_t level_channels,
        uint32_t base_channels,
        uint32_t output_channels
    );

    torch::Tensor forward(
        torch::Tensor x,
        const std::vector<torch::Tensor>& pyramid
    );
};



struct NeuralNet {
    static inline std::shared_ptr<PyramidUNet> model = nullptr;

    static void load(const std::string& model_path);

    static void resize_to_match(
        torch::Tensor& source,
        const torch::Tensor& target
    );
};