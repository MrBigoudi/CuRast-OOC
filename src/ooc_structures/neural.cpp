#include "neural.h"


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


torch::Tensor PyramidUNet::forward(
    torch::Tensor x,
    const std::vector<torch::Tensor>& pyramid)
{
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
    x = pool->forward(skip0);
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
    model = std::make_shared<PyramidUNet>(5, 16, 3);
    torch::load(model, model_path.c_str());
}