#pragma once

#include "gpuVersion.h"
#include <torch/torch.h>

#include <array>
#include <string>
#include <vector>


// =============================================================================
// Model settings
// =============================================================================
//
// A model is ONE file exported by train.py (<checkpoint_dir>/<run.name>.pt):
//   - the resolved training settings (the [model], [data] and [inference]
//     sections of the TOML, with every default filled in),
//   - the weights,
//   - a reference input / output pair.
// NeuralNet::load(path) builds the matching network from the settings, copies
// the weights and checks its own output against the reference, so the C++
// side never has to be told which experiment a file comes from.
//
// Two architectures: the gated U-Net of train.py (PointFillNet) and the pyramid
// vision transformer of train_v2.py (PyramidViT), chosen by model.architecture.

enum class Architecture     { Conv, ViT };                // model.architecture ("conv" | "vit")
enum class MultiscaleInput  { Rendered, Pooled, None };   // model.multiscale_input
enum class InputColorSpace  { RGB, Lab };                 // model.input_color_space
enum class EncoderSplit     { None, Modality, Channel };  // model.encoder.split
enum class FusionMode       { Avg, Max, Concat, Conv, Attention };
enum class OutputHead       { RGB, Kernel };              // model.output
enum class OutputActivation { Sigmoid, Linear };          // model.output_activation

struct ModelSettings {
    std::string  name = "unnamed";                        // run.name
    Architecture architecture = Architecture::Conv;       // read from the export file

    // Input channels per pyramid level, in this order: RGB, [LOD], [depth]
    bool use_lod   = false;
    bool lod_rgb   = false;
    bool use_depth = true;
    bool depth_rgb = false;

    uint32_t         base_channels     = 16;
    MultiscaleInput  multiscale_input  = MultiscaleInput::Rendered;
    InputColorSpace  color_space       = InputColorSpace::RGB;
    bool             prev_frame        = false;           // previous output fed back
    bool             prev_reprojected  = false;

    EncoderSplit     encoder_split        = EncoderSplit::None;
    FusionMode       fusion               = FusionMode::Avg;
    bool             share_depth          = false;
    uint32_t         branch_base_channels = 16;

    OutputHead               output             = OutputHead::RGB;
    OutputActivation         output_activation  = OutputActivation::Linear;
    std::array<uint32_t, 4>  kernel_sizes       = { 5, 5, 5, 7 };
    bool                     kernel_residual    = false;
    float                    kernel_residual_scale = 0.1f;

    // [inference]: colour written into empty input pixels
    bool  hole_color_is_background = false;               // "background"
    float hole_color[3] = { 1.0f, 1.0f, 1.0f };

    // [model.vit] of train_v2.py, read from the export file (not user settings)
    struct ViT {
        uint32_t embed_dim       = 192;
        uint32_t depth           = 8;
        uint32_t heads           = 6;
        float    mlp_ratio       = 4.0f;
        uint32_t window          = 8;     // attention window, in tokens
        bool     shifted_windows = true;
        bool     refine          = true;
        uint32_t refine_channels = 16;
        bool     conv_stem       = false; // [model.vit.conv_stem] enabled
        bool     include_raw     = true;  // raw pyramid values in the embedding too
    } vit;

    float negative_slope = 0.2f;                          // LeakyReLU in GatedConv2d

    uint32_t lod_channels()   const { return use_lod   ? (lod_rgb   ? 3u : 1u) : 0u; }
    uint32_t depth_channels() const { return use_depth ? (depth_rgb ? 3u : 1u) : 0u; }
    uint32_t level_channels() const { return 3u + lod_channels() + depth_channels(); }
    uint32_t prev_channels()  const { return prev_frame ? 4u : 0u; }   // RGB + validity
    // Pyramid levels the renderer must provide (1: level 0 only)
    uint32_t input_levels()   const { return multiscale_input == MultiscaleInput::Rendered ? 4u : 1u; }

    // Channel indices per encoder branch (same rule as build_channel_groups in train.py)
    std::vector<std::vector<int64_t>> channel_groups() const;

    // From the "settings" entry of an export file (nested dicts)
    static ModelSettings from_export(const torch::IValue& settings);
    std::string summary() const;
};


// =============================================================================
// Building blocks (sub-module names match train.py so the weights map 1:1)
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

    GatedDoubleConv(uint32_t in_channels, uint32_t out_channels, float negative_slope = 0.2f);

    torch::Tensor forward(torch::Tensor x);
};


// One encoder working on a subset of the input channels ("encoders.N")
struct BranchEncoder : torch::nn::Module {
    torch::Tensor        channel_index;                    // buffer, not saved by Python
    torch::nn::AvgPool2d pool = nullptr;
    std::shared_ptr<GatedDoubleConv> enc[4];               // "enc0".."enc3"

    BranchEncoder(const std::vector<int64_t>& channels, uint32_t base_channels, float negative_slope);

    // returns skip0, skip1, skip2, bottleneck
    std::array<torch::Tensor, 4> forward(const std::vector<torch::Tensor>& levels);
};


// Merges the branch features of one level ("fuse0".."fuse3")
struct Fusion : torch::nn::Module {
    FusionMode mode;
    uint32_t   out_channels;

    std::shared_ptr<GatedConvolution> mix;                 // "mix"   (Conv)
    torch::nn::Conv2d                 score = nullptr;     // "score" (Attention)

    Fusion(FusionMode mode, uint32_t num_branches, uint32_t channels, float negative_slope);

    torch::Tensor forward(const std::vector<torch::Tensor>& features);
};


// 1x1 conv -> K*K kernel logits (+ blend logit) ("head0".."head3")
struct KernelHead : torch::nn::Module {
    uint32_t          kernel_size;
    bool              with_blend;
    torch::nn::Conv2d conv = nullptr;

    KernelHead(uint32_t in_channels, uint32_t kernel_size, bool with_blend);

    // returns (filtered colours, blend weight)
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& features,
                                                    const torch::Tensor& colors);
};


// =============================================================================
// Network: every variant of train.py's PointFillNet
// =============================================================================
//
//   input levels (rendered pyramid, or level 0 only)
//        | [RGB -> Lab]   [+ previous output, avg-pooled per level]
//   encoder: enc0..enc3                      (encoder.split = none)
//         or encoders.N + fuse0..fuse3        (one encoder per channel group)
//   decoder: up2, up1, up0
//   output : output_conv (+ sigmoid)          (output = rgb)
//         or head3..head0 kernels on the input colours, coarse to fine
//            [+ residual]                     (output = kernel)

// Common interface of the two architectures (what NeuralNet runs)
struct FillNetwork : torch::nn::Module {
    virtual ~FillNetwork() = default;

    // pyramid: input_levels() tensors [1, level_channels, H >> l, W >> l]
    // prev   : [1, 4, H, W] previous output + validity (ignored without prev_frame)
    virtual torch::Tensor forward(const std::vector<torch::Tensor>& pyramid, const torch::Tensor& prev) = 0;
};

struct PointFillNet : FillNetwork {
    static constexpr uint32_t NB_LEVELS = 4;

    ModelSettings        settings;
    torch::nn::AvgPool2d pool = nullptr;

    std::shared_ptr<GatedDoubleConv>            enc[NB_LEVELS];   // single encoder
    torch::nn::ModuleList                       encoders = nullptr;
    std::vector<std::shared_ptr<BranchEncoder>> branches;         // = encoders, typed
    std::shared_ptr<Fusion>                     fuse[NB_LEVELS];

    std::shared_ptr<GatedDoubleConv> up[NB_LEVELS - 1];           // "up0".."up2"

    torch::nn::Conv2d           output_conv = nullptr;            // output = rgb
    std::shared_ptr<KernelHead> head[NB_LEVELS];                  // output = kernel
    torch::nn::Conv2d           residual = nullptr;

    explicit PointFillNet(const ModelSettings& settings);

    torch::Tensor forward(const std::vector<torch::Tensor>& pyramid, const torch::Tensor& prev) override;

private:
    std::vector<torch::Tensor> input_levels(const std::vector<torch::Tensor>& pyramid,
                                            const torch::Tensor& prev);
};


// =============================================================================
// Pyramid vision transformer (train_v2.py), same sub-module names
// =============================================================================
//
//   levels (+ Lab, + previous output)  --pixel_unshuffle(8, 4, 2, 1)-->  one token
//   per coarsest-level pixel  [+ conv stem features "stem.enc0..3", unshuffled]
//   -> embed (1x1 conv) -> + pos (depthwise 3x3) -> blocks.N (window attention,
//   every other block shifted by window/2; MLP) -> norm
//   -> head (linear D -> 8*8*3, pixel_shuffle) [+ refine]           (output = rgb)
//   or kernel_heads.L (per-pixel kernels on the input colours)      (output = kernel)
//      [+ scaled tanh(refine)]                                      (kernel.residual)
//
// Inputs whose size is not a multiple of 8 are edge-padded and the output cropped.

struct WindowAttention : torch::nn::Module {
    int64_t heads, window, shift;
    torch::nn::Linear qkv = nullptr;                      // "qkv"
    torch::nn::Linear proj = nullptr;                     // "proj"

    WindowAttention(int64_t dim, int64_t heads, int64_t window, int64_t shift);
    torch::Tensor forward(const torch::Tensor& x);       // (B, H, W, D) token grid
};

struct TransformerBlock : torch::nn::Module {
    torch::nn::LayerNorm             norm1 = nullptr;
    std::shared_ptr<WindowAttention> attn;
    torch::nn::LayerNorm             norm2 = nullptr;
    torch::nn::Sequential            mlp = nullptr;      // "mlp.0" Linear, GELU, "mlp.2" Linear

    TransformerBlock(int64_t dim, int64_t heads, float mlp_ratio, int64_t window, int64_t shift);
    torch::Tensor forward(const torch::Tensor& x);
};

struct PyramidViT : FillNetwork {
    static constexpr uint32_t NB_LEVELS = 4;
    static constexpr int64_t  PATCH = 8;                  // level-0 pixels per token side

    ModelSettings        settings;
    torch::nn::AvgPool2d pool = nullptr;

    std::shared_ptr<BranchEncoder> stem;                  // "stem.enc0".."stem.enc3" (conv_stem)
    torch::nn::Conv2d     embed = nullptr;
    torch::nn::Conv2d     pos = nullptr;
    torch::nn::ModuleList blocks = nullptr;
    std::vector<std::shared_ptr<TransformerBlock>> block_list;
    torch::nn::LayerNorm  norm = nullptr;

    torch::nn::Linear     head = nullptr;                 // output = rgb
    torch::nn::ModuleList kernel_heads = nullptr;         // output = kernel
    std::vector<torch::nn::Linear> kernel_head_list;
    torch::nn::Sequential refine = nullptr;               // "refine.0" conv, GELU, "refine.2" conv

    explicit PyramidViT(const ModelSettings& settings);

    torch::Tensor forward(const std::vector<torch::Tensor>& pyramid, const torch::Tensor& prev) override;
};


// =============================================================================
// Runtime wrapper
// =============================================================================

struct NeuralNet {
    // Folder scanned for exported models (*.pt; training checkpoints are *.pth)
    static inline std::string models_dir = "models";

    static inline ModelSettings                 settings;
    static inline std::shared_ptr<FillNetwork>  net;      // PointFillNet or PyramidViT
    static inline std::string                   loaded_path;   // "" when no model is loaded
    static inline std::string                   last_error;    // last failed load, for the UI

    // Previous clamped output [1, 3, H, W] (models with prev_frame)
    static inline torch::Tensor prev_output;

    // Sorted paths of the *.pt files in models_dir
    static std::vector<std::string> list_models();

    // Unload the current model (GPU memory released), then load model_path.
    // "" only unloads. Returns false (and sets last_error) if loading failed;
    // no model is loaded in that case.
    static bool switch_model(const std::string& model_path);

    // Free the current model and return its GPU memory to the driver
    static void unload();

    static bool is_loaded() { return net != nullptr; }

    // Load an export file written by train.py. Throws on failure.
    // Prefer switch_model(), which frees the previous model first.
    static void load(const std::string& model_path);

    // Forget the previous frame (camera cut, scene change, resize, ...)
    static void reset_temporal_state();

    // true if a model is loaded and it reads the lower pyramid levels
    static bool needs_rendered_pyramid();

    // CuRastSettings::useMultiScale = useNN && the model reads the lower levels
    // (left alone while screenshots are taken). Call once per frame.
    static void sync_rendering_settings();

    // No-op unless CuRastSettings::useNN is on and a model is loaded
    static void infer(RenderTarget& target);
};