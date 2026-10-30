#include "neural.h"

#include <c10/cuda/CUDACachingAllocator.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

namespace F = torch::nn::functional;


// =============================================================================
// Tensor helpers (same operations as train.py)
// =============================================================================

// Bilinearly resize source to target's spatial size if they differ (resize_to_match)
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
            .align_corners(false));
}

// Unconditional bilinear upsample to the reference size (decoder path)
static torch::Tensor upsample_to(const torch::Tensor& x, const torch::Tensor& reference)
{
    return F::interpolate(
        x,
        F::InterpolateFuncOptions()
            .size(std::vector<int64_t>{ reference.size(-2), reference.size(-1) })
            .mode(torch::kBilinear)
            .align_corners(false));
}

// sRGB -> CIELAB (D65) on the first 3 channels: L/100, a/128, b/128 (RGB2LabInput)
static torch::Tensor rgb_to_lab(const torch::Tensor& input)
{
    auto x    = input.to(torch::kFloat32);
    auto rgb  = x.slice(1, 0, 3).clamp(0.0, 1.0);
    auto rest = x.slice(1, 3);

    auto linear = torch::where(rgb > 0.04045,
                               ((rgb + 0.055) / 1.055).clamp_min(0.0).pow(2.4),
                               rgb / 12.92);
    auto r = linear.slice(1, 0, 1);
    auto g = linear.slice(1, 1, 2);
    auto b = linear.slice(1, 2, 3);

    auto X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    auto Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    auto Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    auto white = torch::tensor({ 0.95047f, 1.0f, 1.08883f }, x.options()).view({ 1, 3, 1, 1 });
    auto xyz = torch::cat({ X, Y, Z }, 1) / white;

    const double eps   = 216.0 / 24389.0;
    const double kappa = 24389.0 / 27.0;
    auto f = torch::where(xyz > eps, xyz.clamp_min(eps).pow(1.0 / 3.0), (kappa * xyz + 16.0) / 116.0);
    auto fx = f.slice(1, 0, 1);
    auto fy = f.slice(1, 1, 2);
    auto fz = f.slice(1, 2, 3);

    auto L  = (116.0 * fy - 16.0) / 100.0;
    auto a  = (500.0 * (fx - fy)) / 128.0;
    auto bb = (200.0 * (fy - fz)) / 128.0;
    return torch::cat({ L, a, bb, rest }, 1);
}

// out[p] = sum_j softmax(logits[p])_j * colors[p + offset_j], replicate padding (apply_kernel)
static torch::Tensor apply_kernel(const torch::Tensor& colors_in, const torch::Tensor& logits, int64_t k)
{
    auto weights = torch::softmax(logits.to(torch::kFloat32), 1);
    auto colors  = colors_in.to(torch::kFloat32);
    const int64_t pad = k / 2;
    auto padded = F::pad(colors, F::PadFuncOptions({ pad, pad, pad, pad }).mode(torch::kReplicate));
    const int64_t H = colors.size(-2);
    const int64_t W = colors.size(-1);

    auto out = torch::zeros_like(colors);
    int64_t index = 0;
    for (int64_t dy = 0; dy < k; dy++) {
        for (int64_t dx = 0; dx < k; dx++) {
            out = out + weights.slice(1, index, index + 1) * padded.slice(2, dy, dy + H).slice(3, dx, dx + W);
            index++;
        }
    }
    return out;
}


// =============================================================================
// ModelSettings
// =============================================================================

// "model.encoder.split" -> value inside nested GenericDicts
static torch::IValue lookup(const torch::IValue& root, const std::string& dotted)
{
    torch::IValue current = root;
    std::stringstream path(dotted);
    std::string part;
    while (std::getline(path, part, '.')) {
        auto dict = current.toGenericDict();
        torch::IValue key(part);
        if (!dict.contains(key)) {
            throw std::runtime_error("Setting '" + dotted + "' is missing from the model file");
        }
        current = dict.at(key);
    }
    return current;
}

static double as_number(const torch::IValue& v)
{
    return v.isInt() ? double(v.toInt()) : v.toDouble();
}

static std::string get_string(const torch::IValue& s, const std::string& key) { return lookup(s, key).toStringRef(); }
static bool        get_bool  (const torch::IValue& s, const std::string& key) { return lookup(s, key).toBool(); }
static double      get_number(const torch::IValue& s, const std::string& key) { return as_number(lookup(s, key)); }

template <typename Enum>
static Enum parse_enum(const std::string& key, const std::string& value,
                       std::initializer_list<std::pair<const char*, Enum>> options)
{
    for (const auto& [name, e] : options) {
        if (value == name) return e;
    }
    throw std::runtime_error("Unsupported value '" + value + "' for setting '" + key + "'");
}

ModelSettings ModelSettings::from_export(const torch::IValue& s)
{
    ModelSettings m;
    m.name      = get_string(s, "run.name");
    m.use_lod   = get_bool(s, "data.use_lod");
    m.lod_rgb   = get_string(s, "data.lod_mode") == "RGB";
    m.use_depth = get_bool(s, "data.use_depth");
    m.depth_rgb = get_string(s, "data.depth_mode") == "RGB";

    m.base_channels    = uint32_t(get_number(s, "model.base_channels"));
    m.multiscale_input = parse_enum<MultiscaleInput>("model.multiscale_input", get_string(s, "model.multiscale_input"),
        { { "rendered", MultiscaleInput::Rendered }, { "pooled", MultiscaleInput::Pooled }, { "none", MultiscaleInput::None } });
    m.color_space = parse_enum<InputColorSpace>("model.input_color_space", get_string(s, "model.input_color_space"),
        { { "rgb", InputColorSpace::RGB }, { "lab", InputColorSpace::Lab } });
    m.prev_frame       = get_bool(s, "model.prev_frame");
    m.prev_reprojected = get_bool(s, "model.prev_reprojected");

    m.encoder_split = parse_enum<EncoderSplit>("model.encoder.split", get_string(s, "model.encoder.split"),
        { { "none", EncoderSplit::None }, { "modality", EncoderSplit::Modality }, { "channel", EncoderSplit::Channel } });
    m.fusion = parse_enum<FusionMode>("model.encoder.fusion", get_string(s, "model.encoder.fusion"),
        { { "avg", FusionMode::Avg }, { "max", FusionMode::Max }, { "concat", FusionMode::Concat },
          { "conv", FusionMode::Conv }, { "attention", FusionMode::Attention } });
    m.share_depth          = get_bool(s, "model.encoder.share_depth");
    m.branch_base_channels = uint32_t(get_number(s, "model.encoder.branch_base_channels"));

    m.output = parse_enum<OutputHead>("model.output", get_string(s, "model.output"),
        { { "rgb", OutputHead::RGB }, { "kernel", OutputHead::Kernel } });
    m.output_activation = parse_enum<OutputActivation>("model.output_activation", get_string(s, "model.output_activation"),
        { { "linear", OutputActivation::Linear }, { "sigmoid", OutputActivation::Sigmoid } });

    auto sizes = lookup(s, "model.kernel.sizes").toList();
    if (sizes.size() != 4) throw std::runtime_error("model.kernel.sizes must hold 4 values");
    for (size_t i = 0; i < 4; i++) m.kernel_sizes[i] = uint32_t(as_number(sizes.get(i)));
    m.kernel_residual       = get_bool(s, "model.kernel.residual");
    m.kernel_residual_scale = float(get_number(s, "model.kernel.residual_scale"));

    // Architecture (older exports have no key: they are all U-Nets) and [model.vit]
    {
        auto model = lookup(s, "model").toGenericDict();
        if (model.contains(torch::IValue("architecture"))) {
            m.architecture = parse_enum<Architecture>("model.architecture", get_string(s, "model.architecture"),
                { { "conv", Architecture::Conv }, { "vit", Architecture::ViT } });
        }
        if (m.architecture == Architecture::ViT) {
            m.vit.embed_dim       = uint32_t(get_number(s, "model.vit.embed_dim"));
            m.vit.depth           = uint32_t(get_number(s, "model.vit.depth"));
            m.vit.heads           = uint32_t(get_number(s, "model.vit.heads"));
            m.vit.mlp_ratio       = float(get_number(s, "model.vit.mlp_ratio"));
            m.vit.window          = uint32_t(get_number(s, "model.vit.window"));
            m.vit.shifted_windows = get_bool(s, "model.vit.shifted_windows");
            m.vit.refine          = get_bool(s, "model.vit.refine");
            m.vit.refine_channels = uint32_t(get_number(s, "model.vit.refine_channels"));
            m.vit.conv_stem       = get_bool(s, "model.vit.conv_stem.enabled");
            m.vit.include_raw     = get_bool(s, "model.vit.conv_stem.include_raw");
            if (m.multiscale_input != MultiscaleInput::Rendered || m.encoder_split != EncoderSplit::None) {
                throw std::runtime_error("ViT models need multiscale_input = rendered and encoder.split = none");
            }
            if (m.vit.embed_dim % m.vit.heads != 0) {
                throw std::runtime_error("model.vit.embed_dim must be a multiple of model.vit.heads");
            }
        }
    }

    // auto hole = lookup(s, "inference.hole_color");
    // if (hole.isString()) {
    //     if (hole.toStringRef() != "background") {
    //         throw std::runtime_error("inference.hole_color must be [r, g, b] or \"background\"");
    //     }
    //     m.hole_color_is_background = true;
    // } else {
    //     auto rgb = hole.toList();
    //     if (rgb.size() != 3) throw std::runtime_error("inference.hole_color must have 3 values");
    //     for (size_t i = 0; i < 3; i++) m.hole_color[i] = float(as_number(rgb.get(i)));
    // }
    m.hole_color_is_background = true;
    return m;
}

std::vector<std::vector<int64_t>> ModelSettings::channel_groups() const
{
    const int64_t lc = level_channels();
    std::vector<int64_t> color = { 0, 1, 2 };
    std::vector<int64_t> lod, depth;
    for (int64_t c = 3; c < 3 + int64_t(lod_channels()); c++) lod.push_back(c);
    for (int64_t c = 3 + int64_t(lod_channels()); c < lc; c++) depth.push_back(c);

    std::vector<std::vector<int64_t>> groups;
    if (encoder_split == EncoderSplit::None) {
        std::vector<int64_t> all;
        for (int64_t c = 0; c < lc; c++) all.push_back(c);
        groups.push_back(all);
    } else if (encoder_split == EncoderSplit::Modality) {
        for (const auto& g : { color, lod, depth }) {
            if (!g.empty()) groups.push_back(g);
        }
    } else {
        for (int64_t c = 0; c < lc; c++) groups.push_back({ c });
    }

    if (share_depth && !depth.empty() && encoder_split != EncoderSplit::None) {
        for (auto& g : groups) {
            for (int64_t d : depth) {
                if (std::find(g.begin(), g.end(), d) == g.end()) g.push_back(d);
            }
        }
    }
    return groups;
}

std::string ModelSettings::summary() const
{
    static const char* multiscale_names[] = { "rendered", "pooled", "none" };
    static const char* split_names[]      = { "none", "modality", "channel" };
    static const char* fusion_names[]     = { "avg", "max", "concat", "conv", "attention" };
    std::ostringstream ss;
    ss << "'" << name << "': " << (architecture == Architecture::ViT ? "vit" : "conv")
       << " level_channels=" << level_channels()
       << " base=" << base_channels
       << " multiscale=" << multiscale_names[int(multiscale_input)]
       << " colour=" << (color_space == InputColorSpace::Lab ? "lab" : "rgb")
       << " prev_frame=" << (prev_frame ? "yes" : "no")
       << " encoder=" << split_names[int(encoder_split)] << "/" << fusion_names[int(fusion)]
       << " output=" << (output == OutputHead::Kernel ? "kernel" : "rgb");
    if (architecture == Architecture::ViT) {
        ss << " vit(dim=" << vit.embed_dim << " depth=" << vit.depth << " heads=" << vit.heads
           << " window=" << vit.window << (vit.shifted_windows ? " shifted" : "")
           << (vit.refine ? " refine" : "") << (vit.conv_stem ? " conv_stem" : "")
           << (vit.conv_stem && !vit.include_raw ? "(no raw)" : "") << ")";
    }
    ss << " hole_color=";
    if (hole_color_is_background) ss << "background";
    else ss << "(" << hole_color[0] << ", " << hole_color[1] << ", " << hole_color[2] << ")";
    return ss.str();
}


// =============================================================================
// Building blocks
// =============================================================================

GatedConvolution::GatedConvolution(
    uint32_t in_channels, uint32_t out_channels, uint32_t kernel_size, uint32_t padding, float negative_slope)
    : out_channels(out_channels)
{
    conv = register_module("conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, out_channels * 2, kernel_size).padding(padding)));
    activation = register_module("activation", torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().negative_slope(negative_slope).inplace(false)));
    gate_activation = register_module("gate_activation", torch::nn::Sigmoid());
}

torch::Tensor GatedConvolution::forward(torch::Tensor x)
{
    auto parts = conv->forward(x).split(out_channels, 1);
    return activation->forward(parts[0]) * gate_activation->forward(parts[1]);
}

GatedDoubleConv::GatedDoubleConv(uint32_t in_channels, uint32_t out_channels, float negative_slope)
{
    conv1 = register_module("conv1", std::make_shared<GatedConvolution>(in_channels, out_channels, 3, 1, negative_slope));
    conv2 = register_module("conv2", std::make_shared<GatedConvolution>(out_channels, out_channels, 3, 1, negative_slope));
}

torch::Tensor GatedDoubleConv::forward(torch::Tensor x)
{
    return conv2->forward(conv1->forward(x));
}

BranchEncoder::BranchEncoder(const std::vector<int64_t>& channels, uint32_t base, float slope)
{
    const uint32_t n = uint32_t(channels.size());
    channel_index = register_buffer("channel_index", torch::tensor(channels, torch::kLong));
    pool = register_module("pool", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2)));
    enc[0] = register_module("enc0", std::make_shared<GatedDoubleConv>(n,            base,     slope));
    enc[1] = register_module("enc1", std::make_shared<GatedDoubleConv>(base + n,     base * 2, slope));
    enc[2] = register_module("enc2", std::make_shared<GatedDoubleConv>(base * 2 + n, base * 4, slope));
    enc[3] = register_module("enc3", std::make_shared<GatedDoubleConv>(base * 4 + n, base * 8, slope));
}

std::array<torch::Tensor, 4> BranchEncoder::forward(const std::vector<torch::Tensor>& levels)
{
    std::vector<torch::Tensor> p;
    for (const auto& level : levels) p.push_back(level.index_select(1, channel_index));

    auto skip0 = enc[0]->forward(p[0]);
    auto skip1 = enc[1]->forward(torch::cat({ resize_to_match(pool->forward(skip0), p[1]), p[1] }, 1));
    auto skip2 = enc[2]->forward(torch::cat({ resize_to_match(pool->forward(skip1), p[2]), p[2] }, 1));
    auto bottleneck = enc[3]->forward(torch::cat({ resize_to_match(pool->forward(skip2), p[3]), p[3] }, 1));
    return { skip0, skip1, skip2, bottleneck };
}

Fusion::Fusion(FusionMode mode, uint32_t num_branches, uint32_t channels, float slope)
    : mode(mode), out_channels(mode == FusionMode::Concat ? channels * num_branches : channels)
{
    if (mode == FusionMode::Conv) {
        mix = register_module("mix", std::make_shared<GatedConvolution>(channels * num_branches, channels, 1, 0, slope));
    } else if (mode == FusionMode::Attention) {
        score = register_module("score", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels * num_branches, num_branches, 1)));
    }
}

torch::Tensor Fusion::forward(const std::vector<torch::Tensor>& features)
{
    switch (mode) {
        case FusionMode::Avg: {
            auto out = features[0];
            for (size_t i = 1; i < features.size(); i++) out = out + features[i];
            return out / double(features.size());
        }
        case FusionMode::Max: {
            auto out = features[0];
            for (size_t i = 1; i < features.size(); i++) out = torch::maximum(out, features[i]);
            return out;
        }
        case FusionMode::Concat:
            return torch::cat(features, 1);
        case FusionMode::Conv:
            return mix->forward(torch::cat(features, 1));
        case FusionMode::Attention: {
            auto weights = torch::softmax(score->forward(torch::cat(features, 1)).to(torch::kFloat32), 1);
            auto out = features[0] * weights.slice(1, 0, 1);
            for (size_t i = 1; i < features.size(); i++) {
                out = out + features[i] * weights.slice(1, int64_t(i), int64_t(i) + 1);
            }
            return out;
        }
    }
    throw std::runtime_error("Unknown fusion mode");
}

KernelHead::KernelHead(uint32_t in_channels, uint32_t kernel_size, bool with_blend)
    : kernel_size(kernel_size), with_blend(with_blend)
{
    conv = register_module("conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, kernel_size * kernel_size + (with_blend ? 1 : 0), 1)));
}

std::pair<torch::Tensor, torch::Tensor> KernelHead::forward(const torch::Tensor& features, const torch::Tensor& colors)
{
    auto raw = conv->forward(features).to(torch::kFloat32);
    const int64_t k2 = int64_t(kernel_size) * kernel_size;
    auto filtered = apply_kernel(colors, raw.slice(1, 0, k2), kernel_size);
    auto blend = with_blend ? torch::sigmoid(raw.slice(1, k2, k2 + 1))
                            : torch::ones_like(filtered.slice(1, 0, 1));
    return { filtered, blend };
}


// =============================================================================
// PointFillNet
// =============================================================================

PointFillNet::PointFillNet(const ModelSettings& s)
    : settings(s)
{
    const uint32_t base = s.base_channels;
    const uint32_t c[NB_LEVELS] = { base, base * 2, base * 4, base * 8 };
    const uint32_t in_channels = s.level_channels() + s.prev_channels();
    const float    slope = s.negative_slope;

    pool = register_module("pool", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2)));

    // ---- encoder ----
    uint32_t fused[NB_LEVELS];
    if (s.encoder_split == EncoderSplit::None) {
        const uint32_t extra = (s.multiscale_input == MultiscaleInput::None) ? 0 : in_channels;
        enc[0] = register_module("enc0", std::make_shared<GatedDoubleConv>(in_channels,  c[0], slope));
        enc[1] = register_module("enc1", std::make_shared<GatedDoubleConv>(c[0] + extra, c[1], slope));
        enc[2] = register_module("enc2", std::make_shared<GatedDoubleConv>(c[1] + extra, c[2], slope));
        enc[3] = register_module("enc3", std::make_shared<GatedDoubleConv>(c[2] + extra, c[3], slope));
        for (uint32_t l = 0; l < NB_LEVELS; l++) fused[l] = c[l];
    } else {
        const auto groups = s.channel_groups();
        const uint32_t bb = s.branch_base_channels;
        encoders = register_module("encoders", torch::nn::ModuleList());
        for (const auto& group : groups) {
            auto branch = std::make_shared<BranchEncoder>(group, bb, slope);
            encoders->push_back(branch);                       // "encoders.N"
            branches.push_back(branch);
        }
        const uint32_t n = uint32_t(groups.size());
        for (uint32_t l = 0; l < NB_LEVELS; l++) {
            fuse[l] = register_module("fuse" + std::to_string(l),
                                      std::make_shared<Fusion>(s.fusion, n, bb << l, slope));
            fused[l] = fuse[l]->out_channels;
        }
    }

    // ---- decoder ----
    up[2] = register_module("up2", std::make_shared<GatedDoubleConv>(fused[3] + fused[2], c[2], slope));
    up[1] = register_module("up1", std::make_shared<GatedDoubleConv>(c[2] + fused[1],     c[1], slope));
    up[0] = register_module("up0", std::make_shared<GatedDoubleConv>(c[1] + fused[0],     c[0], slope));

    // ---- output head ----
    if (s.output == OutputHead::RGB) {
        output_conv = register_module("output_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(c[0], 3, 1)));
    } else {
        head[0] = register_module("head0", std::make_shared<KernelHead>(c[0],     s.kernel_sizes[0], true));
        head[1] = register_module("head1", std::make_shared<KernelHead>(c[1],     s.kernel_sizes[1], true));
        head[2] = register_module("head2", std::make_shared<KernelHead>(c[2],     s.kernel_sizes[2], true));
        head[3] = register_module("head3", std::make_shared<KernelHead>(fused[3], s.kernel_sizes[3], false));
        residual = register_module("residual", torch::nn::Conv2d(torch::nn::Conv2dOptions(c[0], 3, 1)));
    }
}

std::vector<torch::Tensor> PointFillNet::input_levels(const std::vector<torch::Tensor>& pyramid,
                                                      const torch::Tensor& prev)
{
    std::vector<torch::Tensor> levels;
    for (const auto& p : pyramid) {
        levels.push_back(settings.color_space == InputColorSpace::Lab ? rgb_to_lab(p) : p);
    }

    if (settings.multiscale_input != MultiscaleInput::Rendered) {
        auto p0 = settings.prev_frame ? torch::cat({ levels[0], prev.to(levels[0].dtype()) }, 1) : levels[0];
        if (settings.multiscale_input == MultiscaleInput::None) return { p0 };
        auto p1 = pool->forward(p0);
        auto p2 = pool->forward(p1);
        return { p0, p1, p2, pool->forward(p2) };
    }

    if (settings.prev_frame) {
        auto q = prev.to(levels[0].dtype());
        for (size_t i = 0; i < levels.size(); i++) {
            if (i > 0) q = resize_to_match(pool->forward(q), levels[i]);
            levels[i] = torch::cat({ levels[i], q }, 1);
        }
    }
    return levels;
}

torch::Tensor PointFillNet::forward(const std::vector<torch::Tensor>& pyramid, const torch::Tensor& prev)
{
    TORCH_CHECK(pyramid.size() == settings.input_levels(),
                "Model '", settings.name, "' expects ", settings.input_levels(), " input levels, got ", pyramid.size());
    TORCH_CHECK(pyramid[0].size(1) == settings.level_channels(),
                "Model '", settings.name, "' expects ", settings.level_channels(), " channels per level, got ",
                pyramid[0].size(1));

    auto levels = input_levels(pyramid, prev);

    // ---- encoder ----
    torch::Tensor skip0, skip1, skip2, bottleneck;
    if (settings.encoder_split == EncoderSplit::None) {
        skip0 = enc[0]->forward(levels[0]);
        if (settings.multiscale_input == MultiscaleInput::None) {
            skip1      = enc[1]->forward(pool->forward(skip0));
            skip2      = enc[2]->forward(pool->forward(skip1));
            bottleneck = enc[3]->forward(pool->forward(skip2));
        } else {
            skip1      = enc[1]->forward(torch::cat({ resize_to_match(pool->forward(skip0), levels[1]), levels[1] }, 1));
            skip2      = enc[2]->forward(torch::cat({ resize_to_match(pool->forward(skip1), levels[2]), levels[2] }, 1));
            bottleneck = enc[3]->forward(torch::cat({ resize_to_match(pool->forward(skip2), levels[3]), levels[3] }, 1));
        }
    } else {
        std::vector<torch::Tensor> per_level[NB_LEVELS];
        for (auto& branch : branches) {
            auto out = branch->forward(levels);
            for (uint32_t l = 0; l < NB_LEVELS; l++) per_level[l].push_back(out[l]);
        }
        skip0      = fuse[0]->forward(per_level[0]);
        skip1      = fuse[1]->forward(per_level[1]);
        skip2      = fuse[2]->forward(per_level[2]);
        bottleneck = fuse[3]->forward(per_level[3]);
    }

    // ---- decoder ----
    auto d2 = up[2]->forward(torch::cat({ upsample_to(bottleneck, skip2), skip2 }, 1));
    auto d1 = up[1]->forward(torch::cat({ upsample_to(d2, skip1), skip1 }, 1));
    auto d0 = up[0]->forward(torch::cat({ upsample_to(d1, skip0), skip0 }, 1));

    // ---- output ----
    if (settings.output == OutputHead::RGB) {
        auto x = output_conv->forward(d0);
        return settings.output_activation == OutputActivation::Sigmoid ? torch::sigmoid(x) : x;
    }

    // Kernels on the untransformed input colours, coarse to fine
    auto out = head[3]->forward(bottleneck, pyramid[3].slice(1, 0, 3)).first;
    const torch::Tensor* features[3] = { &d0, &d1, &d2 };
    for (int l = 2; l >= 0; l--) {
        auto [filtered, blend] = head[l]->forward(*features[l], pyramid[l].slice(1, 0, 3));
        out = blend * filtered + (1.0 - blend) * upsample_to(out, filtered);
    }
    if (settings.kernel_residual) {
        out = out + settings.kernel_residual_scale * torch::tanh(residual->forward(d0).to(torch::kFloat32));
    }
    return out;
}


// =============================================================================
// Pyramid vision transformer (train_v2.py)
// =============================================================================

static int64_t positive_mod(int64_t a, int64_t m) { return ((a % m) + m) % m; }

WindowAttention::WindowAttention(int64_t dim, int64_t heads, int64_t window, int64_t shift)
    : heads(heads), window(window), shift(shift)
{
    qkv  = register_module("qkv",  torch::nn::Linear(dim, 3 * dim));
    proj = register_module("proj", torch::nn::Linear(dim, dim));
}

// Same as train_v2.py window_partition / attention / window_merge: the grid is
// padded by `shift` tokens at the top / left (and up to a multiple of the window
// at the bottom / right); padded tokens are masked out as keys.
torch::Tensor WindowAttention::forward(const torch::Tensor& x)
{
    const int64_t B = x.size(0), H = x.size(1), W = x.size(2), D = x.size(3);
    const int64_t w = window;
    const int64_t pad_top = shift, pad_left = shift;
    const int64_t pad_bottom = positive_mod(-(H + pad_top), w);
    const int64_t pad_right  = positive_mod(-(W + pad_left), w);

    auto xp = F::pad(x, F::PadFuncOptions({ 0, 0, pad_left, pad_right, pad_top, pad_bottom }));
    const int64_t Hp = xp.size(1), Wp = xp.size(2);
    const int64_t nh = Hp / w, nw = Wp / w, L = w * w;

    auto valid = torch::zeros({ Hp, Wp }, torch::TensorOptions().dtype(torch::kBool).device(x.device()));
    valid.slice(0, pad_top, pad_top + H).slice(1, pad_left, pad_left + W).fill_(true);

    auto windows = xp.view({ B, nh, w, nw, w, D }).permute({ 0, 1, 3, 2, 4, 5 }).reshape({ B * nh * nw, L, D });
    auto mask = valid.view({ nh, w, nw, w }).permute({ 0, 2, 1, 3 }).reshape({ nh * nw, L }).repeat({ B, 1 });

    const int64_t n = windows.size(0);
    auto qkv_out = qkv->forward(windows).view({ n, L, 3, heads, D / heads }).permute({ 2, 0, 3, 1, 4 });
    auto q = qkv_out[0], k = qkv_out[1], v = qkv_out[2];                    // (n, heads, L, d)
    auto out = at::scaled_dot_product_attention(q, k, v, mask.unsqueeze(1).unsqueeze(1));
    out = proj->forward(out.transpose(1, 2).reshape({ n, L, D }));

    auto merged = out.view({ B, nh, nw, w, w, D }).permute({ 0, 1, 3, 2, 4, 5 }).reshape({ B, Hp, Wp, D });
    return merged.slice(1, pad_top, pad_top + H).slice(2, pad_left, pad_left + W);
}

TransformerBlock::TransformerBlock(int64_t dim, int64_t heads, float mlp_ratio, int64_t window, int64_t shift)
{
    const int64_t hidden = int64_t(dim * mlp_ratio);                       // int() in Python
    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim })));
    attn  = register_module("attn", std::make_shared<WindowAttention>(dim, heads, window, shift));
    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim })));
    mlp   = register_module("mlp", torch::nn::Sequential(
        torch::nn::Linear(dim, hidden), torch::nn::GELU(), torch::nn::Linear(hidden, dim)));
}

torch::Tensor TransformerBlock::forward(const torch::Tensor& x)
{
    auto y = x + attn->forward(norm1->forward(x));
    return y + mlp->forward(norm2->forward(y));
}

PyramidViT::PyramidViT(const ModelSettings& s)
    : settings(s)
{
    const auto& v = s.vit;
    const int64_t channels = s.level_channels() + s.prev_channels();     // per level
    const int64_t dim = v.embed_dim;

    pool = register_module("pool", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2)));

    // Values per token: raw pyramid (C * (8*8 + 4*4 + 2*2 + 1)) and / or conv stem features
    int64_t raw_values = 0, stem_values = 0;
    for (int64_t l = 0; l < NB_LEVELS; l++) raw_values += channels * (PATCH >> l) * (PATCH >> l);
    int64_t token_values = raw_values;
    if (v.conv_stem) {
        std::vector<int64_t> all_channels;
        for (int64_t ch = 0; ch < channels; ch++) all_channels.push_back(ch);
        stem = register_module("stem", std::make_shared<BranchEncoder>(all_channels, s.base_channels, s.negative_slope));
        for (int64_t l = 0; l < NB_LEVELS; l++) stem_values += int64_t(s.base_channels << l) * (PATCH >> l) * (PATCH >> l);
        token_values = stem_values + (v.include_raw ? raw_values : 0);
    }

    embed = register_module("embed", torch::nn::Conv2d(torch::nn::Conv2dOptions(token_values, dim, 1)));
    pos   = register_module("pos", torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 3).padding(1).groups(dim)));

    const int64_t shift = v.shifted_windows ? int64_t(v.window / 2) : 0;
    blocks = register_module("blocks", torch::nn::ModuleList());
    for (uint32_t i = 0; i < v.depth; i++) {
        auto block = std::make_shared<TransformerBlock>(dim, v.heads, v.mlp_ratio, v.window, (i % 2) ? shift : 0);
        blocks->push_back(block);                                             // "blocks.N"
        block_list.push_back(block);
    }
    norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim })));

    if (s.output == OutputHead::RGB) {
        head = register_module("head", torch::nn::Linear(dim, 3 * PATCH * PATCH));
    } else {
        kernel_heads = register_module("kernel_heads", torch::nn::ModuleList());
        for (int64_t l = 0; l < NB_LEVELS; l++) {
            const int64_t k = s.kernel_sizes[l];
            const int64_t factor = PATCH >> l;
            const bool with_blend = l < NB_LEVELS - 1;
            auto linear = torch::nn::Linear(dim, (k * k + (with_blend ? 1 : 0)) * factor * factor);
            kernel_heads->push_back(linear);                                  // "kernel_heads.L"
            kernel_head_list.push_back(linear);
        }
    }

    // rgb output: optional refine; kernel output: only as the residual
    if ((v.refine && s.output == OutputHead::RGB) || (s.output == OutputHead::Kernel && s.kernel_residual)) {
        const int64_t r = v.refine_channels;
        refine = register_module("refine", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3 + channels, r, 3).padding(1)),
            torch::nn::GELU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(r, 3, 3).padding(1))));
    }
}

torch::Tensor PyramidViT::forward(const std::vector<torch::Tensor>& pyramid_in, const torch::Tensor& prev_in)
{
    TORCH_CHECK(pyramid_in.size() == NB_LEVELS, "ViT '", settings.name, "' needs the ", NB_LEVELS, " rendered levels");
    TORCH_CHECK(pyramid_in[0].size(1) == settings.level_channels(),
                "ViT '", settings.name, "' expects ", settings.level_channels(), " channels per level");

    // Sizes that are not a multiple of 8 (other window sizes): edge-pad every
    // level to the padded grid, crop the output at the end
    const int64_t H = pyramid_in[0].size(-2), W = pyramid_in[0].size(-1);
    const int64_t Hp = (H + PATCH - 1) / PATCH * PATCH, Wp = (W + PATCH - 1) / PATCH * PATCH;
    auto pad_to = [](const torch::Tensor& t, int64_t h, int64_t w) {
        const int64_t ph = h - t.size(-2), pw = w - t.size(-1);
        if (ph <= 0 && pw <= 0) return t;
        return F::pad(t, F::PadFuncOptions({ 0, std::max<int64_t>(pw, 0), 0, std::max<int64_t>(ph, 0) })
                             .mode(torch::kReplicate));
    };
    std::vector<torch::Tensor> pyramid;
    for (int64_t l = 0; l < NB_LEVELS; l++) pyramid.push_back(pad_to(pyramid_in[l], Hp >> l, Wp >> l));

    // Network input per level: colour transform + previous frame (pooled per level)
    std::vector<torch::Tensor> levels;
    for (const auto& p : pyramid) {
        levels.push_back(settings.color_space == InputColorSpace::Lab ? rgb_to_lab(p) : p);
    }
    if (settings.prev_frame) {
        auto q = pad_to(prev_in, Hp, Wp).to(levels[0].dtype());
        for (size_t i = 0; i < levels.size(); i++) {
            if (i > 0) q = resize_to_match(pool->forward(q), levels[i]);
            levels[i] = torch::cat({ levels[i], q }, 1);
        }
    }

    // Tokens: one per coarsest-level pixel, every level unshuffled onto that grid
    const int64_t th = Hp / PATCH, tw = Wp / PATCH;
    auto to_tokens = [&](const std::vector<torch::Tensor>& maps, std::vector<torch::Tensor>& out) {
        for (size_t l = 0; l < maps.size(); l++) {
            const int64_t factor = PATCH >> l;
            auto t = factor > 1 ? torch::pixel_unshuffle(maps[l], factor) : maps[l];
            if (t.size(-2) != th || t.size(-1) != tw) {
                t = F::interpolate(t, F::InterpolateFuncOptions()
                                          .size(std::vector<int64_t>{ th, tw })
                                          .mode(torch::kBilinear).align_corners(false));
            }
            out.push_back(t);
        }
    };
    std::vector<torch::Tensor> tokens;
    if (!settings.vit.conv_stem || settings.vit.include_raw) to_tokens(levels, tokens);
    if (stem) {
        auto features = stem->forward(levels);
        std::vector<torch::Tensor> feature_list(features.begin(), features.end());
        to_tokens(feature_list, tokens);
    }
    auto x = embed->forward(torch::cat(tokens, 1));                            // (B, D, th, tw)
    x = x + pos->forward(x);

    x = x.permute({ 0, 2, 3, 1 }).contiguous();                               // (B, th, tw, D)
    for (auto& block : block_list) x = block->forward(x);
    x = norm->forward(x);

    torch::Tensor out;
    if (settings.output == OutputHead::RGB) {
        out = torch::pixel_shuffle(head->forward(x).permute({ 0, 3, 1, 2 }), PATCH);   // (B, 3, Hp, Wp)
        if (refine) out = out + refine->forward(torch::cat({ out, levels[0].to(out.dtype()) }, 1));
    } else {
        // Kernels on the (untransformed) input colours, coarse to fine
        for (int64_t l = NB_LEVELS - 1; l >= 0; l--) {
            const int64_t factor = PATCH >> l;
            const int64_t k = settings.kernel_sizes[l];
            auto colors = pyramid[l].slice(1, 0, 3);
            auto raw = kernel_head_list[l]->forward(x).permute({ 0, 3, 1, 2 });
            raw = (factor > 1 ? torch::pixel_shuffle(raw, factor) : raw).to(torch::kFloat32);
            raw = resize_to_match(raw, colors);
            auto filtered = apply_kernel(colors, raw.slice(1, 0, k * k), k);
            if (!out.defined()) {
                out = filtered;                                                // coarsest level
            } else {
                auto blend = torch::sigmoid(raw.slice(1, k * k, k * k + 1));
                out = blend * filtered + (1.0 - blend) * upsample_to(out, filtered);
            }
        }
        if (refine) {                                                          // kernel.residual
            auto correction = refine->forward(torch::cat({ out, levels[0].to(out.dtype()) }, 1));
            out = out + settings.kernel_residual_scale * torch::tanh(correction.to(torch::kFloat32));
        }
    }
    return out.slice(2, 0, H).slice(3, 0, W);
}


// =============================================================================
// Loading: weights + self-check against the Python reference output
// =============================================================================

static void copy_state_dict(const c10::impl::GenericDict& state, torch::nn::Module& dst)
{
    torch::NoGradGuard no_grad;

    auto params  = dst.named_parameters(/*recurse=*/true);
    auto buffers = dst.named_buffers(/*recurse=*/true);
    size_t copied_params = 0;

    for (const auto& item : state) {
        const std::string name = item.key().toStringRef();
        const torch::Tensor value = item.value().toTensor();

        torch::Tensor* t = params.find(name);
        const bool is_param = (t != nullptr);
        if (t == nullptr) t = buffers.find(name);
        if (t == nullptr) {
            throw std::runtime_error("Trained tensor '" + name + "' has no match in the C++ network");
        }
        if (t->sizes() != value.sizes()) {
            std::ostringstream ss;
            ss << "Shape mismatch for '" << name << "': C++ " << t->sizes() << " vs trained " << value.sizes();
            throw std::runtime_error(ss.str());
        }
        t->copy_(value);
        if (is_param) copied_params++;
    }

    if (copied_params != params.size()) {
        std::ostringstream ss;
        ss << "Only " << copied_params << " of " << params.size()
           << " C++ parameters were found in the model file";
        throw std::runtime_error(ss.str());
    }
    println("Copied {} parameter tensors", copied_params);
}

// Disables TF32 for its lifetime (restored even if an exception is thrown).
// On Ampere and newer GPUs, cuDNN convolutions use TF32 (10-bit mantissa) by
// default, which alone changes the output by ~1e-3 through this network.
struct FullFP32Scope {
    bool cudnn, cublas;
    FullFP32Scope()
        : cudnn(at::globalContext().allowTF32CuDNN()), cublas(at::globalContext().allowTF32CuBLAS())
    {
        at::globalContext().setAllowTF32CuDNN(false);
        at::globalContext().setAllowTF32CuBLAS(false);
    }
    ~FullFP32Scope()
    {
        at::globalContext().setAllowTF32CuDNN(cudnn);
        at::globalContext().setAllowTF32CuBLAS(cublas);
    }
};

static void verify_reference(const c10::impl::GenericDict& reference, FillNetwork& net)
{
    torch::NoGradGuard no_grad;
    FullFP32Scope full_fp32;
    auto to_cuda = [](const torch::IValue& v) { return v.toTensor().to(torch::kCUDA, torch::kFloat32); };

    std::vector<torch::Tensor> pyramid;
    auto levels = reference.at(torch::IValue("pyramid")).toList();
    for (size_t i = 0; i < levels.size(); i++) pyramid.push_back(to_cuda(levels.get(i)));
    auto prev     = to_cuda(reference.at(torch::IValue("prev")));
    auto expected = to_cuda(reference.at(torch::IValue("output")));

    auto output = net.forward(pyramid, prev);
    const float diff = (output - expected).abs().max().item<float>();
    println("C++ vs Python reference output: max abs diff {}", diff);

    // A wrong architecture or weight mapping gives differences of 0.1 - 1.
    // Up to a few 1e-3 is floating-point noise: reduced precision (TF32) on
    // either side, or different convolution algorithms.
    constexpr float EXACT     = 1e-3f;
    constexpr float TOLERATED = 2e-2f;
    if (diff > TOLERATED) {
        throw std::runtime_error("The C++ network does not reproduce the trained model (diff " +
                                 std::to_string(diff) + ")");
    }
    if (diff > EXACT) {
        println("  (accepted: floating-point precision differences, not a model mismatch)");
    }
}


// =============================================================================
// NeuralNet
// =============================================================================

void NeuralNet::load(const std::string& model_path)
{
    static_assert(CRenderingSettings::NB_PYRAMID_LEVELS == PointFillNet::NB_LEVELS,
                  "PointFillNet is built for exactly 4 pyramid levels");

    std::ifstream file(model_path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open model file '" + model_path + "'");
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    println("Loading model '{}' ({} bytes)", model_path, bytes.size());

    // Written by train.py export_model() with torch.save
    auto root = torch::pickle_load(bytes).toGenericDict();
    if (!root.contains(torch::IValue("format")) ||
        root.at(torch::IValue("format")).toStringRef() != "pointcloudfaker-model-v1") {
        throw std::runtime_error("'" + model_path + "' is not a model exported by train.py "
                                 "(python train.py <settings.toml> --export-only <checkpoint.pth>)");
    }

    ModelSettings s = ModelSettings::from_export(root.at(torch::IValue("settings")));
    println("Model {}", s.summary());

    // What the renderer can provide
    if (s.use_lod)          throw std::runtime_error("LOD input channels are not produced by the renderer");
    if (s.depth_rgb)        throw std::runtime_error("depth_mode = \"RGB\" is not produced by the renderer");
    if (s.prev_reprojected) throw std::runtime_error("prev_reprojected models need motion vectors (not implemented)");

    std::shared_ptr<FillNetwork> model;
    if (s.architecture == Architecture::ViT) model = std::make_shared<PyramidViT>(s);
    else                                     model = std::make_shared<PointFillNet>(s);
    copy_state_dict(root.at(torch::IValue("state_dict")).toGenericDict(), *model);
    model->to(torch::kCUDA);
    model->eval();
    verify_reference(root.at(torch::IValue("reference")).toGenericDict(), *model);

    settings = s;
    net = model;
    loaded_path = model_path;
    last_error.clear();
    reset_temporal_state();
    println("Model ready, renders {} pyramid level(s)", settings.input_levels());
}

std::vector<std::string> NeuralNet::list_models()
{
    std::vector<std::string> paths;
    std::error_code ec;
    if (!std::filesystem::is_directory(models_dir, ec)) return paths;
    for (const auto& entry : std::filesystem::directory_iterator(models_dir, ec)) {
        if (entry.is_regular_file(ec) && entry.path().extension() == ".pt") {
            paths.push_back(entry.path().string());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

void NeuralNet::unload()
{
    const bool had_model = (net != nullptr);
    net.reset();                     // last reference: parameters and buffers are released
    prev_output = torch::Tensor();
    settings = ModelSettings{};
    loaded_path.clear();

    // PyTorch keeps freed blocks in its caching allocator; hand them back to
    // the driver so the next model (and the renderer) can use that memory.
    if (torch::cuda::is_available()) {
        torch::cuda::synchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    if (had_model) println("Model unloaded");
}

bool NeuralNet::switch_model(const std::string& model_path)
{
    unload();                        // never two models in memory at once
    last_error.clear();
    if (model_path.empty()) return true;

    try {
        load(model_path);
        return true;
    } catch (const std::exception& e) {
        last_error = "'" + model_path + "': " + e.what();
        println("Failed to load model {}", last_error);
        unload();                    // drop whatever the failed load allocated
        return false;
    }
}

void NeuralNet::reset_temporal_state()
{
    prev_output = torch::Tensor();
}

bool NeuralNet::needs_rendered_pyramid()
{
    return net != nullptr && settings.input_levels() > 1;
}

void NeuralNet::sync_rendering_settings()
{
    if (GpuVersion::isTakingScreenshots) return;    // the screenshot code sets it itself
    CuRastSettings::useMultiScale = CuRastSettings::useNN && needs_rendered_pyramid();
}


void NeuralNet::infer(RenderTarget& target)
{
    if (!CuRastSettings::useNN || !net) return;

    // This frame was rendered without the lower levels the model reads (e.g. the
    // NN was just switched on): request them and skip this one frame.
    if (needs_rendered_pyramid() && !CuRastSettings::useMultiScale) {
        if (!GpuVersion::isTakingScreenshots) CuRastSettings::useMultiScale = true;
        return;
    }

    const uint32_t NB_LEVELS = settings.input_levels();     // levels the model reads
    uint32_t W0 = target.width;
    uint32_t H0 = target.height;
    const uint32_t C = settings.level_channels();

    // ---------------------------------------------------------------
    // 1. Per-level resolve into uint32 RGBA buffers
    // ---------------------------------------------------------------
    static CUdeviceptr d_resolved[CRenderingSettings::NB_PYRAMID_LEVELS] = {};
    static uint64_t    d_resolved_capacity[CRenderingSettings::NB_PYRAMID_LEVELS] = {};

    uint64_t level_cb_offsets_host[CRenderingSettings::NB_PYRAMID_LEVELS] = {};
    uint64_t level_tensor_offsets_host[CRenderingSettings::NB_PYRAMID_LEVELS] = {};
    {
        uint64_t cb_off = 0, tensor_off = 0;
        for (uint32_t level = 0; level < CRenderingSettings::NB_PYRAMID_LEVELS; level++) {
            uint64_t pixels = uint64_t(W0 >> level) * (H0 >> level);
            level_cb_offsets_host[level] = cb_off;               // colour buffers always hold 4 levels
            cb_off += pixels;
            if (level < NB_LEVELS) {
                level_tensor_offsets_host[level] = tensor_off;
                tensor_off += uint64_t(C) * pixels;
            }
        }
    }

    bool enableEDL = CuRastSettings::enableEDL;

    // Colour of empty pixels: what the model was trained with
    float hole_rgb[3] = { settings.hole_color[0], settings.hole_color[1], settings.hole_color[2] };
    if (settings.hole_color_is_background) {
        hole_rgb[0] = CuRastSettings::background.x;
        hole_rgb[1] = CuRastSettings::background.y;
        hole_rgb[2] = CuRastSettings::background.z;
    }
    uint32_t backgroundColor = 0;
    {
        uint8_t* bg = (uint8_t*)&backgroundColor;
        for (int i = 0; i < 3; i++) bg[i] = uint8_t(clamp(hole_rgb[i] * 255.0f + 0.5f, 0.0f, 255.0f));
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

        auto d_resolved_level = d_resolved[level];
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
    const uint64_t total_float_bytes = total_floats * sizeof(float);

    static float*   d_input = nullptr;
    static uint64_t d_input_capacity = 0;
    if (total_float_bytes > d_input_capacity) {
        if (d_input) cudaFree(d_input);
        cudaMalloc(&d_input, total_float_bytes);
        d_input_capacity = total_float_bytes;
    }

    const uint64_t output_float_bytes = 3ull * W0 * H0 * sizeof(float);
    static float*   d_output = nullptr;
    static uint64_t d_output_capacity = 0;
    if (output_float_bytes > d_output_capacity) {
        if (d_output) cudaFree(d_output);
        cudaMalloc(&d_output, output_float_bytes);
        d_output_capacity = output_float_bytes;
    }

    // ---------------------------------------------------------------
    // 3. Upload offset arrays
    // ---------------------------------------------------------------
    static CUdeviceptr d_level_cb_offsets = 0;
    static CUdeviceptr d_level_tensor_offsets = 0;
    if (d_level_cb_offsets == 0) {
        cuMemAlloc(&d_level_cb_offsets,     CRenderingSettings::NB_PYRAMID_LEVELS * sizeof(uint64_t));
        cuMemAlloc(&d_level_tensor_offsets, CRenderingSettings::NB_PYRAMID_LEVELS * sizeof(uint64_t));
    }
    cuMemcpyHtoDAsync(d_level_cb_offsets,     level_cb_offsets_host,     sizeof(level_cb_offsets_host),     0);
    cuMemcpyHtoDAsync(d_level_tensor_offsets, level_tensor_offsets_host, sizeof(level_tensor_offsets_host), 0);

    // ---------------------------------------------------------------
    // 4. Unpack resolved RGBA (+ log depth) into the planar float tensor
    //    (only the first NB_LEVELS levels)
    // ---------------------------------------------------------------
    static uint32_t flip_y = 1;      // dataset PNGs were written with stbi vertical flip
    {
        uint32_t block_size = 256;
        uint32_t grid_size  = (uint32_t(W0) * H0 + block_size - 1) / block_size;
        OptionalLaunchSettings launch_settings = { .gridsize = grid_size, .blocksize = block_size };

        // Levels the model does not read are never touched by the kernel
        auto d_r0 = d_resolved[0];
        auto d_r1 = NB_LEVELS > 1 ? d_resolved[1] : d_resolved[0];
        auto d_r2 = NB_LEVELS > 2 ? d_resolved[2] : d_resolved[0];
        auto d_r3 = NB_LEVELS > 3 ? d_resolved[3] : d_resolved[0];

        uint32_t nb_levels   = NB_LEVELS;
        uint32_t nb_channels = C;
        void* unpack_args[] = {
            &d_r0, &d_r1, &d_r2, &d_r3,
            &target.colorbuffer,
            &d_input,
            &W0, &H0,
            &nb_levels,
            &nb_channels,
            &flip_y,
            &d_level_cb_offsets,
            &d_level_tensor_offsets
        };
        GpuVersion::prog->launch("kernel_unpack_resolved_pyramid_to_tensor", unpack_args, launch_settings);
    }

    // ---------------------------------------------------------------
    // 5. Zero-copy per-level tensor views into d_input
    // ---------------------------------------------------------------
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    std::vector<torch::Tensor> pyramid;
    for (uint32_t level = 0; level < NB_LEVELS; level++) {
        int64_t w = W0 >> level;
        int64_t h = H0 >> level;
        pyramid.push_back(torch::from_blob(d_input + level_tensor_offsets_host[level], { 1, int64_t(C), h, w }, options));
    }

    // ---------------------------------------------------------------
    // 6. Previous frame (temporal models) + inference
    // ---------------------------------------------------------------
    torch::NoGradGuard no_grad;

    torch::Tensor prev;
    if (settings.prev_frame) {
        const bool have_prev = prev_output.defined() && prev_output.size(2) == H0 && prev_output.size(3) == W0;
        prev = have_prev
            ? torch::cat({ prev_output, torch::ones({ 1, 1, H0, W0 }, options) }, 1)   // validity = 1
            : torch::zeros({ 1, 4, H0, W0 }, options);                               // cold start
    } else {
        prev = torch::zeros({ 1, 1, 1, 1 }, options);                                // unused
    }

    // Linear / kernel outputs are unbounded: clamp like eval in Python
    torch::Tensor output = net->forward(pyramid, prev).clamp(0.0, 1.0);
    if (settings.prev_frame) {
        prev_output = output.detach().clone();                                       // [1, 3, H0, W0]
    }

    torch::Tensor output_squeezed = output.squeeze(0).contiguous();                   // [3, H0, W0]
    cudaMemcpy(d_output, output_squeezed.data_ptr<float>(), output_float_bytes, cudaMemcpyDeviceToDevice);

    // ---------------------------------------------------------------
    // 7. Pack model output into colorbuffer[level 0]
    // ---------------------------------------------------------------
    {
        uint64_t num_pixels = uint64_t(W0) * H0;
        uint32_t block_size = 256;
        uint32_t grid_size  = uint32_t((num_pixels + block_size - 1) / block_size);
        OptionalLaunchSettings launch_settings = { .gridsize = grid_size, .blocksize = block_size };

        void* pack_args[] = { &d_output, &target.colorbuffer, &W0, &H0, &flip_y };
        GpuVersion::prog->launch("kernel_pack_tensor_to_colorbuffer", pack_args, launch_settings);
    }
}