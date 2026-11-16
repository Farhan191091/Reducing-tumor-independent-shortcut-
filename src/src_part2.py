# %% [markdown]
# ## 2. Pre-processing, image cache and models

# %%
# ─── 2.1 Normalisation statistics (scratch models: per-fold training images) ───
def compute_fold_stats(fold, max_per_class=200, seed=SPLIT_SEED):
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for p, y in FOLDS[fold]["train"]:
        by_cls[y].append(p)
    s = s2 = n = 0.0
    for y in sorted(by_cls):
        for p in tqdm(rng.sample(by_cls[y], min(max_per_class, len(by_cls[y]))), desc=f"  fold {fold} stats, class {y}", leave=False):
            a = np.asarray(Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), np.float64) / 255.0
            s += a.sum(); s2 += (a ** 2).sum(); n += a.size
    mean = s / n
    return {"mean": [float(mean)] * 3, "std": [float(np.sqrt(max(s2 / n - mean ** 2, 1e-8)))] * 3}

fs_path = OUT_DIR / "fold_stats_3class.json"
FOLD_STATS = {int(k): v for k, v in (load_json(fs_path) or {}).items()}
if len(FOLD_STATS) != N_FOLDS:
    FOLD_STATS = {k: compute_fold_stats(k) for k in range(1, N_FOLDS + 1)}
    save_json(FOLD_STATS, fs_path)
for k, v in FOLD_STATS.items():
    print(f"fold {k}: mean {v['mean'][0]:.4f}  std {v['std'][0]:.4f}")

# %%
# ─── 2.2 Model registry ───
MODEL_SPECS = {
    # --- NEW (Run 5): solutions and controls ---
    "ROI-Loc":        dict(kind="roi", crop="box", loc=True),        # SOLUTION 1: crop + location numbers
    "CNX-Transplant": dict(kind="guide", seg=False, pool=False, cf=False, guide=None, transplant=True),  # SOLUTION 2
    "CNX-Strip":      dict(kind="guide", seg=False, pool=False, cf=False, guide=None, strip=True),       # CONTROL
    "ROI-Crop-m06":   dict(kind="roi", crop="box", margin=0.6),      # context sweep
    "ROI-Crop-m15":   dict(kind="roi", crop="box", margin=1.5),      # context sweep
    "ROI-m06-Transplant": dict(kind="roi", crop="box", margin=0.6, transplant=True),   # Run 6: context + transplant
    # --- NEW (Run 4): structural remedy. Same backbone; the input is only the region found by MTL-Seg's tumour map ---
    "ROI-Crop":     dict(kind="roi", crop="box"),      # box around the found tumour (+20% context)
    "ROI-Mask":     dict(kind="roi", crop="mask"),     # same box, everything outside the found tumour blacked out
    # --- NEW (Run 3): all on a pretrained ConvNeXt-Tiny backbone, same GPU training pipeline ("guide" kind) ---
    "CTR-Net":      dict(kind="guide", seg=True,  pool=True,  cf=True,  guide=None),     # PROPOSED
    "CTR-noCF":     dict(kind="guide", seg=True,  pool=True,  cf=False, guide=None),
    "CTR-noSeg":    dict(kind="guide", seg=False, pool=False, cf=True,  guide=None),
    "CNX-Plain":    dict(kind="guide", seg=False, pool=False, cf=False, guide=None),
    "MTL-Seg":      dict(kind="guide", seg=True,  pool=False, cf=False, guide=None),
    "RRR":          dict(kind="guide", seg=False, pool=False, cf=False, guide="rrr"),
    "CAM-Guide":    dict(kind="guide", seg=False, pool=False, cf=False, guide="cam"),
    "CTR-MambaKAN": dict(kind="tamtk", use_seg=True, use_seq=True, use_kan=True, seg=True, pool=True, cf=True, guide=None),
    # --- Run 1 models (scratch CNN / MTK family and ImageNet baselines) ---
    "MTK":               dict(kind="mtk", use_cnn=True,  use_seq=True,  use_kan=True),
    "MTK-noSeq":         dict(kind="mtk", use_cnn=True,  use_seq=False, use_kan=True),
    "MTK-noKAN":         dict(kind="mtk", use_cnn=True,  use_seq=True,  use_kan=False),
    "MTK-noCNN":         dict(kind="mtk", use_cnn=False, use_seq=True,  use_kan=True),
    "CNN-Backbone":      dict(kind="mtk", use_cnn=True,  use_seq=False, use_kan=False),
    "CNN-Backbone-Wide": dict(kind="mtk", use_cnn=True,  use_seq=False, use_kan=False, cnn_hidden=1536),
    "ResNet-50":         dict(kind="timm", arch="resnet50"),
    "EfficientNet-B0":   dict(kind="timm", arch="efficientnet_b0"),
    "ConvNeXt-Tiny":     dict(kind="timm", arch="convnext_tiny"),
    "DeiT-Small":        dict(kind="timm", arch="deit_small_patch16_224"),
}
MODEL_ORDER = list(MODEL_SPECS)
PROPOSED = "ROI-Loc"     # pre-registered main solution (falls back to ROI-Crop if not trained yet, see 3.4)
ABLATIONS = ["MTK-noSeq", "MTK-noKAN", "MTK-noCNN", "CNN-Backbone", "CNN-Backbone-Wide"]
CTR_FAMILY = ["CTR-Net", "CTR-noCF", "CTR-noSeg", "CTR-MambaKAN", "MTL-Seg", "RRR", "CAM-Guide", "CNX-Plain"]
TA_ABLATIONS = ["CTR-noCF", "CTR-noSeg", "CTR-MambaKAN", "MTL-Seg", "RRR", "CAM-Guide", "CNX-Plain"]
BASELINES = ["CNN-Backbone", "CNN-Backbone-Wide", "ResNet-50", "EfficientNet-B0", "ConvNeXt-Tiny", "DeiT-Small"]
FAMILY_COLOR = {"proposed": "#d1495b", "solution": "#2a9d8f", "control": "#6c757d", "roi": "#f08a8f", "ctr-ablation": "#9b5de5", "guidance": "#edae49",
                "pretrained": "#30638e", "scratch": "#888888"}
FAMILY_LABEL = {"proposed": "proposed solution", "solution": "Run 5 solutions", "control": "control (skull-stripped)",
                "roi": "ROI variants", "ctr-ablation": "CTR (Run 3 remedy) + ablations",
                "guidance": "existing guidance methods", "pretrained": "pretrained baselines", "scratch": "scratch / MTK models"}

def family_of(name):
    if name == PROPOSED: return "proposed"
    if name in ("ROI-Loc", "CNX-Transplant", "ROI-m06-Transplant"): return "solution"
    if name == "CNX-Strip": return "control"
    if name.startswith("ROI"): return "roi"
    if name.startswith("CTR"): return "ctr-ablation"
    if name in ("MTL-Seg", "RRR", "CAM-Guide"): return "guidance"
    if MODEL_SPECS[name]["kind"] in ("timm", "guide", "tamtk"): return "pretrained"
    return "scratch"

def new_pipeline(name):
    """Run-3 models: GPU augmentation, tumour masks and hidden versions available in every batch."""
    return MODEL_SPECS[name]["kind"] in ("guide", "tamtk") and "seg" in MODEL_SPECS[name]

def is_pretrained(name):
    return MODEL_SPECS[name]["kind"] in ("timm", "tamtk", "guide", "roi")

def is_roi(name):
    return MODEL_SPECS[name]["kind"] == "roi"

def uses_masks(name):
    s = MODEL_SPECS[name]
    return bool(s.get("seg") or s.get("cf") or s.get("guide"))

def norm_for(name, fold):
    if is_pretrained(name):
        return IMAGENET_MEAN, IMAGENET_STD
    return FOLD_STATS[fold]["mean"], FOLD_STATS[fold]["std"]

# %%
# ─── 2.3 MedTriMambaKAN v3 (fixed) ───
# Fixes vs the submitted v2:
#  * sequence pathway: the old CLS token (depthwise conv, kernel 5, two blocks) only
#    saw the first 4 tokens of the 14x14 grid (top-left corner). v3 uses a real
#    selective state-space scan (Mamba S6) in 4 directions over a 7x7 token grid
#    (all directions in one time loop) and mean-pools ALL tokens.
#  * KAN pathway: inputs are layer-normalised + tanh so they fall inside the
#    Gaussian RBF centres in [-1, 1] (otherwise the basis is ~0 and the layer is linear).
#  * pathways can be switched off -> ablations use exactly the same code.

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU())
    def forward(self, x):
        return self.block(x)

class MultiDirSSM(nn.Module):
    """Mamba (S6) selective scan in plain PyTorch, K scan directions processed TOGETHER
    (separate weights per direction, one shared time loop) -> K times fewer sequential steps."""
    def __init__(self, d, k_dirs=4, d_state=16, d_conv=3):
        super().__init__()
        self.K, self.d, self.N, self.r, self.kc = k_dirs, d, d_state, max(1, math.ceil(d / 16)), d_conv
        def w(*shape, fan_in):
            return nn.Parameter(torch.empty(*shape).uniform_(-1, 1) / math.sqrt(fan_in))
        self.W_in = w(k_dirs, d, 2 * d, fan_in=d)
        self.conv_w = w(k_dirs * d, 1, d_conv, fan_in=d_conv)
        self.conv_b = nn.Parameter(torch.zeros(k_dirs * d))
        self.W_x = w(k_dirs, d, self.r + 2 * d_state, fan_in=d)
        self.W_dt = w(k_dirs, self.r, d, fan_in=self.r)
        dt = torch.exp(torch.rand(k_dirs, d) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.b_dt = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))          # standard Mamba dt init
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(k_dirs, d, 1))
        self.D = nn.Parameter(torch.ones(k_dirs, d))
        self.W_out = w(k_dirs, d, d, fan_in=d)

    def forward(self, xs):                                   # xs: (B, K, L, d)
        with torch.autocast(device_type=xs.device.type, enabled=False):
            xs = xs.float(); B, K, L, d = xs.shape
            u, z = torch.einsum("bkld,kde->bkle", xs, self.W_in).chunk(2, dim=-1)
            u = F.conv1d(u.permute(0, 1, 3, 2).reshape(B, K * d, L), self.conv_w, self.conv_b,
                         padding=self.kc - 1, groups=K * d)[..., :L]
            u = F.silu(u.reshape(B, K, d, L).permute(0, 1, 3, 2))
            dt, Bm, Cm = torch.einsum("bkld,kde->bkle", u, self.W_x).split([self.r, self.N, self.N], dim=-1)
            dt = F.softplus(torch.einsum("bklr,krd->bkld", dt, self.W_dt) + self.b_dt[None, :, None])
            dA = torch.exp(dt.unsqueeze(-1) * -torch.exp(self.A_log)[None, :, None])   # (B,K,L,d,N)
            dBu = dt.unsqueeze(-1) * Bm.unsqueeze(3) * u.unsqueeze(-1)
            h = xs.new_zeros(B, K, d, self.N); ys = []
            for t in range(L):
                h = dA[:, :, t] * h + dBu[:, :, t]
                ys.append((h * Cm[:, :, t].unsqueeze(2)).sum(-1))
            y = (torch.stack(ys, 2) + u * self.D[None, :, None]) * F.silu(z)
            return torch.einsum("bkld,kde->bkle", y, self.W_out)

class CrossScanBlock(nn.Module):
    """Pre-norm residual block scanning an HxW token grid in 4 orders (rows fwd/bwd, cols fwd/bwd)."""
    def __init__(self, d, hw):
        super().__init__()
        self.hw = hw
        self.norm = nn.LayerNorm(d)
        self.ssm = MultiDirSSM(d, k_dirs=4)
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):                                    # x: (B, h*w, d)
        h = w = self.hw; B, L, d = x.shape
        xn = self.norm(x)
        col = xn.view(B, h, w, d).transpose(1, 2).reshape(B, L, d)
        ys = self.ssm(torch.stack([xn, xn.flip(1), col, col.flip(1)], dim=1))
        yc = ys[:, 2] + ys[:, 3].flip(1)
        y = ys[:, 0] + ys[:, 1].flip(1) + yc.view(B, w, h, d).transpose(1, 2).reshape(B, L, d)
        x = x + y / 4
        return x + self.ffn(x)

class GaussianKAN(nn.Module):
    """KAN-inspired layer: SiLU base + learnable weights on 5 Gaussian RBFs (centres in [-1, 1])."""
    def __init__(self, din, dout, n_basis=5, width=0.5):
        super().__init__()
        self.base = nn.Linear(din, dout, bias=False)
        self.spline = nn.Parameter(torch.randn(dout, din, n_basis) * 0.1)
        self.register_buffer("centres", torch.linspace(-1, 1, n_basis))
        self.width = width
        self.norm = nn.LayerNorm(dout)
    def forward(self, x):
        basis = torch.exp(-((x.unsqueeze(-1) - self.centres) ** 2) / self.width)
        return self.norm(self.base(F.silu(x)) + torch.einsum("big,oig->bo", basis, self.spline))

class MedTriMambaKAN(nn.Module):
    def __init__(self, num_classes=N_CLASSES, use_cnn=True, use_seq=True, use_kan=True,
                 cnn_hidden=None, fused_dim=128, seq_depth=1):
        super().__init__()
        assert use_cnn or use_seq or use_kan
        self.use = {"cnn": use_cnn, "seq": use_seq, "kan": use_kan}
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.GELU())
        self.stage1, self.stage2 = ConvBlock(32, 64, 2), ConvBlock(64, 128, 2)
        self.stage3, self.stage4 = ConvBlock(128, 192, 2), ConvBlock(192, 256, 2)
        dims = {}
        if use_cnn:
            layers = ([nn.Linear(256, cnn_hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(cnn_hidden, 128)]
                      if cnn_hidden else [nn.Linear(256, 128)])
            self.cnn_head = nn.Sequential(*layers, nn.GELU(), nn.Dropout(0.2)); dims["cnn"] = 128
        if use_seq:
            # 14x14 -> 7x7 tokens (2x2 average pooling): 49 scan steps instead of 196
            self.seq_reduce = nn.Sequential(nn.Conv2d(256, 128, 1, bias=False), nn.BatchNorm2d(128), nn.AvgPool2d(2))
            self.seq_blocks = nn.Sequential(*[CrossScanBlock(128, 7) for _ in range(seq_depth)])
            self.seq_norm = nn.LayerNorm(128); dims["seq"] = 128
        if use_kan:
            self.kan_in = nn.LayerNorm(128)
            self.kan = nn.Sequential(GaussianKAN(128, 128), GaussianKAN(128, 64)); dims["kan"] = 64
        self.paths = list(dims)
        self.proj = nn.ModuleDict({k: nn.Linear(v, fused_dim) for k, v in dims.items()})
        if len(self.paths) > 1:
            self.gate = nn.Sequential(nn.Linear(fused_dim * len(self.paths), fused_dim), nn.GELU(),
                                      nn.Linear(fused_dim, len(self.paths)))
        self.fuse_norm, self.fuse_drop = nn.LayerNorm(fused_dim), nn.Dropout(0.25)
        if use_cnn:
            self.skip_alpha = nn.Parameter(torch.tensor(0.1))
        self.classifier = nn.Sequential(nn.Linear(fused_dim, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def cam_layer(self):
        return self.stage4.block[3]          # last conv of stage 4 (14x14)

    def forward(self, x, return_gate=False, return_embed=False):
        x2 = self.stage2(self.stage1(self.stem(x)))
        x4 = self.stage4(self.stage3(x2))
        feats = {}
        if self.use["cnn"]:
            feats["cnn"] = self.cnn_head(x4.mean((2, 3)))
        if self.use["seq"]:
            tok = self.seq_reduce(x4).flatten(2).transpose(1, 2)       # (B, 49, 128)
            feats["seq"] = self.seq_norm(self.seq_blocks(tok)).mean(1)
        if self.use["kan"]:
            feats["kan"] = self.kan(torch.tanh(self.kan_in(x2.mean((2, 3)))))
        p = [self.proj[k](feats[k]) for k in self.paths]
        if len(p) > 1:
            g = torch.softmax(self.gate(torch.cat(p, 1)), dim=1)
            fused = sum(g[:, i:i + 1] * p[i] for i in range(len(p)))
        else:
            g = torch.ones(x.shape[0], 1, device=x.device); fused = p[0]
        fused = self.fuse_drop(self.fuse_norm(fused))
        if self.use["cnn"]:
            fused = fused + self.skip_alpha * feats["cnn"]
        if return_embed:
            return fused
        logits = self.classifier(fused)
        return (logits, g) if return_gate else logits

class TumourAwareMTK(nn.Module):
    """TA-MTK: pretrained ConvNeXt-Tiny features -> three pathways (CNN, Seq = 4-direction Mamba scan, KAN)
    fused by a softmax gate (same design as MTK), plus a light tumour-mask decoder.
    The predicted tumour map is used for MASK-GUIDED POOLING: every pathway pools
    0.5 x (global average) + 0.5 x (average inside the predicted tumour), so the class decision
    is pushed to come from the tumour region. Without the mask head (use_seg=False) all pooling is global."""
    def __init__(self, num_classes=N_CLASSES, use_seg=True, use_seq=True, use_kan=True, pretrained=True, fused_dim=128):
        super().__init__()
        self.use_seg = use_seg
        self.backbone = timm.create_model("convnext_tiny", pretrained=pretrained, features_only=True, out_indices=(1, 2, 3))
        c1, c2, c3 = self.backbone.feature_info.channels()          # 192 @28x28, 384 @14x14, 768 @7x7
        self.cam_id = nn.Identity()                                  # Grad-CAM++ hook (last stage, 7x7, as ConvNeXt-Tiny)
        self.cam_id2 = nn.Identity()                                 # 14x14 features (LayerCAM explanations)
        if use_seg:
            self.lat3, self.lat2, self.lat1 = nn.Conv2d(c3, 128, 1), nn.Conv2d(c2, 128, 1), nn.Conv2d(c1, 128, 1)
            self.seg_head = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
                                          nn.Conv2d(64, 1, 1))
        dims = {"cnn": 128}
        self.cnn_head = nn.Sequential(nn.Linear(c3, 128), nn.GELU(), nn.Dropout(0.2))
        if use_seq:
            self.seq_reduce = nn.Sequential(nn.Conv2d(c3, 128, 1, bias=False), nn.BatchNorm2d(128))
            self.seq_blocks = CrossScanBlock(128, 7)
            self.seq_norm = nn.LayerNorm(128); dims["seq"] = 128
        if use_kan:
            self.kan_in = nn.Sequential(nn.Linear(c2, 128), nn.LayerNorm(128))
            self.kan = nn.Sequential(GaussianKAN(128, 128), GaussianKAN(128, 64)); dims["kan"] = 64
        self.paths = list(dims)
        self.proj = nn.ModuleDict({k: nn.Linear(v, fused_dim) for k, v in dims.items()})
        if len(self.paths) > 1:
            self.gate = nn.Sequential(nn.Linear(fused_dim * len(self.paths), fused_dim), nn.GELU(),
                                      nn.Linear(fused_dim, len(self.paths)))
        self.fuse_norm, self.fuse_drop = nn.LayerNorm(fused_dim), nn.Dropout(0.25)
        self.skip_alpha = nn.Parameter(torch.tensor(0.1))
        self.classifier = nn.Sequential(nn.Linear(fused_dim, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def cam_layer(self):
        return self.cam_id

    def param_groups(self, lr):
        bb = list(self.backbone.parameters()); ids = {id(q) for q in bb}
        heads = [q for q in self.parameters() if id(q) not in ids]
        return [{"params": bb, "base_lr": lr}, {"params": heads, "base_lr": lr * HEAD_LR_MULT}]

    @staticmethod
    def _pool(f, w):                      # f (B,C,H,W), w (B,1,H,W) in [0,1] or None
        g = f.mean((2, 3))
        if w is None:
            return g
        return 0.5 * g + 0.5 * (f * w).sum((2, 3)) / (w.sum((2, 3)) + 1e-4)

    def forward(self, x, return_gate=False, return_seg=False, return_embed=False):
        f1, f2, f3 = self.backbone(x)
        f2 = self.cam_id2(f2); f3 = self.cam_id(f3)
        seg = w3 = w2 = None
        if self.use_seg:
            d = self.lat3(f3)
            d = self.lat2(f2) + F.interpolate(d, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            d = self.lat1(f1) + F.interpolate(d, size=f1.shape[-2:], mode="bilinear", align_corners=False)
            seg = self.seg_head(d)                                    # (B,1,28,28) tumour logits
            prob = torch.sigmoid(seg.float()).to(f3.dtype)
            w3, w2 = F.adaptive_avg_pool2d(prob, f3.shape[-2:]), F.adaptive_avg_pool2d(prob, f2.shape[-2:])
        feats = {"cnn": self.cnn_head(self._pool(f3, w3))}
        if "seq" in self.paths:
            t = self.seq_reduce(f3); B, C, H, W = t.shape
            t = self.seq_norm(self.seq_blocks(t.flatten(2).transpose(1, 2))).transpose(1, 2).reshape(B, C, H, W)
            feats["seq"] = self._pool(t, w3)
        if "kan" in self.paths:
            feats["kan"] = self.kan(torch.tanh(self.kan_in(self._pool(f2, w2))))
        p = [self.proj[k](feats[k]) for k in self.paths]
        if len(p) > 1:
            g = torch.softmax(self.gate(torch.cat(p, 1)), dim=1)
            fused = sum(g[:, i:i + 1] * p[i] for i in range(len(p)))
        else:
            g = torch.ones(x.shape[0], 1, device=x.device); fused = p[0]
        fused = self.fuse_drop(self.fuse_norm(fused)) + self.skip_alpha * feats["cnn"]
        if return_embed:
            return fused
        logits = self.classifier(fused)
        if return_seg:
            return logits, seg
        return (logits, g) if return_gate else logits

class GuidedConvNeXt(nn.Module):
    """ConvNeXt-Tiny backbone + linear head, with OPTIONAL tumour-mask decoder (seg) and
    mask-guided pooling (pool). Used for CTR-Net, its ablations and the guidance baselines."""
    def __init__(self, num_classes=N_CLASSES, seg=False, pool=False, pretrained=True):
        super().__init__()
        self.use_seg, self.use_pool = seg, pool and seg
        self.backbone = timm.create_model("convnext_tiny", pretrained=pretrained, features_only=True, out_indices=(1, 2, 3))
        c1, c2, c3 = self.backbone.feature_info.channels()           # 192 @28, 384 @14, 768 @7
        self.cam_id = nn.Identity()                                   # 7x7 features  (CAM-Guide penalty)
        self.cam_id2 = nn.Identity()                                  # 14x14 features (LayerCAM explanations)
        if seg:
            self.lat3, self.lat2, self.lat1 = nn.Conv2d(c3, 128, 1), nn.Conv2d(c2, 128, 1), nn.Conv2d(c1, 128, 1)
            self.seg_head = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
                                          nn.Conv2d(64, 1, 1))
        self.head_norm = nn.LayerNorm(c3)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c3, num_classes))

    def cam_layer(self):
        return self.cam_id2

    def param_groups(self, lr):
        bb = list(self.backbone.parameters()); ids = {id(q) for q in bb}
        return [{"params": bb, "base_lr": lr},
                {"params": [q for q in self.parameters() if id(q) not in ids], "base_lr": lr * HEAD_LR_MULT}]

    def forward(self, x, return_seg=False, return_feat=False, return_embed=False):
        f1, f2, f3 = self.backbone(x)
        f2 = self.cam_id2(f2); f3 = self.cam_id(f3)
        seg = None
        if self.use_seg:
            d = self.lat3(f3)
            d = self.lat2(f2) + F.interpolate(d, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            d = self.lat1(f1) + F.interpolate(d, size=f1.shape[-2:], mode="bilinear", align_corners=False)
            seg = self.seg_head(d)                                    # (B,1,28,28) tumour logits
        g = f3.mean((2, 3))
        if self.use_pool:                                             # 0.5 x global + 0.5 x inside predicted tumour
            w = F.adaptive_avg_pool2d(torch.sigmoid(seg.float()).to(f3.dtype), f3.shape[-2:])
            g = 0.5 * g + 0.5 * (f3 * w).sum((2, 3)) / (w.sum((2, 3)) + 1e-4)
        emb = self.head_norm(g)
        if return_embed:
            return emb
        logits = self.head(emb)
        if return_feat:
            return logits, f3
        if return_seg:
            return logits, seg
        return logits

N_LOC = 6   # location numbers of ROI-Loc (see loc_feats in 2.4)

class RoiLocNet(nn.Module):
    """SOLUTION 1. Two inputs that a radiologist would also use, and nothing else:
    (a) what the tumour looks like = ConvNeXt-Tiny on the ROI crop, (b) WHERE it is relative to the brain =
    6 numbers (vertical position, distance from the midline, distance from the centre, relative area, width, height).
    The whole slice (skull, slice position, plane) is never seen, so it cannot be used as a shortcut."""
    def __init__(self, num_classes=N_CLASSES, pretrained=True):
        super().__init__()
        self.cnn = GuidedConvNeXt(num_classes, seg=False, pool=False, pretrained=pretrained)
        self.loc = nn.Sequential(nn.Linear(N_LOC, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(768 + 32, num_classes))
        self.loc_override = None          # set to a (N_LOC,) tensor to replace every location (knock-out test)
    def param_groups(self, lr):
        bb = list(self.cnn.backbone.parameters()); ids = {id(q) for q in bb}
        return [{"params": bb, "base_lr": lr},
                {"params": [q for q in self.parameters() if id(q) not in ids], "base_lr": lr * HEAD_LR_MULT}]
    def cam_layer(self):
        return self.cnn.cam_id2
    def forward(self, x, loc=None, return_embed=False):
        emb = self.cnn(x, return_embed=True)
        if loc is None:
            loc = torch.zeros(x.shape[0], N_LOC, device=x.device, dtype=emb.dtype)
        if self.loc_override is not None:
            loc = self.loc_override.to(x.device, emb.dtype).expand(x.shape[0], -1)
        z = torch.cat([emb, self.loc(loc.to(emb.dtype))], 1)
        return z if return_embed else self.head(z)

def build_model(name, pretrained=True, num_classes=N_CLASSES):
    spec = MODEL_SPECS[name]
    if spec["kind"] == "roi" and spec.get("loc"):
        return RoiLocNet(num_classes, pretrained=pretrained)
    if spec["kind"] == "guide":
        return GuidedConvNeXt(num_classes, spec["seg"], spec["pool"], pretrained=pretrained)
    if spec["kind"] == "roi":
        return GuidedConvNeXt(num_classes, seg=False, pool=False, pretrained=pretrained)
    if spec["kind"] == "mtk":
        return MedTriMambaKAN(num_classes, spec["use_cnn"], spec["use_seq"], spec["use_kan"], spec.get("cnn_hidden"))
    if spec["kind"] == "tamtk":
        return TumourAwareMTK(num_classes, spec["use_seg"], spec["use_seq"], spec["use_kan"], pretrained=pretrained)
    return timm.create_model(spec["arch"], pretrained=pretrained, num_classes=num_classes)

def embed(model, x):
    """Penultimate feature vector (used for the t-SNE figure)."""
    m = model.module if isinstance(model, nn.DataParallel) else model
    if isinstance(m, (MedTriMambaKAN, TumourAwareMTK, GuidedConvNeXt)):
        return m(x, return_embed=True)
    return m.forward_head(m.forward_features(x), pre_logits=True)

def cam_target_layer(model, name):
    m = model.module if isinstance(model, nn.DataParallel) else model
    # all explanation layers have a 14x14 grid (stride 16) -> sharper and comparable across models
    if MODEL_SPECS[name]["kind"] in ("mtk", "guide", "roi"):
        return m.cam_layer()
    if MODEL_SPECS[name]["kind"] == "tamtk":
        return m.cam_id2
    arch = MODEL_SPECS[name]["arch"]
    if arch.startswith("resnet"):       return m.layer3[-1]
    if arch.startswith("efficientnet"): return m.blocks[4]
    if arch.startswith("convnext"):     return m.stages[2]
    return None                          # DeiT: token maps -> not used for heat maps

# Parameter / FLOP / latency table
banner("2.3 Model size / speed table", "parameters, compute (GMACs) and time per image, ~1 min")
rows = []
for name in MODEL_ORDER:
    m = build_model(name, pretrained=False).to(DEVICE).eval()
    n_par = sum(p.numel() for p in m.parameters())
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    with torch.no_grad():
        for _ in range(3): m(x)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10): m(x)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
    gflops = np.nan
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(x)
        gflops = fc.get_total_flops() / 2e9          # multiply-adds -> GMACs
    except Exception:
        pass
    rows.append({"Model": name, "Params (M)": round(n_par / 1e6, 3), "GMACs": round(gflops, 2),
                 "ms/image (bs=1)": round((time.perf_counter() - t0) / 10 * 1000, 1)})
    del m
df_complexity = pd.DataFrame(rows)
print(df_complexity.to_string(index=False))
df_complexity.to_csv(OUT_DIR / "tables" / "table_complexity.csv", index=False)
if DEVICE.type == "cuda": torch.cuda.empty_cache()

# %%
# ─── 2.4 Datasets, training transforms, and a fast evaluation cache ───
def train_transform(mean, std):
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.Grayscale(num_output_channels=3),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(15),
        T.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomAutocontrast(p=0.3),
        T.GaussianBlur(3, sigma=(0.1, 1.0)),
        T.ToTensor(), T.Normalize(mean, std)])


_U8_CACHE = {}
def load_u8(path):
    """Grayscale, bicubic 224x224, uint8 - identical to Resize -> Grayscale -> ToTensor (x255)."""
    a = _U8_CACHE.get(path)
    if a is None:
        with Image.open(path) as im:
            a = np.asarray(im.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), np.uint8)
        _U8_CACHE[path] = a
    return a

def load_mask(path):
    return np.asarray(Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127

class ImageList(Dataset):
    """Images are read from disk ONCE (224x224 uint8, ~50 KB each) and kept in RAM.
    Same pixels as before (grayscale -> bicubic 224), but no disk reading every epoch."""
    def __init__(self, items, tf):
        self.items, self.tf = list(items), tf
        self.arrs = [load_u8(p) for p, _ in self.items]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.arrs[i], mode="L")), int(self.items[i][1])

from torchvision.transforms import v2 as T2
from torchvision import tv_tensors

class ImageMaskList(Dataset):
    """Same augmentation as train_transform, but flips / rotation / affine are applied to the
    image AND its tumour mask together; brightness / contrast / blur only to the image."""
    def __init__(self, items, mean, std):
        self.items = list(items)
        self.arrs = [load_u8(p) for p, _ in self.items]
        self.masks = [MASKS.get(p) for p, _ in self.items]
        self.geo = T2.Compose([T2.RandomHorizontalFlip(0.5), T2.RandomRotation(15),
                               T2.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5)])
        self.photo = T2.Compose([T2.ColorJitter(brightness=0.2, contrast=0.2), T2.RandomAutocontrast(p=0.3),
                                 T2.GaussianBlur(3, sigma=(0.1, 1.0))])
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        img = tv_tensors.Image(torch.from_numpy(self.arrs[i])[None])
        has = self.masks[i] is not None
        mk = tv_tensors.Mask(torch.from_numpy(self.masks[i].astype(np.uint8))[None] if has
                             else torch.zeros(1, IMG_SIZE, IMG_SIZE, dtype=torch.uint8))
        img, mk = self.geo(img, mk)
        img = self.photo(img)
        x = (img.float().div(255).repeat(3, 1, 1) - self.mean) / self.std
        return x, mk.float(), torch.tensor(float(has)), int(self.items[i][1])

class GuidedSet(Dataset):
    """Run-3 pipeline: returns uint8 (5, H, W) = [image, tumour hidden (inpaint), random spot hidden,
    tumour mask, tumour region (mask + margin)], has_mask, label. All augmentation happens on the GPU,
    identically for the 5 channels (so the hidden versions differ from the image ONLY in the hidden region)."""
    def __init__(self, items, strip=False):
        self.items, self.strip = list(items), strip
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        img = load_u8(p); h = HIDE.get(p)
        if self.strip:                      # CNX-Strip: the model only ever sees skull-stripped images
            img = skull_strip(img); h = None
        if h is not None:
            st = np.stack([img, h["inpaint"], h["random"], MASKS[p].astype(np.uint8) * 255, h["region"].astype(np.uint8) * 255])
            has = 1.0
        else:
            z = np.zeros_like(img); st = np.stack([img, img, img, z, z]); has = 0.0
        return torch.from_numpy(st), torch.tensor(has), int(y)

def gpu_augment(b):
    """b: uint8 (B,5,H,W) on GPU -> images (B,3,H,W) in [0,1] [orig, hidden, random-hidden], masks (B,2,H,W).
    Same augmentation family as train_transform: flip, rotation 15, translate 5%, scale 0.9-1.1, shear 5,
    brightness/contrast 0.2, autocontrast p=0.3, Gaussian blur sigma 0.1-1.0."""
    x = b.float() / 255; B = x.shape[0]; dev = x.device
    U = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi)
    ang, sh = torch.deg2rad(U(-15, 15)), torch.deg2rad(U(-5, 5))
    sc, tx, ty = U(0.9, 1.1), U(-0.1, 0.1), U(-0.1, 0.1)
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = cos / sc * flip; theta[:, 0, 1] = (-sin + torch.tan(sh) * cos) / sc
    theta[:, 1, 0] = sin / sc * flip; theta[:, 1, 1] = (cos + torch.tan(sh) * sin) / sc
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    imgs = F.grid_sample(x[:, :3], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    masks = (F.grid_sample(x[:, 3:], grid, mode="nearest", padding_mode="zeros", align_corners=False) > 0.5).float()
    bri, con = U(0.8, 1.2).view(B, 1, 1, 1), U(0.8, 1.2).view(B, 1, 1, 1)
    imgs = imgs * bri
    mu = imgs[:, :1].mean((2, 3), keepdim=True)
    imgs = ((imgs - mu) * con + mu).clamp(0, 1)
    ac = (torch.rand(B, device=dev) < 0.3).view(B, 1, 1, 1)
    lo = imgs[:, :1].amin((2, 3), keepdim=True); hi = imgs[:, :1].amax((2, 3), keepdim=True)
    imgs = torch.where(ac, ((imgs - lo) / (hi - lo + 1e-6)).clamp(0, 1), imgs)
    sig = U(0.1, 1.0)
    k1 = torch.exp(-(torch.tensor([-1.0, 0.0, 1.0], device=dev)[None] ** 2) / (2 * sig[:, None] ** 2))
    k1 = k1 / k1.sum(1, keepdim=True); k2 = (k1[:, :, None] * k1[:, None, :])          # (B,3,3)
    k2 = k2.repeat_interleave(3, 0)[:, None]                                            # (B*3,1,3,3)
    imgs = F.conv2d(F.pad(imgs.reshape(1, B * 3, *imgs.shape[-2:]), (1, 1, 1, 1), mode="reflect"),
                    k2, groups=B * 3).reshape(B, 3, *imgs.shape[-2:])
    return imgs, masks

def norm3(x1, mean, std):
    """(B,1,H,W) in [0,1] -> normalised 3-channel input."""
    m = torch.tensor([float(v) for v in mean], device=x1.device).view(1, 3, 1, 1)
    s = torch.tensor([float(v) for v in std], device=x1.device).view(1, 3, 1, 1)
    return (x1.repeat(1, 3, 1, 1) - m) / s

# ─── ROI remedy: crop the region of the (predicted or expert) tumour ───
def box_from_mask(mask, min_pixels=ROI_MIN_PIXELS):
    """Largest connected region of a binary mask -> (x, y, w, h, component) or None (= no tumour found)."""
    m = np.asarray(mask, np.uint8)
    if m.sum() < min_pixels:
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[k, cv2.CC_STAT_AREA] < min_pixels:
        return None
    x, y, w, h = (int(v) for v in st[k, :4])
    return x, y, w, h, lab == k

def crop_roi(img, box, mode, margin=ROI_MARGIN, shift=(0.0, 0.0)):
    """Square crop around the box (+margin on each side), resized to 224. mode 'mask': outside the region = 0."""
    x, y, w, h, comp = box
    if mode == "mask":
        keep = cv2.dilate(comp.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        img = np.where(keep, img, 0).astype(np.uint8)
    s = max(max(w, h) * (1 + 2 * margin), 32)
    cx, cy = x + w / 2 + shift[0] * s, y + h / 2 + shift[1] * s
    im = Image.fromarray(img).crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)))
    return np.asarray(im.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), np.uint8)

class RoiSet(Dataset):
    """Training crops from the EXPERT mask with random margin / shift (the model must cope with an imperfect
    box at test time, where the box comes from the predicted tumour map). margin = the model's box margin
    (training draws it from 0.5x-1.75x around that value, as in Run 4: 0.2 -> 0.1-0.35).
    loc=True (ROI-Loc): also returns the 6 location numbers (+ small noise, as predicted maps are imperfect)."""
    def __init__(self, items, tf, mode, margin=ROI_MARGIN, loc=False, transplant=False):
        self.items, self.tf, self.mode, self.margin, self.loc = list(items), tf, mode, margin, loc
        self.arrs = [load_u8(p) for p, _ in self.items]
        self.boxes = [box_from_mask(MASKS[p]) if p in MASKS else None for p, _ in self.items]
        self.locs = [loc_feats(a, MASKS[p]) if (loc and p in MASKS) else np.zeros(N_LOC, np.float32)
                     for a, (p, _) in zip(self.arrs, self.items)]
        self.transplant = transplant
        if transplant:     # Run 6: the context around the crop comes from a patient of ANOTHER class
            self.ok = [i for i, (p, _) in enumerate(self.items) if p in HIDE]
            self.by_cls = {k: [i for i in self.ok if self.items[i][1] == k] for k in range(N_CLASSES)}
            self.bm = {self.items[i][0]: brain_bm(self.arrs[i], self.items[i][0]) for i in self.ok}
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        img, b = self.arrs[i], self.boxes[i]
        r = np.random.default_rng()
        p, y = self.items[i]
        if self.transplant and p in HIDE and r.random() < TRANSPLANT_PROB:
            others = [k for k in range(N_CLASSES) if k != y and self.by_cls[k]]
            k = int(r.choice(others)); j = int(r.choice(self.by_cls[k])); q = self.items[j][0]
            res = transplant(img, HIDE[p]["inpaint"], self.bm[p], self.arrs[j], MASKS[q], self.bm[q], r)
            bt = box_from_mask(res[1]) if res is not None else None
            if bt is not None:
                crop = crop_roi(res[0], bt, self.mode, margin=r.uniform(0.5 * self.margin, 1.75 * self.margin),
                                shift=tuple(r.uniform(-0.1, 0.1, 2)))
                return self.tf(Image.fromarray(crop, mode="L")), k
        if b is not None:
            img = crop_roi(img, b, self.mode, margin=r.uniform(0.5 * self.margin, 1.75 * self.margin),
                           shift=tuple(r.uniform(-0.1, 0.1, 2)))
        x = self.tf(Image.fromarray(img, mode="L"))
        if self.loc:
            l = self.locs[i] + r.normal(0, 0.03, N_LOC).astype(np.float32)
            return x, torch.from_numpy(l.astype(np.float32)), int(self.items[i][1])
        return x, int(self.items[i][1])

# ─── Run 5 helpers: location numbers, skull stripping, tumour transplant ───
_BM_CACHE = {}
def brain_bm(img, key=None):
    """Head/brain mask (Otsu, cell 1.8), cached by path when a key is given."""
    if key is not None and key in _BM_CACHE:
        return _BM_CACHE[key]
    bm = brain_mask(img)
    if key is not None:
        _BM_CACHE[key] = bm
    return bm

def loc_feats(img, mask, bm=None):
    """6 numbers describing WHERE the tumour is relative to the brain outline (no image content):
    vertical position (-1 top .. +1 bottom), distance from the left-right midline (0..1), distance from the brain
    centre, log relative area, relative width, relative height. Left/right is deliberately NOT used."""
    mk = np.asarray(mask, bool)
    bm = brain_mask(img) if bm is None else bm
    if not mk.any() or not bm.any():
        return np.zeros(N_LOC, np.float32)
    ys, xs = np.where(mk); bys, bxs = np.where(bm)
    by0, bx0 = bys.min(), bxs.min(); bh = max(bys.max() - by0, 1); bw = max(bxs.max() - bx0, 1)
    vy = 2 * (ys.mean() - by0) / bh - 1; vx = 2 * (xs.mean() - bx0) / bw - 1
    return np.array([vy, abs(vx), math.sqrt(vy ** 2 + vx ** 2),
                     math.log(mk.sum() / max(bm.sum(), 1) + 1e-4) / 5 + 1,
                     (xs.max() - xs.min() + 1) / bw, (ys.max() - ys.min() + 1) / bh], np.float32)

def skull_strip(img):
    """Approximate skull stripping for 2D slices: keep only the head mask eroded by STRIP_ERODE pixels
    (removes scalp + skull ring). Applied to every input of CNX-Strip EXCEPT BraTS (already skull-stripped)."""
    bm = brain_mask(img)
    keep = cv2.erode(bm.astype(np.uint8), np.ones((STRIP_ERODE, STRIP_ERODE), np.uint8)).astype(bool)
    return np.where(keep, img, 0).astype(np.uint8)

def transplant(rec_img, rec_hidden, rec_bm, don_img, don_mask, don_bm, rng):
    """Paste the DONOR's tumour into the RECIPIENT slice whose own tumour was removed (inpainted).
    The tumour keeps its own relative position in the brain (location is a legitimate cue), but everything
    around it - skull, slice level, plane - now comes from a patient of ANOTHER class.
    Returns (image, mask) or None if the tumour does not fit."""
    ys, xs = np.where(don_mask); bys, bxs = np.where(don_bm); rys, rxs = np.where(rec_bm)
    if len(ys) == 0 or len(rys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = 4; y0, x0 = max(y0 - pad, 0), max(x0 - pad, 0); y1, x1 = min(y1 + pad, IMG_SIZE), min(x1 + pad, IMG_SIZE)
    patch = don_img[y0:y1, x0:x1].astype(np.float32)
    alpha = cv2.GaussianBlur(cv2.dilate(don_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32),
                             (7, 7), 2)[y0:y1, x0:x1]
    scale = float(np.clip((np.ptp(rys) + 1) / (np.ptp(bys) + 1), 0.7, 1.4))
    h, w = max(int(round((y1 - y0) * scale)), 3), max(int(round((x1 - x0) * scale)), 3)
    patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    med_d = np.median(don_img[don_bm & ~don_mask]) + 1e-3; med_r = np.median(rec_img[rec_bm]) + 1e-3
    patch = patch * float(np.clip(med_r / med_d, 0.6, 1.6))              # brightness matched to the recipient
    ry = (ys.mean() - bys.min()) / max(np.ptp(bys), 1); rx = (xs.mean() - bxs.min()) / max(np.ptp(bxs), 1)
    if rng.random() < 0.5:
        rx = 1 - rx                                                         # either hemisphere
    cy = rys.min() + ry * np.ptp(rys) + rng.normal(0, 3); cx = rxs.min() + rx * np.ptp(rxs) + rng.normal(0, 3)
    ty, tx = int(round(cy - h / 2)), int(round(cx - w / 2))
    if ty < 0 or tx < 0 or ty + h > IMG_SIZE or tx + w > IMG_SIZE:
        return None
    if rec_bm[ty:ty + h, tx:tx + w][alpha > 0.5].mean() < 0.9:              # tumour must land inside the head
        return None
    out = rec_hidden.astype(np.float32).copy()
    region = out[ty:ty + h, tx:tx + w]
    out[ty:ty + h, tx:tx + w] = alpha * patch + (1 - alpha) * region
    mk = np.zeros((IMG_SIZE, IMG_SIZE), bool); mk[ty:ty + h, tx:tx + w] = alpha > 0.5
    return np.clip(out, 0, 255).astype(np.uint8), mk

class TransplantSet(GuidedSet):
    """SOLUTION 2. With probability TRANSPLANT_PROB the image becomes: recipient slice (its own tumour removed
    by inpainting) + a tumour from a training patient of ANOTHER class; the label is the donor's class.
    The rest of the slice therefore carries no information about the label. NOTE: inpainting is used here only to
    remove the recipient's tumour; the reliance TEST uses 'swap' and 'tissue from another patient' instead, so the
    model cannot pass the test by recognising the training operation (the CTR-Net mistake of Run 3)."""
    def __init__(self, items):
        super().__init__(items)
        self.ok = [i for i, (p, _) in enumerate(self.items) if p in HIDE]
        self.by_cls = {k: [i for i in self.ok if self.items[i][1] == k] for k in range(N_CLASSES)}
        self.bm = {self.items[i][0]: brain_bm(load_u8(self.items[i][0]), self.items[i][0]) for i in self.ok}
        self.n_ok = self.n_try = 0
    def __getitem__(self, i):
        p, y = self.items[i]
        rng = np.random.default_rng()
        if p in HIDE and rng.random() < TRANSPLANT_PROB:
            others = [k for k in range(N_CLASSES) if k != y and self.by_cls[k]]
            k = int(rng.choice(others)); j = int(rng.choice(self.by_cls[k])); q = self.items[j][0]
            res = transplant(load_u8(p), HIDE[p]["inpaint"], self.bm[p], load_u8(q), MASKS[q], self.bm[q], rng)
            if res is not None:
                img, mk = res
                z = np.zeros_like(img)
                st = np.stack([img, img, img, mk.astype(np.uint8) * 255, z])
                return torch.from_numpy(st), torch.tensor(0.0), k
        return super().__getitem__(i)

def to_batch(arrs, mean, std, device=DEVICE):
    x = torch.from_numpy(np.stack(arrs)).to(device).float().div_(255).unsqueeze(1).repeat(1, 3, 1, 1)
    m = torch.tensor([float(v) for v in mean], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    s = torch.tensor([float(v) for v in std], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (x - m) / s

def class_weights(items):
    c = np.bincount([y for _, y in items], minlength=N_CLASSES).astype(float)
    w = 1.0 / np.maximum(c, 1); w = w / w.sum() * N_CLASSES
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)

# ─── Fast metrics (used everywhere) ───
def hard(P):
    """Predicted class per row; -1 = ABSTAINED (an ROI model that found no tumour outputs exactly equal
    probabilities - this must not be counted as 'glioma', which np.argmax would do)."""
    P = np.asarray(P); pred = P.argmax(1)
    pred[(P.max(1) - P.min(1)) < 1e-6] = -1
    return pred

def macro_f1(y, p, n=N_CLASSES, labels=None):
    """Macro-F1; abstained predictions (-1) count as a miss for their true class (no false positive elsewhere)."""
    labels = range(n) if labels is None else labels
    y, p = np.asarray(y), np.asarray(p); ok = p >= 0
    cm = np.bincount(y[ok] * n + p[ok], minlength=n * n).reshape(n, n)
    f = []
    for k in labels:
        tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k, :].sum() - tp + int(((y == k) & ~ok).sum())
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))

def macro_auc(y, probs):
    y = np.asarray(y); present = np.unique(y)
    if len(present) == 2:                     # e.g. deduplicated BRISC has no glioma: AUC of the 2 present classes
        a, b = present
        try:
            return float(roc_auc_score(y == b, probs[:, b] / (probs[:, a] + probs[:, b] + 1e-12)))
        except ValueError:
            return float("nan")
    try:
        return float(roc_auc_score(np.asarray(y), probs, multi_class="ovr", labels=list(range(N_CLASSES))))
    except ValueError:
        return float("nan")

def per_class_recall(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {IDX2CLASS[k]: (float((p[y == k] == k).mean()) if (y == k).any() else float("nan")) for k in range(N_CLASSES)}

def softmax_np(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)

# %% [markdown]
# ### 2.5 Run 5 inputs at a glance (check that the new inputs look sensible before trusting any result)

# %%
# ─── 2.5 What the new solutions actually see ───
if MASKS:
    rng_v = np.random.default_rng(7)
    tr_items = FOLDS[1]["train"]
    by_c = {k: [p for p, y in tr_items if y == k and p in HIDE] for k in range(N_CLASSES)}
    # (a) tumour transplant examples
    rows_ = []
    for rec_c, don_c in [(2, 0), (0, 1), (1, 2), (2, 1)]:
        for _ in range(50):
            p, q = rng_v.choice(by_c[rec_c]), rng_v.choice(by_c[don_c])
            res = transplant(load_u8(p), HIDE[p]["inpaint"], brain_mask(load_u8(p)), load_u8(q), MASKS[q], brain_mask(load_u8(q)), rng_v)
            if res is not None:
                rows_.append((p, q, rec_c, don_c, res)); break
    if rows_:
        fig, ax = plt.subplots(len(rows_), 3, figsize=(7.2, 2.45 * len(rows_)), squeeze=False)
        for r, (p, q, rc, dc, (img, mk)) in enumerate(rows_):
            ax[r, 0].imshow(load_u8(p), cmap="gray"); ax[r, 0].contour(MASKS[p], levels=[0.5], colors="lime", linewidths=0.7)
            ax[r, 1].imshow(load_u8(q), cmap="gray"); ax[r, 1].contour(MASKS[q], levels=[0.5], colors="orange", linewidths=0.7)
            ax[r, 2].imshow(img, cmap="gray"); ax[r, 2].contour(mk, levels=[0.5], colors="orange", linewidths=0.7)
            ax[r, 0].set_title(f"recipient: {CLASSES[rc]} patient", fontsize=7)
            ax[r, 1].set_title(f"donor tumour: {CLASSES[dc]}", fontsize=7)
            ax[r, 2].set_title(f"training image -> label '{CLASSES[dc]}'", fontsize=7, color=CLASS_COLORS[CLASSES[dc]])
            for a in ax[r]: a.axis("off")
        fig.suptitle("Solution 2 - tumour transplant: the surroundings come from a patient of ANOTHER class,\n"
                     "so only the tumour itself tells the label (green = own tumour, removed; orange = donor tumour)", fontsize=8)
        fig.tight_layout(); fig.savefig(FIG_DIR / "fig_transplant_examples.png", bbox_inches="tight"); plt.show()
    # (b) skull stripping (control experiment)
    ex = [rng_v.choice(by_c[k]) for k in range(N_CLASSES)] + ([BRATS_ITEMS[0][0]] if BRATS_ITEMS else [])
    fig, ax = plt.subplots(2, len(ex), figsize=(2.1 * len(ex), 4.4), squeeze=False)
    for c, p in enumerate(ex):
        a = load_u8(p); is_b = c == len(ex) - 1 and BRATS_ITEMS
        ax[0, c].imshow(a, cmap="gray"); ax[1, c].imshow(a if is_b else skull_strip(a), cmap="gray")
        ax[0, c].set_title("BraTS (already stripped)" if is_b else f"Figshare {CLASSES[c]}", fontsize=7)
        for a_ in ax[:, c]: a_.axis("off")
    ax[1, 0].text(-12, IMG_SIZE / 2, "what CNX-Strip sees", rotation=90, va="center", ha="right", fontsize=7)
    fig.suptitle("Control - skull stripping: does ROI win on BraTS only because BraTS has no skull?", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_strip_examples.png", bbox_inches="tight"); plt.show()
    # (c) ROI-Loc input: crop + 6 location numbers
    names6 = ["vertical pos.", "dist. midline", "dist. centre", "rel. area", "rel. width", "rel. height"]
    fig, ax = plt.subplots(2, 3, figsize=(7.5, 5.2))
    for c in range(N_CLASSES):
        p = rng_v.choice(by_c[c]); a = load_u8(p); bm = brain_mask(a); b = box_from_mask(MASKS[p]); lf = loc_feats(a, MASKS[p], bm)
        ax[0, c].imshow(a, cmap="gray"); ax[0, c].contour(bm, levels=[0.5], colors="cyan", linewidths=0.6)
        ax[0, c].contour(MASKS[p], levels=[0.5], colors="lime", linewidths=0.8)
        ax[0, c].set_title(f"{CLASSES[c]}: brain outline (cyan) + tumour", fontsize=7)
        ax[1, c].imshow(crop_roi(a, b, "box"), cmap="gray")
        ax[1, c].set_title("crop + " + ", ".join(f"{v:+.2f}" for v in lf[:3]) + "\n" + ", ".join(f"{v:.2f}" for v in lf[3:]), fontsize=6)
        for a_ in ax[:, c]: a_.axis("off")
    fig.suptitle("Solution 1 - ROI-Loc sees ONLY the tumour crop + 6 location numbers\n(" + ", ".join(names6) + ")", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_roiloc_inputs.png", bbox_inches="tight"); plt.show()
