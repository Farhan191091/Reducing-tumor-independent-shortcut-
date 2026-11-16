# %% [markdown]
# ## 9. Explainability against expert tumour masks (LayerCAM at 14x14 for every model, with chance baselines)
# Raw maps only (no brain masking, no smoothing) for all metrics; maps are averaged over the 5 fold models.
# Sets: deduplicated BRISC tumour images (100 per class) and BraTS slices (150 per subset), all with expert masks.
# Baselines: a random map hits the mask with probability = mask area fraction; "centre" = mask contains the image centre.

# %%
# ─── 9.1 LayerCAM engine ───
class GradCAMpp:   # name kept for compatibility; computes LayerCAM
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        def fwd(_, __, out):
            self.acts = out
            out.register_hook(lambda g: setattr(self, "grads", g))
        self.handle = layer.register_forward_hook(fwd)

    def __call__(self, x, target):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[0, target].backward()
        A, G = self.acts.detach().float(), self.grads.detach().float()
        cam = F.relu((F.relu(G) * A).sum(1, keepdim=True))       # LayerCAM (Jiang et al. 2021): pixel-wise weights
        cam = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), torch.softmax(logits.detach(), 1)[0].cpu().numpy()

    def close(self):
        self.handle.remove()

def ensemble_cams(name, items, target_of, seed=None):
    """Mean LayerCAM over the 5 fold models of one seed. target_of(i) -> class index."""
    seed = READY[name][0] if seed is None else seed
    cams = np.zeros((len(items), IMG_SIZE, IMG_SIZE), np.float32)
    for f in range(1, N_FOLDS + 1):
        m = load_checkpoint(name, seed, f); mean, std = norm_for(name, f)
        engine = GradCAMpp(m, cam_target_layer(m, name))
        for i, (p, _) in enumerate(tqdm(items, desc=f"  LayerCAM {name} fold {f}", leave=False)):
            cams[i] += engine(to_batch([load_u8(p)], mean, std), target_of(i))[0] / N_FOLDS
        engine.close(); del m; gc.collect()
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return cams


# %%
# ─── 9.2 Quantitative attribution metrics (cached per model -> later runs reuse them) ───
xai_items = []          # (path, label, mask_source, dataset)
for p_, y_ in TEST_ITEMS:                                           # internal test: expert masks from the .mat files
    if p_ in MASKS:
        xai_items.append((p_, y_, "figshare", "Internal test"))
rng = random.Random(SPLIT_SEED)
for k in range(N_CLASSES):
    pool = [(p, y) for p, y in BRISC_CLEAN if y == k and Path(p).stem in BRISC_MASKS]
    for p, y in rng.sample(pool, min(100, len(pool))):
        xai_items.append((p, y, BRISC_MASKS[Path(p).stem], "BRISC (dedup)"))
if len(BRATS_META):
    for d in ["gli", "men"]:
        sub = BRATS_META[BRATS_META.dataset == d]
        sub = sub.sample(min(150, len(sub)), random_state=SPLIT_SEED)
        for r in sub.itertuples():
            xai_items.append((r.path, r.y, str(BRATS_DIR / d / "masks_tc" / Path(r.file).name), f"BraTS-{d.upper()}"))
print(f"XAI images: {dict(Counter(d for *_, d in xai_items))}")

def xai_mask(i):
    p, _, src, _ = xai_items[i]
    return MASKS[p] if src == "figshare" else load_mask(src)

def up224(c):
    return np.asarray(Image.fromarray(np.asarray(c, np.float32)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))

XAI_KEY = "layercam14_" + hashlib.md5("|".join(p for p, *_ in xai_items).encode()).hexdigest()[:10]
xai_rows, ALL_CAMS = [], {}                  # ALL_CAMS[name] = (n, 56, 56) float16 heat maps (shown upsampled)
if RUN_XAI and RUN_EVALUATION and XAI_MODELS and xai_items:
    masks = [xai_mask(i) for i in range(len(xai_items))]
    pos = {}
    for s_name in ("test", "brisc", "brats"):
        if s_name in EVAL_SETS:
            pos.update({p: (s_name, i) for i, p in enumerate(paths_of(s_name))})
    banner("9.2 XAI vs expert tumour masks", f"{len(XAI_MODELS)} models x {len(xai_items)} images x 5 folds "
           "(~3-8 min per model; cached models are instant)")
    for name in XAI_MODELS:
        cache = OUT_DIR / "xai" / f"{name}_{XAI_KEY}.npz"; rows_csv = OUT_DIR / "xai" / f"{name}_{XAI_KEY}_rows.csv"
        if cache.exists() and rows_csv.exists():
            ALL_CAMS[name] = np.load(cache)["cams"]; xai_rows += pd.read_csv(rows_csv).to_dict("records")
            print(f"{name:16s} XAI loaded from cache"); continue
        t0 = time.time()
        cams = ensemble_cams(name, [(p, y) for p, y, _, _ in xai_items], lambda i: xai_items[i][1])
        P = {s: ensemble_probs(name, s) for s in {"test", "brisc", "brats"} & set(EVAL_SETS)}
        rows_m = []
        for i, ((p, y, _, d), cam, mk) in enumerate(zip(xai_items, cams, masks)):
            if mk.sum() == 0:
                continue
            s, j = pos[p]
            yx = np.unravel_index(cam.argmax(), cam.shape)
            thr = cam >= 0.3
            rows_m.append({"Model": name, "item": i, "Dataset": d, "Class": IDX2CLASS[y],
                           "correct": bool(P[s][j].argmax() == y), "pred": int(P[s][j].argmax()), "conf": float(P[s][j].max()),
                           "pointing": float(mk[yx]), "energy_in_mask": float(cam[mk].sum() / (cam.sum() + 1e-8)),
                           "iou@0.3": float((thr & mk).sum() / ((thr | mk).sum() + 1e-8)),
                           "chance": float(mk.mean()), "centre": float(mk[IMG_SIZE // 2, IMG_SIZE // 2])})
        small = F.avg_pool2d(torch.from_numpy(cams)[:, None], 4)[:, 0].numpy().astype(np.float16)
        np.savez_compressed(cache, cams=small); pd.DataFrame(rows_m).to_csv(rows_csv, index=False)
        ALL_CAMS[name] = small; xai_rows += rows_m
        print(f"{name:16s} XAI done ({(time.time() - t0) / 60:.1f} min)")
    df_xai_img = pd.DataFrame(xai_rows)
    df_xai_img.to_csv(OUT_DIR / "tables" / "xai_per_image.csv", index=False)
    agg = (df_xai_img.groupby(["Dataset", "Model"])
           [["pointing", "energy_in_mask", "iou@0.3", "chance", "centre"]].mean().round(3))
    agg["n"] = df_xai_img.groupby(["Dataset", "Model"]).size()
    agg_c = (df_xai_img[df_xai_img.correct].groupby(["Dataset", "Model"])
             [["pointing", "energy_in_mask"]].mean().round(3).add_suffix(" (correct only)"))
    df_xai = agg.join(agg_c)
    print(df_xai.to_string())
    df_xai.to_csv(OUT_DIR / "tables" / "table_xai.csv")
    (df_xai_img.groupby(["Dataset", "Class", "Model"])[["pointing", "energy_in_mask", "iou@0.3", "chance"]]
     .mean().round(3).to_csv(OUT_DIR / "tables" / "table_xai_by_class.csv"))
    print("pointing = heat-map peak inside the expert tumour outline; energy_in_mask = share of heat inside it; "
          "chance = what a random map would get (tumour area).")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for a, metric in zip(ax, ["pointing", "energy_in_mask"]):
        piv = df_xai_img.groupby(["Dataset", "Model"])[metric].mean().unstack()[XAI_MODELS]
        piv.plot.bar(ax=a, rot=0, width=0.8)
        ch = df_xai_img.groupby("Dataset")["chance"].mean()
        for i, dname in enumerate(piv.index):
            a.hlines(ch[dname], i - 0.4, i + 0.4, colors="k", linestyles="--", lw=1)
        a.set(title=f"{metric} (dashed = random map)", ylim=(0, 1)); a.legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_xai_metrics.png"); plt.show()

# %%
# ─── 9.3 Main explainability figure: where does each model look? ───
def xai_compare_panel(picks, fname):
    cols = [("MRI + expert outline", None, None)]
    if SEG_MAPS.get("Internal test") and SEG_MODEL:
        cols.append((f"{PROPOSED}: region used" if is_roi(PROPOSED) else f"{SEG_MODEL}: predicted tumour", SEG_MODEL,
                     "roi" if is_roi(PROPOSED) else "seg"))
    for n in ["CAM-Guide", "CNX-Transplant", "CNX-Plain", "CTR-Net", "MTK"]:
        if n in ALL_CAMS: cols.append((f"{n}: heat map", n, "cam"))
    seg_lookup = {p: mp for its, maps in SEG_MAPS.values() for (p, _), mp in zip(its, maps)}
    info = {n: df_xai_img[df_xai_img.Model == n].set_index("item") for n in ALL_CAMS}
    test_index = {q: i for i, q in enumerate(paths_of("test"))}
    brats_index = {q: i for i, q in enumerate(paths_of("brats"))} if "brats" in EVAL_SETS else {}
    BRATS_P = ensemble_probs(PROPOSED, "brats") if "brats" in EVAL_SETS and PROPOSED in READY else None
    fig, ax = plt.subplots(len(picks), len(cols), figsize=(2.25 * len(cols), 2.45 * len(picks)), squeeze=False)
    for r, i in enumerate(picks):
        p, y, _, d = xai_items[i]; img = load_u8(p); mk = xai_mask(i)
        for c, (title, n, kind) in enumerate(cols):
            a = ax[r, c]; a.imshow(img, cmap="gray"); a.axis("off")
            if kind is None:
                a.contour(mk, levels=[0.5], colors="lime", linewidths=1.0)
                a.set_title(f"True: {IDX2CLASS[y]}", fontsize=8, color=CLASS_COLORS[IDX2CLASS[y]], fontweight="bold")
            elif kind == "cam":
                cam = up224(ALL_CAMS[n][i].astype(np.float32)); cam = cam / max(cam.max(), 1e-8); row = info[n].loc[i]
                a.imshow(cam, cmap=BLOOD, vmin=0, vmax=1)
                a.contour(mk, levels=[0.5], colors="lime", linewidths=0.8)
                ok = bool(row["correct"])
                a.set_title(f"{'✓' if ok else '✗'} {IDX2CLASS[int(row['pred'])][:5]} ({row['conf']:.2f}) | "
                            f"in tumour {100 * row['energy_in_mask']:.0f}%", fontsize=6.5, color="green" if ok else "red")
            else:
                sm = seg_lookup.get(p)
                if sm is not None:
                    a.imshow(np.ma.masked_less(sm, 0.5), cmap="autumn", alpha=0.55, vmin=0, vmax=1)
                    a.contour(mk, levels=[0.5], colors="lime", linewidths=0.8)
                    txt = f"Dice {dice(sm >= 0.5, mk):.2f}"
                    if kind == "roi":
                        b = box_from_mask(sm >= 0.5)
                        if b is not None:
                            x, y_, w, h, _ = b; s = max(max(w, h) * (1 + 2 * ROI_MARGIN), 32)
                            a.add_patch(plt.Rectangle((x + w / 2 - s / 2, y_ + h / 2 - s / 2), s, s, fill=False, ec="cyan", lw=1.2))
                        ti = test_index.get(p); bi = brats_index.get(p)
                        pr = (TEST_PROBS[PROPOSED][ti] if ti is not None and PROPOSED in TEST_PROBS else
                              BRATS_P[bi] if bi is not None and BRATS_P is not None else None)
                        if pr is not None:
                            hp = int(hard(pr[None])[0]); ok = hp == y
                            txt = f"{'✓' if ok else '✗'} {IDX2CLASS[hp][:5]} ({pr.max():.2f}) | " + txt
                    a.set_title(txt, fontsize=6.5)
            if r == 0:
                a.set_title(f"{title}\n" + a.get_title(), fontsize=a.title.get_fontsize(), color=a.title.get_color(),
                            fontweight="bold" if kind is None else "normal")
    fig.suptitle("Green = expert tumour outline. Cyan box = the only region the ROI model sees. Heat maps: LayerCAM, "
                 "14x14, mean of 5 fold models (red = important; same 0-1 scale for every model); "
                 "'in tumour' = share of heat inside the outline. Sharper maps: section 9.7.", fontsize=8, y=1.01)
    fig.tight_layout(); fig.savefig(FIG_DIR / fname, bbox_inches="tight", dpi=300); plt.show()

XAI_REF = next((n for n in ["CAM-Guide", "CNX-Plain", "CTR-Net"] if n in ALL_CAMS), None)
if RUN_XAI and ALL_CAMS and XAI_REF:
    R_ = df_xai_img[(df_xai_img.Model == XAI_REF) & (df_xai_img.Dataset == "Internal test")]
    rng_x = np.random.default_rng(2026); picks = []
    for c in CLASSES:                                   # one random correct example per class (fixed seed)
        cand = R_[(R_.Class == c) & R_.correct].item.values
        if len(cand): picks.append(int(rng_x.choice(cand)))
    wrong = R_[~R_.correct].item.values                 # + up to 2 random errors (honest figure)
    picks += [int(i) for i in rng_x.choice(wrong, min(2, len(wrong)), replace=False)] if len(wrong) else []
    xai_compare_panel(picks, "fig_xai_main_panel.png")
    ext = [int(i) for d in ["BraTS-GLI", "BraTS-MEN", "BRISC (dedup)"]
           for i in rng_x.choice([j for j, it in enumerate(xai_items) if it[3] == d] or [-1], 1) if i >= 0]
    if ext:
        xai_compare_panel(ext, "fig_xai_external_panel.png")

# %%
# ─── 9.4 Heat-map comparison across models, with the expert tumour outline (green) ───
if RUN_XAI and ALL_CAMS:
    rng = np.random.default_rng(7); pick = []
    for d, k in [("Internal test", 2), ("BRISC (dedup)", 1), ("BraTS-GLI", 2), ("BraTS-MEN", 2)]:
        idx = [i for i, it in enumerate(xai_items) if it[3] == d]
        pick += [int(i) for i in rng.choice(idx, min(k, len(idx)), replace=False)] if idx else []
    names = [n for n in XAI_MODELS if n in ALL_CAMS]
    fig, ax = plt.subplots(len(pick), 1 + len(names), figsize=(1.9 * (1 + len(names)), 1.9 * len(pick)), squeeze=False)
    for r, i in enumerate(pick):
        p, y, _, d = xai_items[i]
        img, mk = load_u8(p), xai_mask(i)
        ax[r, 0].imshow(img, cmap="gray"); ax[r, 0].contour(mk, levels=[0.5], colors="lime", linewidths=0.8)
        ax[r, 0].set_title(f"{d}\n{IDX2CLASS[y]}", fontsize=7)
        for c, n in enumerate(names, 1):
            cam = up224(ALL_CAMS[n][i].astype(np.float32))
            ax[r, c].imshow(img, cmap="gray")
            ax[r, c].imshow(cam, cmap=BLOOD, vmin=0, vmax=1)
            ax[r, c].contour(mk, levels=[0.5], colors="lime", linewidths=0.6)
            yx = np.unravel_index(cam.argmax(), cam.shape)
            ax[r, c].plot(yx[1], yx[0], "w+", ms=7, mew=1.5)
            if r == 0:
                ax[r, c].set_title(n, fontsize=8)
        for a in ax[r]: a.axis("off")
    fig.suptitle("LayerCAM (mean of 5 fold models). Green = expert tumour outline, + = map peak", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_xai_heatmaps_models.png"); plt.show()

# %%
# ─── 9.5 Counterfactual figure: same slice, tumour visible vs tumour hidden ───
if RUN_EVALUATION and REL and PROPOSED in REL and "CNX-Plain" in REL:
    hid = REL["CNX-Plain"]["hidden (swap)"]; ok_plain = hid.argmax(1) == REL_Y
    rng_c = np.random.default_rng(11); rows_ = []
    for k in range(N_CLASSES):                          # slices where the plain model is right WITHOUT the tumour
        cand = np.where(ok_plain & (REL_Y == k) & (REL_DF.role.values == "test"))[0]
        cand = cand if len(cand) else np.where(ok_plain & (REL_Y == k))[0]
        if len(cand): rows_.append(int(rng_c.choice(cand)))
    models_ = list(dict.fromkeys(m for m in ["CNX-Plain", "CNX-Transplant", "CTR-Net", PROPOSED] if m in REL))
    fig, ax = plt.subplots(len(rows_), 2 + len(models_), figsize=(2.3 * (2 + len(models_)), 2.3 * len(rows_)), squeeze=False)
    for r, i in enumerate(rows_):
        p = REL_DF.filepath.values[i]
        ax[r, 0].imshow(load_u8(p), cmap="gray"); ax[r, 0].contour(MASKS[p], levels=[0.5], colors="lime", linewidths=0.8)
        ax[r, 1].imshow(HIDE[p]["swap"], cmap="gray")
        ax[r, 0].set_title(f"Original ({CLASSES[REL_Y[i]]})", fontsize=7); ax[r, 1].set_title("Tumour hidden", fontsize=7)
        for a in ax[r, :2]: a.axis("off")
        for c, m in enumerate(models_, 2):
            po, ph = REL[m]["original"][i], REL[m]["hidden (swap)"][i]; xs = np.arange(N_CLASSES)
            ax[r, c].bar(xs - 0.2, po, 0.4, color="#30638e", label="tumour visible")
            ax[r, c].bar(xs + 0.2, ph, 0.4, color="#d1495b", label="tumour hidden")
            ax[r, c].axhline(1 / 3, color="k", ls=":", lw=0.8)
            ax[r, c].set(ylim=(0, 1), xticks=xs, title=m)
            ax[r, c].set_xticklabels([k[:4] for k in CLASSES], fontsize=7)
            if r == 0 and c == 2: ax[r, c].legend(frameon=False, fontsize=6)
    fig.suptitle("Predicted probabilities with the tumour visible vs hidden. A model that looks at the tumour should "
                 "become unsure (dotted line = 1/3) when it is hidden.", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_counterfactual_examples.png", bbox_inches="tight"); plt.show()

# %%
# ─── 9.6 Heat-map sanity check (Adebayo et al. 2018): do the maps depend on what the model learned? ───
# The trained model's layers are replaced by random weights step by step (head first, then deeper stages).
# If a heat map stays the SAME after randomisation, it shows image edges, not the model's reasoning (fails).
# Measure: rank correlation between the trained model's map and the randomised model's map (low = passes).
import copy
from scipy.stats import spearmanr as _spr
SANITY_MODELS = [n for n in ["CAM-Guide", "CNX-Plain", "CTR-Net", "MTL-Seg"] if n in READY]
if RUN_XAI and SANITY_MODELS:
    banner("9.6 Heat-map sanity check (cascading randomisation)", f"{len(SANITY_MODELS)} models x 60 test slices (~2 min)")
    rng_s = np.random.default_rng(5)
    sitems = [TEST_ITEMS[i] for i in rng_s.choice(len(TEST_ITEMS), min(60, len(TEST_ITEMS)), replace=False)]
    steps = [("head", ["head", "head_norm"]), ("+ stage 4", ["backbone.stages_3"]),
             ("+ stage 3", ["backbone.stages_2"]), ("+ stage 2", ["backbone.stages_1"])]
    def maps_of(model, name, targets, mean, std):
        eng = GradCAMpp(model, cam_target_layer(model, name)); out = []
        for (p, _), t in zip(sitems, targets):
            out.append(F.avg_pool2d(torch.from_numpy(eng(to_batch([load_u8(p)], mean, std), t)[0])[None, None], 4)[0, 0].numpy())
        eng.close(); return np.stack(out)
    san_rows, examples = [], {}
    for n in SANITY_MODELS:
        m0 = copy.deepcopy(load_checkpoint(n, READY[n][0], 1)); mean, std = norm_for(n, 1)
        targets = predict_logits(m0, [load_u8(p) for p, _ in sitems], mean, std).argmax(1)
        base = maps_of(m0, n, targets, mean, std); examples[n] = [base[0]]
        for label, mods in steps:
            for mod_name in mods:
                for sub in m0.get_submodule(mod_name).modules():
                    if hasattr(sub, "reset_parameters"):
                        sub.reset_parameters()
            cur = maps_of(m0, n, targets, mean, std); examples[n].append(cur[0])
            rho = np.nanmean([_spr(a.ravel(), b.ravel())[0] for a, b in zip(base, cur)])
            san_rows.append({"Model": n, "Randomised up to": label, "Similarity to trained map (rank corr.)": round(float(rho), 3)})
        del m0; free_mem()
    df_san = pd.DataFrame(san_rows).pivot(index="Model", columns="Randomised up to", values="Similarity to trained map (rank corr.)")
    df_san = df_san[[s for s, _ in steps]]
    print("Rank correlation between the trained model's heat map and the randomised model's map.")
    print("PASS = drops clearly (e.g. below ~0.3) as the model is randomised; FAIL = stays high (map = edge detector).")
    print(df_san.to_string()); df_san.to_csv(OUT_DIR / "tables" / "table_xai_sanity.csv")
    fig, ax = plt.subplots(len(SANITY_MODELS), 1 + len(steps), figsize=(1.9 * (1 + len(steps)), 2 * len(SANITY_MODELS)), squeeze=False)
    img0 = load_u8(sitems[0][0])
    for r, n in enumerate(SANITY_MODELS):
        for c, (mp, lab) in enumerate(zip(examples[n], ["trained"] + [s for s, _ in steps])):
            ax[r, c].imshow(img0, cmap="gray"); ax[r, c].imshow(up224(mp / (mp.max() + 1e-8)), cmap=BLOOD, vmin=0, vmax=1)
            ax[r, c].axis("off"); ax[r, c].set_title(lab, fontsize=7)
        ax[r, 0].text(-10, 112, n, rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle("Sanity check: heat maps should change when the model's weights are randomised", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_xai_sanity.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 9.7 High-resolution explanations for EVERY model, including the ROI models (Run 6)
# * **Sharper maps:** LayerCAM on two feature levels (28x28 and 14x14) fused, instead of 14x14 only.
# * **ROI models:** the map is computed on the crop the model actually sees and pasted back onto the full slice
#   (so it can be compared with the expert outline like every other model).
# * **Fair baseline for ROI:** a crop is mostly tumour, so a random map already scores well *inside the crop*.
#   We therefore also report **chance inside the crop** and the ratio "heat in tumour / chance".
# * One colour scale for every model: **blood red = important** (0-1, same scale everywhere).

# %%
# ─── 9.7a High-resolution heat maps (cached per model) ───
def _bb(m):
    return m.cnn.backbone if isinstance(m, RoiLocNet) else m.backbone

class HiResCAM:
    """Multi-level LayerCAM (Jiang et al. 2021) for ConvNeXt-based models: maps at 28x28 and 14x14, each normalised,
    averaged. Returns (map in input space 224x224, class probabilities)."""
    def __init__(self, model):
        self.model, self.feats = model, None
        self.h = _bb(model).register_forward_hook(lambda mod, inp, out: setattr(self, "feats", out))
    def __call__(self, x, target, loc=None):
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = self.model(x.float(), loc=loc) if loc is not None else self.model(x.float())
            f1, f2 = self.feats[0], self.feats[1]
            g1, g2 = torch.autograd.grad(out[0, int(target)], [f1, f2])
        maps = []
        for A, G in [(f1, g1), (f2, g2)]:
            c = F.relu((F.relu(G) * A).sum(1, keepdim=True)).detach().float()
            c = F.interpolate(c, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
            maps.append((c - c.min()) / (c.max() - c.min() + 1e-8))
        cam = (maps[0] + maps[1]) / 2; cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), torch.softmax(out.detach().float(), 1)[0].cpu().numpy()
    def close(self):
        self.h.remove()

def roi_window(b, margin):
    """Exactly the square that crop_roi cuts out: (x0, y0, x1, y1)."""
    x, y, w, h, _ = b; s = max(max(w, h) * (1 + 2 * margin), 32); cx, cy = x + w / 2, y + h / 2
    return int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)

def paste_back(cam_crop, win):
    x0, y0, x1, y1 = win; big = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    cs = np.asarray(Image.fromarray(cam_crop.astype(np.float32)).resize((x1 - x0, y1 - y0), Image.BILINEAR))
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, IMG_SIZE), min(y1, IMG_SIZE)
    if xb > xa and yb > ya:
        big[ya:yb, xa:xb] = cs[ya - y0:yb - y0, xa - x0:xb - x0]
    return big

def model_inputs(name, f, arrs):
    """What the model really sees: full slices, or (crop, location numbers, window) for ROI models."""
    if not is_roi(name):
        return [(a, None, None) for a in arrs]
    spec = MODEL_SPECS[name]; mg = spec.get("margin", ROI_MARGIN); out = []
    for a, mp in zip(arrs, roi_seg_maps(arrs, f)):
        b = box_from_mask(mp >= 0.5)
        if b is None:
            out.append((None, None, None)); continue
        lc = torch.from_numpy(loc_feats(a, b[4])[None]).to(DEVICE) if spec.get("loc") else None
        out.append((crop_roi(a, b, spec["crop"], margin=mg), lc, roi_window(b, mg)))
    return out

def hires_cams(name, arrs, targets, seed=None):
    """Mean (over the 5 fold models) high-resolution map in FULL-SLICE coordinates + fraction of folds that found a tumour."""
    seed = READY[name][0] if seed is None else seed
    cams = np.zeros((len(arrs), IMG_SIZE, IMG_SIZE), np.float32); found = np.zeros(len(arrs)); wins = [[] for _ in arrs]
    for f in range(1, N_FOLDS + 1):
        m = load_checkpoint(name, seed, f); mean, std = norm_for(name, f); eng = HiResCAM(m)
        for i, (inp, lc, win) in enumerate(model_inputs(name, f, arrs)):
            if inp is None:
                continue
            cam, _ = eng(to_batch([inp], mean, std), targets[i], lc)
            cams[i] += (paste_back(cam, win) if win is not None else cam) / N_FOLDS; found[i] += 1 / N_FOLDS
            if win is not None: wins[i].append(win)
        eng.close(); free_mem()
    return cams, found, wins

HR_MODELS = [n for n in ["ROI-Loc", "ROI-m06-Transplant", "ROI-Crop-m06", "CNX-Transplant", "CAM-Guide", "CNX-Plain",
                         "CTR-Net", "MTL-Seg"] if n in READY]
HR_CAMS, hr_rows, HR_WINS = {}, [], {}
if RUN_XAI and RUN_EVALUATION and HR_MODELS and xai_items:
    banner("9.7a High-resolution heat maps", f"{len(HR_MODELS)} models x {len(xai_items)} images x 5 folds (~3-6 min per model, cached)")
    masks = [xai_mask(i) for i in range(len(xai_items))]
    arrs_x = [load_u8(p) for p, *_ in xai_items]; tg = [y for _, y, _, _ in xai_items]
    pos = {}
    for s_name in ("test", "brisc", "brats"):
        if s_name in EVAL_SETS:
            pos.update({p: (s_name, i) for i, p in enumerate(paths_of(s_name))})
    HR_KEY = "hires_" + XAI_KEY
    for name in HR_MODELS:
        cache = OUT_DIR / "xai" / f"{name}_{HR_KEY}.npz"; rows_csv = OUT_DIR / "xai" / f"{name}_{HR_KEY}_rows.csv"
        if cache.exists() and rows_csv.exists():
            HR_CAMS[name] = np.load(cache)["cams"].astype(np.float32); hr_rows += pd.read_csv(rows_csv).to_dict("records")
            print(f"{name:20s} loaded from cache"); continue
        t0 = time.time(); _CUR_SET["name"] = None
        cams, found, wins = hires_cams(name, arrs_x, tg)
        P = {s: ensemble_probs(name, s) for s in {"test", "brisc", "brats"} & set(EVAL_SETS)}
        rows_m = []
        for i, ((p, y, _, d), cam, mk) in enumerate(zip(xai_items, cams, masks)):
            if mk.sum() == 0:
                continue
            s, j = pos[p]; hp = int(hard(P[s][j][None])[0])
            if wins[i]:
                win_mask = np.zeros_like(mk)
                for (x0, y0, x1, y1) in wins[i]:
                    win_mask[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
                ch_crop = float(mk[win_mask].mean()) if win_mask.any() else float(mk.mean())
            else:
                ch_crop = float(mk.mean())
            has = cam.max() > 0
            yx = np.unravel_index(cam.argmax(), cam.shape); thr = cam >= 0.3 * cam.max() if has else np.zeros_like(mk)
            rows_m.append({"Model": name, "item": i, "Dataset": d, "Class": IDX2CLASS[y], "correct": hp == y, "pred": hp,
                           "conf": float(P[s][j].max()), "tumour found (share of folds)": float(found[i]),
                           "pointing": float(mk[yx]) if has else 0.0,
                           "energy_in_mask": float(cam[mk].sum() / (cam.sum() + 1e-8)),
                           "iou@0.3": float((thr & mk).sum() / ((thr | mk).sum() + 1e-8)),
                           "chance (full slice)": float(mk.mean()), "chance (inside what the model sees)": ch_crop})
        small = F.avg_pool2d(torch.from_numpy(cams)[:, None], 2)[:, 0].numpy().astype(np.float16)   # 112x112
        np.savez_compressed(cache, cams=small); pd.DataFrame(rows_m).to_csv(rows_csv, index=False)
        HR_CAMS[name] = small.astype(np.float32); hr_rows += rows_m
        print(f"{name:20s} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # The 5 fold maps were each scaled 0-1 and then averaged; folds disagree a little, so the averaged peak is lower than 1
    # and the maps looked faint. Re-scale every averaged map to 0-1 (standard practice; all metrics above are scale-free).
    for name in HR_CAMS:
        mx = HR_CAMS[name].reshape(len(HR_CAMS[name]), -1).max(1)[:, None, None]
        HR_CAMS[name] = np.where(mx > 0, HR_CAMS[name] / np.maximum(mx, 1e-8), 0).astype(np.float32)
    df_hr = pd.DataFrame(hr_rows)
    df_hr.to_csv(OUT_DIR / "tables" / "xai_hires_per_image.csv", index=False)
    cols_ = ["pointing", "energy_in_mask", "iou@0.3", "chance (full slice)", "chance (inside what the model sees)",
             "tumour found (share of folds)"]
    agg_hr = df_hr.groupby(["Dataset", "Model"])[cols_].mean()
    agg_hr["heat in tumour / chance"] = agg_hr["energy_in_mask"] / agg_hr["chance (inside what the model sees)"]
    agg_hr["n"] = df_hr.groupby(["Dataset", "Model"]).size()
    agg_hr = agg_hr.round(3)
    print("HIGH-RESOLUTION explanations vs expert tumour outline (mean over images; 5-fold average maps):")
    print(agg_hr.to_string()); agg_hr.to_csv(OUT_DIR / "tables" / "table_xai_hires.csv")
    print("\nHow to read: energy_in_mask = share of the red heat inside the expert outline. For ROI models compare it with")
    print("'chance (inside what the model sees)', because their input is already a crop around the tumour.")
    print("'heat in tumour / chance' > 1 means the model concentrates on the tumour more than a random map would.")

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for a, metric, ttl in [(ax[0], "energy_in_mask", "Share of heat inside the tumour"),
                           (ax[1], "heat in tumour / chance", "Heat in tumour ÷ random-map baseline (fair for ROI)")]:
        piv = agg_hr[metric].unstack()[HR_MODELS]
        piv.plot.bar(ax=a, rot=0, width=0.85, color=[FAMILY_COLOR[family_of(n)] if not n.startswith("ROI-") or n == PROPOSED
                                                      else {"ROI-m06-Transplant": "#264653", "ROI-Crop-m06": "#f4a261"}.get(n, "#f08a8f")
                                                      for n in HR_MODELS])
        if metric == "heat in tumour / chance":
            a.axhline(1, color="k", ls="--", lw=1); a.set_yscale("log")
        a.set(title=ttl, xlabel=""); a.legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle("High-resolution heat maps: how much of the 'red' falls on the tumour?", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_xai_hires_metrics.png", bbox_inches="tight"); plt.show()
else:
    df_hr = pd.DataFrame()

# %%
# ─── 9.7b Main explanation figure (journal style, blood-red scale, large panels) ───
# DISPLAY ONLY: each map is smoothed (Gaussian, sigma = 6 px) and re-scaled to 0-1 so that the important region shows as
# one coherent red area; weak heat (< 35% of the peak) is not drawn. All NUMBERS (tables, "in tumour %") are computed on the
# raw, unsmoothed maps in 9.7a.
DISP_SIGMA, DISP_FLOOR = 6, 0.35

def disp_map(n, i):
    cam = np.asarray(Image.fromarray(HR_CAMS[n][i].astype(np.float32)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    cam = cv2.GaussianBlur(cam, (0, 0), DISP_SIGMA)
    cam = cam / max(cam.max(), 1e-8)
    return np.where(cam >= DISP_FLOOR, cam, 0.0)

def _zoom_box(mk, extra=2.4):
    ys, xs = np.where(mk); cy, cx = ys.mean(), xs.mean()
    s = int(min(max(max(np.ptp(ys), np.ptp(xs), 20) * extra, 64), IMG_SIZE))
    return int(np.clip(cx - s / 2, 0, IMG_SIZE - s)), int(np.clip(cy - s / 2, 0, IMG_SIZE - s)), s

def hires_panel(picks, fname, title, models):
    cols = [("MRI + expert outline", None, "img")]
    for n, kind in models:
        if n in HR_CAMS:
            cols.append((n + (" (zoom)" if kind == "zoom" else ""), n, kind))
    info = {n: df_hr[df_hr.Model == n].set_index("item") for n in HR_CAMS}
    S = 3.0
    fig, ax = plt.subplots(len(picks), len(cols), figsize=(S * len(cols) + 0.9, S * len(picks) + 0.6), squeeze=False)
    for r, i in enumerate(picks):
        p, y, _, d = xai_items[i]; img = load_u8(p); mk = xai_mask(i)
        for c, (ttl, n, kind) in enumerate(cols):
            a = ax[r, c]; a.axis("off")
            if kind == "img":
                a.imshow(img, cmap="gray"); a.contour(mk, levels=[0.5], colors="lime", linewidths=1.4)
                a.set_title(("MRI + expert outline\n" if r == 0 else "") + f"{d} | true: {IDX2CLASS[y]}", fontsize=9,
                            color=CLASS_COLORS[IDX2CLASS[y]], fontweight="bold")
                continue
            row = info[n].loc[i]; hp = int(row["pred"]); ok = hp == y; found = float(row["tumour found (share of folds)"])
            head = (ttl + "\n") if r == 0 else ""
            if is_roi(n) and found == 0:
                a.imshow(img, cmap="gray"); a.contour(mk, levels=[0.5], colors="lime", linewidths=1.0)
                a.text(IMG_SIZE / 2, IMG_SIZE * 0.85, "no tumour found -> unsure", color="yellow", ha="center",
                       fontsize=9, fontweight="bold")
                a.set_title(head + "abstained (referred)", fontsize=9, color="#b8860b"); continue
            cam = disp_map(n, i)
            if kind == "zoom":
                x0, y0, s = _zoom_box(mk); img_, cam_, mk_ = img[y0:y0 + s, x0:x0 + s], cam[y0:y0 + s, x0:x0 + s], mk[y0:y0 + s, x0:x0 + s]
            else:
                img_, cam_, mk_ = img, cam, mk
            a.imshow(img_, cmap="gray"); a.imshow(cam_, cmap=BLOOD, vmin=0, vmax=1, interpolation="bilinear")
            a.contour(mk_, levels=[0.5], colors="lime", linewidths=1.4 if kind == "zoom" else 1.0)
            lab = IDX2CLASS[hp][:5] if hp >= 0 else "unsure"
            extra = f" | {round(found * 5)}/5 folds" if is_roi(n) and found < 1 else ""
            a.set_title(head + f"{'✓' if ok else '✗'} {lab} {row['conf']:.2f} | {100 * row['energy_in_mask']:.0f}% in tumour{extra}",
                        fontsize=8.5, color="green" if ok else "red")
    sm = plt.cm.ScalarMappable(cmap=BLOOD_SOLID, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax.ravel().tolist(), fraction=0.015, pad=0.01)
    cb.set_label("importance (0-1, same scale for every model)", fontsize=9)
    fig.suptitle(title, fontsize=11, y=1.0)
    fig.savefig(FIG_DIR / fname, bbox_inches="tight", dpi=300); plt.show()

MAIN_XAI = [("ROI-m06-Transplant", "zoom"), ("ROI-m06-Transplant", "cam"), ("CNX-Transplant", "cam"),
            ("CAM-Guide", "cam"), ("CNX-Plain", "cam")]
SECOND_XAI = [("ROI-Loc", "zoom"), ("ROI-Loc", "cam"), ("CTR-Net", "cam"), ("MTL-Seg", "cam")]
if RUN_XAI and len(df_hr):
    ref = df_hr[(df_hr.Model == HR_MODELS[0]) & (df_hr.Dataset == "Internal test")]
    rng_h = np.random.default_rng(2026); picks = []
    for c in CLASSES:                                         # one random correct example per class (fixed seed)
        cand = ref[(ref.Class == c) & ref.correct].item.values
        if len(cand): picks.append(int(rng_h.choice(cand)))
    wrong = ref[~ref.correct].item.values                     # + one random error (honest figure)
    if len(wrong): picks.append(int(rng_h.choice(wrong)))
    cap = ("Green = expert tumour outline, red = what the model used (LayerCAM, 28+14 px levels, mean of 5 fold models; "
           f"smoothed sigma={DISP_SIGMA}px, weak heat <{int(100 * DISP_FLOOR)}% of peak hidden; % = share of RAW heat inside the tumour)")
    hires_panel(picks, "fig_xai_hires_main.png", "Our solutions (left) vs. standard models (right) - internal test\n" + cap, MAIN_XAI)
    hires_panel(picks, "fig_xai_hires_other.png", "Other models on the same slices\n" + cap, SECOND_XAI)
    ext = []
    for d in ["BraTS-GLI", "BraTS-MEN", "BRISC (dedup)"]:
        cand = df_hr[(df_hr.Model == HR_MODELS[0]) & (df_hr.Dataset == d)].item.values
        if len(cand): ext.append(int(rng_h.choice(cand)))
    if ext:
        hires_panel(ext, "fig_xai_hires_external.png", "New hospitals (BraTS) and BRISC - no fine-tuning\n" + cap, MAIN_XAI)

# %%
# ─── 9.7c 3D view of the heat maps: is there ONE sharp peak on the tumour, or many peaks elsewhere? ───
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
if RUN_XAI and len(df_hr):
    m3 = [n for n in ["ROI-m06-Transplant", "CNX-Transplant", "CNX-Plain"] if n in HR_CAMS]
    cases = picks[:3]
    fig = plt.figure(figsize=(5.6 * len(m3), 5.0 * len(cases)))
    g = np.arange(0, IMG_SIZE, 3); X, Y = np.meshgrid(g, g)
    for r, i in enumerate(cases):
        img = load_u8(xai_items[i][0]); mk = xai_mask(i)
        for c, n in enumerate(m3):
            a = fig.add_subplot(len(cases), len(m3), r * len(m3) + c + 1, projection="3d")
            cam = np.asarray(Image.fromarray(HR_CAMS[n][i].astype(np.float32)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
            cam = cv2.GaussianBlur(cam, (0, 0), DISP_SIGMA); cam = cam / max(cam.max(), 1e-8)
            a.plot_surface(X, Y, np.zeros_like(X, float), facecolors=plt.cm.gray(img[::3, ::3] / 255.0), rstride=1, cstride=1,
                           shade=False, linewidth=0, antialiased=False)
            Z = cam[::3, ::3]
            a.plot_surface(X, Y, np.where(Z >= 0.05, Z, np.nan), cmap=BLOOD_SOLID, vmin=0, vmax=1, rstride=1, cstride=1,
                           linewidth=0, antialiased=True, alpha=0.95)
            ys, xs = np.where(cv2.morphologyEx(mk.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)
            a.plot(xs[::2], ys[::2], np.zeros(len(xs[::2])) + 0.01, ".", color="lime", ms=2)
            raw = HR_CAMS[n][i].astype(np.float32); mk_s = np.asarray(Image.fromarray(mk.astype(np.uint8)).resize(raw.shape[::-1])) > 0
            e = float(raw[mk_s].sum() / (raw.sum() + 1e-8))
            a.set(zlim=(0, 1.05), xticks=[], yticks=[], zticks=[0, 0.5, 1]); a.invert_yaxis(); a.view_init(elev=42, azim=-60)
            a.set_title(f"{n}\n{IDX2CLASS[xai_items[i][1]]} | {100 * e:.0f}% of heat in tumour", fontsize=11)
    fig.suptitle("3D heat maps: height + red = importance, floor = MRI slice, green = expert tumour outline on the floor.\n"
                 "A good explanation is ONE peak standing on the green outline; a shortcut model shows peaks elsewhere.",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_xai_3d.png", bbox_inches="tight", dpi=220); plt.show()

# %%
# ─── 9.7d Faithfulness (deletion test): do the RED pixels really drive the decision? ───
# Remove the most important pixels step by step (replace by the typical intensity of HEALTHY brain tissue of the same
# slice - a blurred copy would still look like a tumour to an ROI model) and watch the confidence in the true
# class fall. Compare with removing the SAME number of RANDOM pixels. Faster fall than random = faithful explanation.
DEL_STEPS = [0, 1, 2, 5, 10, 20, 35, 50]                     # % of the pixels the model looks at
DEL_MODELS = [n for n in HR_MODELS if n in ["ROI-Loc", "ROI-m06-Transplant", "CNX-Transplant", "CAM-Guide", "CNX-Plain", "CTR-Net"]]
if RUN_XAI and DEL_MODELS:
    banner("9.7d Deletion test", f"{len(DEL_MODELS)} models x 120 internal-test slices x 5 folds (~2 min per model, cached)")
    rng_d = np.random.default_rng(9)
    _pool = {c: [j for j, it in enumerate(xai_items) if it[3] == "Internal test" and it[1] == c] for c in range(N_CLASSES)}
    di = [i for c in range(N_CLASSES) for i in rng_d.choice(_pool[c], min(40, len(_pool[c])), replace=False)]
    del_rows, CURVES = [], {}
    for n in DEL_MODELS:
        cache = OUT_DIR / "xai" / f"{n}_deletion2_{XAI_KEY}.npz"
        if cache.exists():
            z = np.load(cache); Ch, Cr = z["heat"], z["rand"]
        else:
            Ch = np.full((N_FOLDS, len(di), len(DEL_STEPS)), np.nan); Cr = Ch.copy()
            arrs_d = [load_u8(xai_items[i][0]) for i in di]
            for f in range(1, N_FOLDS + 1):
                m = load_checkpoint(n, READY[n][0], f); mean, std = norm_for(n, f); eng = HiResCAM(m)
                for k, (inp, lc, win) in enumerate(model_inputs(n, f, arrs_d)):
                    if inp is None:
                        continue
                    t = xai_items[di[k]][1]; cam, _ = eng(to_batch([inp], mean, std), t, lc)
                    region = np.ones_like(inp, bool) if is_roi(n) else brain_mask(inp)
                    full = arrs_d[k]; bm_f = brain_mask(full); mk_f = xai_mask(di[k])
                    healthy = np.median(full[bm_f & ~mk_f]) if (bm_f & ~mk_f).any() else np.median(full)
                    blur = np.full_like(inp, int(healthy))        # removed pixels = typical HEALTHY brain intensity
                    order = np.argsort(-np.where(region, cam, -1).ravel()); nreg = int(region.sum())
                    ridx = rng_d.permutation(np.flatnonzero(region.ravel()))
                    batch = []
                    for q in DEL_STEPS:
                        kq = int(nreg * q / 100)
                        for idx in (order[:kq], ridx[:kq]):
                            v = inp.copy().ravel(); v[idx] = blur.ravel()[idx]; batch.append(v.reshape(inp.shape))
                    xb = to_batch(batch, mean, std)
                    with torch.no_grad():
                        lo = m(xb, loc=lc.expand(len(batch), -1)) if lc is not None else m(xb)
                    pr = torch.softmax(lo.float(), 1)[:, t].cpu().numpy()
                    Ch[f - 1, k] = pr[0::2] / max(pr[0], 1e-6); Cr[f - 1, k] = pr[1::2] / max(pr[1], 1e-6)
                eng.close(); free_mem()
            np.savez_compressed(cache, heat=Ch, rand=Cr)
        ch, cr = np.nanmean(Ch, 0), np.nanmean(Cr, 0)                         # (images, steps)
        okr = np.isfinite(ch).all(1) & np.isfinite(cr).all(1)
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        auc_h = _trapz(ch[okr], DEL_STEPS, axis=1) / DEL_STEPS[-1]; auc_r = _trapz(cr[okr], DEL_STEPS, axis=1) / DEL_STEPS[-1]
        gap = auc_r - auc_h; boot = [gap[rng_d.integers(0, len(gap), len(gap))].mean() for _ in range(2000)]
        CURVES[n] = (np.nanmean(ch[okr], 0), np.nanmean(cr[okr], 0))
        del_rows.append({"Model": n, "images": int(okr.sum()), "deletion AUC - red pixels": round(float(auc_h.mean()), 3),
                         "deletion AUC - random pixels": round(float(auc_r.mean()), 3),
                         "faithfulness gap (random - red)": round(float(gap.mean()), 3),
                         "95% CI": f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]",
                         "p (Wilcoxon)": float(sps.wilcoxon(auc_r, auc_h).pvalue) if len(gap) > 5 else np.nan})
    df_del = pd.DataFrame(del_rows).sort_values("faithfulness gap (random - red)", ascending=False)
    print("Deletion test (confidence in the true class, relative to the untouched image; lower AUC = falls faster).")
    print("Faithfulness gap > 0 with a CI above 0 = the red pixels matter more than random pixels.")
    print(df_del.to_string(index=False)); df_del.to_csv(OUT_DIR / "tables" / "table_xai_faithfulness.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for n, (h_, r_) in CURVES.items():
        col = {"ROI-m06-Transplant": "#264653"}.get(n, FAMILY_COLOR[family_of(n)])
        ax[0].plot(DEL_STEPS, h_, "-o", ms=3, color=col, label=f"{n} (red pixels)")
        ax[0].plot(DEL_STEPS, r_, "--", color=col, alpha=0.6)
    ax[0].set(xlabel="Pixels removed (% of what the model sees)", ylabel="Confidence (relative to original)",
              title="Deletion curves (solid = red pixels first, dashed = random)", ylim=(0, 1.15))
    ax[0].legend(frameon=False, fontsize=6)
    d_ = df_del.set_index("Model")
    lo_ = [float(s.strip("[]").split(",")[0]) for s in d_["95% CI"]]; hi_ = [float(s.strip("[]").split(",")[1]) for s in d_["95% CI"]]
    g_ = d_["faithfulness gap (random - red)"].values
    ax[1].bar(range(len(d_)), g_, color=[{"ROI-m06-Transplant": "#264653"}.get(n, FAMILY_COLOR[family_of(n)]) for n in d_.index],
              yerr=[g_ - np.array(lo_), np.array(hi_) - g_], capsize=3)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xticks=range(len(d_)), ylabel="faithfulness gap", title="How much more the red pixels matter than random ones")
    ax[1].set_xticklabels(d_.index, rotation=25, ha="right", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_xai_deletion.png", bbox_inches="tight"); plt.show()
else:
    df_del = pd.DataFrame()

# %%
# ─── 9.7e Sanity check of the high-resolution maps, incl. the ROI model (cascading randomisation) ───
SAN2 = [n for n in [next((x for x in [PROPOSED, "ROI-m06-Transplant"] if x in HR_CAMS), None), "CNX-Transplant"] if n in READY]
if RUN_XAI and SAN2:
    rng_s2 = np.random.default_rng(6)
    _pool2 = [j for j, it in enumerate(xai_items) if it[3] == "Internal test"]
    sel = [i for i in rng_s2.choice(_pool2, min(40, len(_pool2)), replace=False)]
    arrs_s = [load_u8(xai_items[i][0]) for i in sel]; rows_s = []
    for n in SAN2:
        m0 = copy.deepcopy(load_checkpoint(n, READY[n][0], 1)); mean, std = norm_for(n, 1)
        ins = model_inputs(n, 1, arrs_s); keep = [k for k, t in enumerate(ins) if t[0] is not None]
        pre = "cnn." if isinstance(m0, RoiLocNet) else ""
        steps2 = [("head", [pre + "head", pre + "head_norm"] + (["head", "loc"] if pre else [])),
                  ("+ stage 4", [pre + "backbone.stages_3"]), ("+ stage 3", [pre + "backbone.stages_2"]),
                  ("+ stage 2", [pre + "backbone.stages_1"])]
        def maps2():
            eng = HiResCAM(m0)
            out = [eng(to_batch([ins[k][0]], mean, std), xai_items[sel[k]][1], ins[k][1])[0] for k in keep]
            eng.close(); return out
        base = maps2()
        for lab, mods in steps2:
            for mn in mods:
                for sub in m0.get_submodule(mn).modules():
                    if hasattr(sub, "reset_parameters"): sub.reset_parameters()
            cur = maps2()
            rows_s.append({"Model": n, "Randomised up to": lab,
                           "rank corr.": round(float(np.nanmean([_spr(a[::4, ::4].ravel(), b[::4, ::4].ravel())[0] for a, b in zip(base, cur)])), 3)})
        del m0; free_mem()
    df_san2 = pd.DataFrame(rows_s).pivot(index="Model", columns="Randomised up to", values="rank corr.")[[s for s, _ in steps2]]
    print("Sanity check of the HIGH-RESOLUTION maps (rank correlation trained vs randomised; low = PASS):")
    print(df_san2.to_string()); df_san2.to_csv(OUT_DIR / "tables" / "table_xai_sanity_hires.csv")

# %% [markdown]
# ## 10. Shortcut case study with the ORIGINAL 4-class models (no training)
# Question: did the 4-class model detect "no tumour" from the brain, or from the image source (IXI vs Figshare)?
# (a) Can non-anatomical features alone separate no-tumour from tumour images?
# (b) What do the old models predict when the brain is erased (only background left), or when only the brain is kept?

# %%
# ─── 10.1 Old 4-class model definition (exact copy of the submitted v2, needed to load its checkpoints) ───
class BSplineKAN(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, degree=3):
        super().__init__()
        self.grid_size = grid_size
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.1)
        self.register_buffer("grid", torch.linspace(-1, 1, grid_size + degree + 1))
        self.act = nn.SiLU()
    def forward(self, x):
        centers = torch.linspace(-1, 1, self.grid_size, device=x.device)
        basis = torch.exp(-((x.unsqueeze(-1) - centers) ** 2) / 0.5)
        return F.linear(self.act(x), self.base_weight) + torch.einsum("big,oig->bo", basis, self.spline_weight)

class GIK(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.kan = BSplineKAN(i, o); self.norm = nn.LayerNorm(o)
    def forward(self, x):
        return self.norm(self.kan(x))

class TriConvBlock(ConvBlock):
    pass

class SpatialMambaLite(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm, self.proj_in = nn.LayerNorm(dim), nn.Linear(dim, dim)
        self.dwconv = nn.Conv1d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.gate, self.proj_out, self.act = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.GELU()
    def forward(self, x):
        x0 = x; x = self.norm(x); z = torch.sigmoid(self.gate(x))
        x = self.proj_out(self.act(self.dwconv(self.proj_in(x).transpose(1, 2)).transpose(1, 2)))
        return x0 + z * x

class TriKANHead(nn.Module):
    def __init__(self, i, h1=128, h2=64):
        super().__init__(); self.gik1, self.gik2 = GIK(i, h1), GIK(h1, h2)
    def forward(self, x):
        return self.gik2(self.gik1(x))

class FeatureFusionGate(nn.Module):
    def __init__(self, c, m, k, d=128):
        super().__init__()
        self.cnn_proj, self.mamba_proj, self.kan_proj = nn.Linear(c, d), nn.Linear(m, d), nn.Linear(k, d)
        self.gate = nn.Sequential(nn.Linear(d * 3, d), nn.GELU(), nn.Linear(d, 3), nn.Softmax(dim=1))
        self.out_norm, self.out_drop = nn.LayerNorm(d), nn.Dropout(0.25)
    def forward(self, c, m, k):
        c, m, k = self.cnn_proj(c), self.mamba_proj(m), self.kan_proj(k)
        w = self.gate(torch.cat([c, m, k], 1))
        return self.out_drop(self.out_norm(w[:, :1] * c + w[:, 1:2] * m + w[:, 2:] * k)), w

class MedTriMambaKAN_v2_4class(nn.Module):
    def __init__(self, num_classes=4, base_dim=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, base_dim, 3, 1, 1, bias=False), nn.BatchNorm2d(base_dim), nn.GELU())
        self.stage1, self.stage2 = TriConvBlock(base_dim, 64, 2), TriConvBlock(64, 128, 2)
        self.stage3, self.stage4 = TriConvBlock(128, 192, 2), TriConvBlock(192, 256, 2)
        self.feature_map_layer = self.stage4.block[0]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cnn_proj = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.20))
        self.mamba_reduce, self.mamba_bn = nn.Conv2d(256, 128, 1, bias=False), nn.BatchNorm2d(128)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
        self.mamba_block1, self.mamba_block2 = SpatialMambaLite(128), SpatialMambaLite(128)
        self.mamba_norm = nn.LayerNorm(128)
        self.kan_pool, self.kan_head = nn.AdaptiveAvgPool2d(1), TriKANHead(128, 128, 64)
        self.fusion = FeatureFusionGate(128, 128, 64, 128)
        self.skip_alpha = nn.Parameter(torch.tensor(0.1))
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.30), nn.Linear(64, num_classes))
    def forward(self, x):
        x2 = self.stage2(self.stage1(self.stem(x))); x4 = self.stage4(self.stage3(x2))
        c = self.cnn_proj(self.global_pool(x4).flatten(1))
        m = self.mamba_bn(self.mamba_reduce(x4)).flatten(2).transpose(1, 2)
        m = torch.cat([self.cls_token.expand(m.size(0), -1, -1), m], 1)
        m = self.mamba_norm(self.mamba_block2(self.mamba_block1(m)))[:, 0]
        k = self.kan_head(self.kan_pool(x2).flatten(1))
        fused, _ = self.fusion(c, m, k)
        return self.classifier(fused + self.skip_alpha * c)

# %%
# ─── 10.2 (a) Non-anatomical feature probe: tumour vs no-tumour, patient-grouped CV ───
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

def brain_mask_old(u8):
    _, th = cv2.threshold(cv2.GaussianBlur(u8, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return th > 0
    m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return cv2.dilate(m, np.ones((11, 11), np.uint8)) > 0

if RUN_SHORTCUT:
    banner("10.2 Shortcut probe", "~2-5 min")
    feats_meta, feats_bg = [], []
    for p in tqdm(df_all.filepath, desc="  Image features (no anatomy)"):
        w, h, _, ext, nbytes = native_props(p)
        u8 = load_u8(p); bm = brain_mask_old(u8); bg = u8[~bm] if (~bm).sum() > 50 else np.zeros(1, np.uint8)
        feats_meta.append([w, h, ext in (".jpg", ".jpeg"), nbytes / 1024])
        feats_bg.append([bg.mean(), bg.std(), bm.mean(), *np.histogram(bg, 16, (0, 255), density=True)[0]])
    y_nt = (df_all.label4 == "notumor").astype(int).values
    probe_rows = []
    for fname, X in [("File metadata (size, format, bytes)", np.array(feats_meta, float)),
                     ("Background intensities only", np.array(feats_bg, float)),
                     ("Both", np.hstack([feats_meta, feats_bg]).astype(float))]:
        prob = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), X, y_nt,
                                 groups=df_all.pid.values, cv=GroupKFold(5), method="predict_proba")[:, 1]
        probe_rows.append({"Features (no anatomy)": fname, "AUC tumour vs no-tumour": round(roc_auc_score(y_nt, prob), 4)})
    df_probe = pd.DataFrame(probe_rows)
    print("Can 'no tumour' be recognised WITHOUT looking at the brain? (AUC 0.5 = no, 1.0 = perfectly)")
    print(df_probe.to_string(index=False))
    df_probe.to_csv(OUT_DIR / "tables" / "table_shortcut_probe.csv", index=False)

# %%
# ─── 10.3 (b) Old 4-class models on the original 713-image test set: original / brain erased / brain only ───
def find_old_ckpt(k):
    p = OLD_ARTIFACT_DIR / "models" / f"MedTriMambaKAN_v2_fold{k}_best.pt"
    if p.exists():
        return p
    hits = list(KAGGLE_IN.rglob(p.name)) if KAGGLE_IN.exists() else []
    return hits[0] if hits else None

def old_fold_stats():
    js = OLD_ARTIFACT_DIR / "outputs" / "fold_stats.json"
    if js.exists():
        return {int(k): v for k, v in json.load(open(js)).items()}
    out = {}                                   # exact replication of the original Cell 10
    for k in range(1, N_FOLDS + 1):
        random.seed(SPLIT_SEED)
        tr = df_all[(df_all.role == "dev") & (df_all.val_fold != k)]
        s = s2 = n = 0.0
        for cls in CLASSES_4:
            files = list(tr[tr.label4 == cls].filepath)
            for f in random.sample(files, min(200, len(files))):
                a = np.asarray(Image.open(f).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), np.float64) / 255
                s += a.sum(); s2 += (a ** 2).sum(); n += a.size
        mu = s / n; out[k] = {"mean": [mu] * 3, "std": [math.sqrt(s2 / n - mu ** 2)] * 3}
    return out

shortcut_rows = []
if RUN_SHORTCUT:
    old_ckpts = {k: find_old_ckpt(k) for k in range(1, N_FOLDS + 1)}
    old_ckpts = {k: v for k, v in old_ckpts.items() if v is not None}
    print("Old 4-class checkpoints found:", sorted(old_ckpts))
    if old_ckpts:
        ostats = old_fold_stats()
        te4 = df_all[df_all.role == "test"].reset_index(drop=True)
        y4 = te4.label4_idx.values
        conds = {"original": [], "brain erased": [], "brain only": []}
        for p in tqdm(te4.filepath, desc="  Building erased images"):
            u8 = load_u8(p); bm = brain_mask_old(u8)
            conds["original"].append(u8)
            conds["brain erased"].append(np.where(bm, 0, u8).astype(np.uint8))
            conds["brain only"].append(np.where(bm, u8, 0).astype(np.uint8))
        probs4 = {c: [] for c in conds}
        for k, ck in old_ckpts.items():
            m = MedTriMambaKAN_v2_4class().to(DEVICE)
            m.load_state_dict(torch.load(ck, map_location=DEVICE), strict=True); m.eval()
            for c, arrs in conds.items():
                probs4[c].append(softmax_np(predict_logits(m, arrs, ostats[k]["mean"], ostats[k]["std"])))
            del m
        for c in conds:
            pr = np.mean(probs4[c], 0).argmax(1)
            for k4, cname in enumerate(CLASSES_4):
                mk = y4 == k4
                shortcut_rows.append({"Input": c, "True class": cname, "n": int(mk.sum()),
                                      "Predicted no-tumour (%)": round(100 * (pr[mk] == 3).mean(), 1),
                                      "Correct (%)": round(100 * (pr[mk] == k4).mean(), 1)})
        df_short = pd.DataFrame(shortcut_rows)
        print("Share of test images that the OLD 4-class model calls 'NO TUMOUR' (%):")
        print(df_short.pivot(index="True class", columns="Input", values="Predicted no-tumour (%)")
              [["original", "brain erased", "brain only"]])
        print("\nShare of test images classified CORRECTLY (%):")
        print(df_short.pivot(index="True class", columns="Input", values="Correct (%)")[["original", "brain erased", "brain only"]])
        df_short.to_csv(OUT_DIR / "tables" / "table_shortcut_erasure.csv", index=False)

        fig, ax = plt.subplots(2, 3, figsize=(6, 4.2))
        for r, cls in enumerate(["glioma", "notumor"]):
            i = int(np.where(te4.label4 == cls)[0][0])
            for c, cond in enumerate(conds):
                ax[r, c].imshow(conds[cond][i], cmap="gray"); ax[r, c].axis("off")
                ax[r, c].set_title(f"{cls} | {cond}", fontsize=7)
        fig.tight_layout(); fig.savefig(FIG_DIR / "fig_shortcut_inputs.png"); plt.show()

# %% [markdown]
# ## 10b. Do large foundation models also classify without the tumour?
# Frozen image features of large pretrained models + a simple linear classifier (same folds, same tests).
# No fine-tuning, so this measures what the foundation features themselves contain.

# %%
# ─── 10b.1 Foundation models: tumour-hiding test, BraTS, BDNeuro, tumour-free slices ───
FM_SPECS = [("DINOv2-S", "timm", "vit_small_patch14_dinov2.lvd142m"),
            ("DINOv2-B", "timm", "vit_base_patch14_dinov2.lvd142m"),
            ("BiomedCLIP", "openclip", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")]

def fm_encoder(kind, ref):
    if kind == "timm":
        m = timm.create_model(ref, pretrained=True, num_classes=0, img_size=IMG_SIZE).to(DEVICE).eval()
        return (lambda x: m(x)), IMAGENET_MEAN, IMAGENET_STD
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=False)
    import open_clip
    m, _ = open_clip.create_model_from_pretrained(ref); m = m.to(DEVICE).eval()
    return (lambda x: m.encode_image(x)), [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]

def fm_embed(enc, arrs, mean, std):
    out = []
    with torch.no_grad():
        for i in range(0, len(arrs), 64):
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                out.append(enc(to_batch(arrs[i:i + 64], mean, std)).float().cpu().numpy())
    return np.concatenate(out)

fm_rows = []
if RUN_EVALUATION and len(REL_DF):
    banner("10b.1 Foundation models (frozen features + linear classifier)", "~3-6 min per model (cached)")
    conds = {"original": None, "hidden (swap)": "swap", "random spot hidden": "random", "skull / position only": "skull"}
    ext = {s: EVAL_SETS[s] for s in ["brats", "bdneuro", "bfree"] if s in EVAL_SETS}
    for fm_name, kind, ref in FM_SPECS:
        cache = OUT_DIR / "fm" / f"{fm_name}.npz"
        try:
            if cache.exists():
                E = dict(np.load(cache))
            else:
                enc, mean, std = fm_encoder(kind, ref); E = {}
                for c, key in conds.items():
                    E[c] = fm_embed(enc, [load_u8(p) if key is None else HIDE[p][key] for p in REL_DF.filepath], mean, std)
                for s, items in ext.items():
                    E[s] = fm_embed(enc, [load_u8(p) for p, _ in items], mean, std)
                cache.parent.mkdir(exist_ok=True); np.savez_compressed(cache, **E); del enc; free_mem()
        except Exception as ex:
            print(f"  {fm_name}: skipped ({type(ex).__name__}: {str(ex)[:120]})"); continue
        P = {c: np.zeros((len(REL_DF), N_CLASSES)) for c in conds}; Pe = {s: 0 for s in ext}
        test_idx = np.where(REL_DF.role == "test")[0]
        for f in range(1, N_FOLDS + 1):
            tr = np.where((REL_DF.role == "dev") & (REL_DF.val_fold != f))[0]
            va = np.where((REL_DF.role == "dev") & (REL_DF.val_fold == f))[0]
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(E["original"][tr], REL_Y[tr])
            for c in conds:
                P[c][va] = clf.predict_proba(E[c][va]); P[c][test_idx] += clf.predict_proba(E[c][test_idx]) / N_FOLDS
            for s in ext:
                Pe[s] = Pe[s] + clf.predict_proba(E[s]) / N_FOLDS
        ba = {c: bal_acc(REL_Y, P[c].argmax(1)) for c in conds}
        row = {"Model": fm_name, "original": round(100 * ba["original"], 1), "hidden (swap)": round(100 * ba["hidden (swap)"], 1),
               "skull / position only": round(100 * ba["skull / position only"], 1),
               "Tumour reliance": round(float(np.clip((ba["original"] - ba["hidden (swap)"]) / (ba["original"] - 1 / 3), 0, 1)), 2),
               "Internal test bal. acc (%)": round(100 * bal_acc(REL_Y[test_idx], P["original"][test_idx].argmax(1)), 1)}
        if "brats" in Pe:
            yb_ = labels_of("brats"); g = Pe["brats"][:, 0] / (Pe["brats"][:, 0] + Pe["brats"][:, 1] + 1e-12)
            row["BraTS AUC"] = round(roc_auc_score(yb_ == 0, g), 4)
        if "bdneuro" in Pe:
            row["BDNeuro bal. acc (%)"] = round(100 * bal_acc(labels_of("bdneuro"), Pe["bdneuro"].argmax(1)), 1)
        if "bfree" in Pe:
            row["Confident on tumour-free (%)"] = round(100 * (Pe["bfree"].max(1) >= 0.85).mean(), 1)
        fm_rows.append(row); print(f"  {fm_name:11s} done")
    if fm_rows:
        df_fm = pd.DataFrame(fm_rows)
        ref = [r for r in rel_rows if r["Model"] in ("CNX-Plain", PROPOSED)]
        print("\nFoundation models (frozen) vs trained models - balanced accuracy (%), guessing = 33.3%:")
        print(df_fm.to_string(index=False)); df_fm.to_csv(OUT_DIR / "tables" / "table_foundation_models.csv", index=False)
        print("Reference (fine-tuned):", {r["Model"]: (r["original"], r["hidden (swap)"]) for r in ref})

# %% [markdown]
# ## 11. Export for GitHub / Zenodo + numbers for the paper

# %%
# ─── 11.0 Study-design figure (Figure 1 of the paper) ───
def box(ax, x, y, w, h, text, fc):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5, wrap=True)
def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))
n_dev = int((df3.role == "dev").sum()); n_te = len(TEST_ITEMS)
fig, ax = plt.subplots(figsize=(13, 5.0)); ax.set(xlim=(0, 13), ylim=(0.6, 5.6)); ax.axis("off")
C_DATA, C_AUDIT, C_MODEL, C_EVAL = "#e8f1f8", "#fbe9e7", "#e3f4f1", "#fff4e0"
for x_, t_ in [(1.25, "1. DATA"), (4.35, "2. DIAGNOSE the problem"), (7.65, "3. SOLVE it"), (11.0, "4. VERIFY (new hospitals + safety)")]:
    ax.text(x_, 5.4, t_, ha="center", fontsize=9, fontweight="bold", color="#333333")
box(ax, 0.1, 4.0, 2.3, 1.1, f"Figshare T1c, 3 tumour types\n{len(df3)} slices / {df3.pid.nunique()} patients\n+ expert masks (.mat)", C_DATA)
box(ax, 0.1, 2.5, 2.3, 1.1, f"Patient-level split\ndev {n_dev} slices (5 folds)\ntest {n_te} slices / {df3[df3.role=='test'].pid.nunique()} patients", C_DATA)
box(ax, 0.1, 1.0, 2.3, 1.1, f"Duplicate audit (5 public sets)\nBRISC {len(BRISC_ITEMS)}->{len(BRISC_CLEAN)}, Mendeley {len(MEND_ITEMS)}->{len(MEND_CLEAN)}\nBDNeuro {len(BD_ITEMS)}->{len(BD_CLEAN)} unique", C_AUDIT)
box(ax, 2.9, 4.0, 2.9, 1.1, "Tumour-hiding test (20+ models)\nhide tumour / random spot / skull only\n-> tumour-reliance score", C_AUDIT)
box(ax, 2.9, 2.5, 2.9, 1.1, "Metric check: bigger hidden areas,\ntissue from another patient;\nplane + location baselines", C_AUDIT)
box(ax, 2.9, 1.0, 2.9, 1.1, "Existing fixes fail: multi-task seg,\nRRR, CAM guidance, counterfactual\ntraining (CTR), foundation models", C_AUDIT)
box(ax, 6.3, 4.0, 2.7, 1.1, "SOLUTION 1  ROI-Loc\ntumour crop + 6 location numbers\n(no whole slice -> no shortcut)", C_MODEL)
box(ax, 6.3, 2.5, 2.7, 1.1, "SOLUTION 2  Tumour transplant\nordinary model; surroundings from a\npatient of another class", C_MODEL)
box(ax, 6.3, 1.0, 2.7, 1.1, "Context sweep (box 0.2 / 0.6 / 1.5 / whole)\n+ context with transplant (combined)\n+ skull-strip control", C_MODEL)
box(ax, 9.5, 4.0, 3.3, 1.1, "Internal test + CV\npatient-level metrics, cluster bootstrap,\nNadeau-Bengio, Holm", C_EVAL)
box(ax, 9.5, 2.5, 3.3, 1.1, f"New hospitals, no fine-tuning\nBraTS-GLI/MEN {len(BRATS_ITEMS)} slices,\nBDNeuro {len(BD_CLEAN)}, Mendeley {len(MEND_CLEAN)} unique", C_EVAL)
box(ax, 9.5, 1.0, 3.3, 1.1, f"Safety: {len(BFREE_ITEMS)} tumour-free slices, shortcut\nalarm, conformal referral, calibration,\nLayerCAM + sanity check", C_EVAL)
for y_ in (4.55, 3.05, 1.55):
    arrow(ax, 2.4, y_, 2.9, y_); arrow(ax, 5.8, y_, 6.3, y_); arrow(ax, 9.0, y_, 9.5, y_)
ax.set_title("Study design: measure the shortcut, show why existing fixes fail, remove it, verify on new hospitals",
             fontsize=10); fig.tight_layout(); fig.savefig(FIG_DIR / "fig_study_design.png", bbox_inches="tight"); plt.show()

# %%
# ─── 11.1 Checkpoint manifest, environment, README ───
manifest = []
for name in READY:
    for s in READY[name]:
        for f in range(1, N_FOLDS + 1):
            r = load_json(run_paths(name, s, f)[1])
            manifest.append({"model": name, "seed": s, "fold": f, "checkpoint": r["ckpt"], "sha256": r["sha256"],
                             "lr": r["lr"], "best_epoch": r["best_epoch"], "val_f1": round(r["val_f1"], 4)})
pd.DataFrame(manifest).to_csv(OUT_DIR / "checkpoint_manifest.csv", index=False)
env = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
       "timm": timm.__version__, "numpy": np.__version__, "pandas": pd.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
       "settings": {k: globals()[k] for k in ["IMG_SIZE", "BATCH_SIZE", "NUM_EPOCHS", "PATIENCE", "WARMUP_EPOCHS",
                                               "WEIGHT_DECAY", "LABEL_SMOOTHING", "MIX_PROB", "N_FOLDS", "SPLIT_SEED"]},
       "learning_rates": {n: get_lr(n) for n in MODEL_ORDER}}
save_json(env, OUT_DIR / "environment.json")
(OUT_DIR / "README.md").write_text(
    "# Brain-tumour shortcut study (Run 5) - reproducibility package\n\n"
    "* `split_3class.csv` - patient-level split (test patients + validation fold of every image)\n"
    "* `overlap_audit.csv` - near-duplicate audit of BRISC / Mendeley vs Figshare\n"
    "* `checkpoint_manifest.csv` - every trained model with SHA-256\n"
    "* `tables/` - every number reported in the paper\n"
    "* `environment.json` - library versions and hyper-parameters\n\n"
    "* `figshare_masks_224.npz` - expert tumour masks matched to every Figshare PNG slice\n"
    "* `xai/` - cached LayerCAM maps (56x56) and per-image XAI scores\n\n"
    "Re-run: attach the checkpoints as a Kaggle dataset, set `RUN_TRAINING=False`, run all cells.\n")
make_artifact_zip()   # final update: same zip, now also with all tables of this run

# %%
# ─── 11.2 Key numbers (copy into the manuscript from these tables only) ───
for t in ["table_success_criteria", "table_solutions", "table_xai_hires", "table_xai_faithfulness", "table_xai_sanity_hires",
          "table_roiloc_parts", "table_skull_control", "table_shortcut_alarm", "table_metric_check",
          "table_reliance", "table_tumour_free", "table_location_plane_baselines", "table_accuracy_by_plane",
          "table_foundation_models", "table_xai_sanity", "table_reliance_vs_generalisation", "table_ablation", "table_cv", "table_cv_ttest",
          "table_test", "table_test_paired", "table_brats", "table_brats_paired", "table_bdneuro",
          "table_segmentation_head", "table_selective_prediction", "table_xai", "table_feature_source",
          "table_calibration", "table_overlap", "table_shortcut_probe", "table_complexity"]:
    p = OUT_DIR / "tables" / f"{t}.csv"
    if p.exists():
        print(f"\n===== {t} =====")
        print(pd.read_csv(p).to_string(index=False))

# %% [markdown]
# ## 12. Did the solutions work? (automatic check of the criteria fixed in cell 0.4)
# ✔ = criterion met, ✘ = not met, ? = model not trained yet (next run). Report ✘ results honestly - they are findings too.

# %%
# ─── 12.1 Success-criteria check ───
def _get(n, col):
    return SOLV.get(n, {}).get(col, np.nan) if SOLV else np.nan
def _alarm(model, set_, col):
    if not len(df_alarm): return np.nan
    d_ = df_alarm[(df_alarm.Model == model) & (df_alarm.Set == set_)]
    return d_[col].iloc[0] if len(d_) and col in d_ else np.nan
res = []
def judge(cid, ok, value):
    res.append({"ID": cid, "Result": "?" if ok is None else ("✔" if ok else "✘"), "Measured": value})
v = (_get("ROI-Loc", "CV macro-F1 (%)"),); judge("S1a", None if np.isnan(v[0]) else v[0] >= 96.0, f"CV macro-F1 {v[0]:.2f}%")
r_, sk = _get("ROI-Loc", "Reliance (swap)"), _get("ROI-Loc", "Skull/position-only BA (%)")
judge("S1b", None if np.isnan(r_) else (r_ >= 0.60 and sk <= 36), f"reliance {r_:.2f}, skull/position-only {sk:.1f}%")
a1, a0 = _get("ROI-Loc", "BraTS AUC"), _get("ROI-Crop", "BraTS AUC")
judge("S1c", None if np.isnan(a1) or np.isnan(a0) else a1 >= a0 - 0.01, f"BraTS AUC {a1:.4f} vs ROI-Crop {a0:.4f}")
r_ = _get("CNX-Transplant", "Reliance (swap)"); judge("S2a", None if np.isnan(r_) else r_ >= 0.50, f"reliance (swap) {r_:.2f}")
r_ = _get("CNX-Transplant", "Reliance (other patient)"); judge("S2b", None if np.isnan(r_) else r_ >= 0.45, f"reliance (other patient) {r_:.2f}")
f_, pit = _get("CNX-Transplant", "CV macro-F1 (%)"), _get("CNX-Transplant", "Pituitary correct when hidden (%)")
judge("S2c", None if np.isnan(f_) else (f_ >= 95.0 and pit < 80), f"CV macro-F1 {f_:.2f}%, pituitary when hidden {pit:.1f}%")
a_s = _get("CNX-Strip", "BraTS AUC")
judge("S3", None if np.isnan(a_s) or np.isnan(a0) else a_s < a0 - 0.02,
      f"CNX-Strip {a_s:.4f} vs ROI-Crop {a0:.4f} vs CNX-Plain {_get('CNX-Plain', 'BraTS AUC'):.4f}")
c_, k_ = _alarm("CNX-Plain", "BraTS", "AURC conf (%)"), _alarm("CNX-Plain", "BraTS", "AURC combo (%)")
judge("S4a", None if np.isnan(c_) else k_ < c_, f"AURC alarm {k_} vs confidence {c_} (%)")
u_ = _alarm("CNX-Plain", "BraTS tumour vs tumour-free", "AUROC combo")
judge("S4b", None if np.isnan(u_) else u_ >= 0.75, f"AUROC {u_} (confidence alone: {_alarm('CNX-Plain', 'BraTS tumour vs tumour-free', 'AUROC conf')})")
judge("S5", None if np.isnan(RHO_METRIC) else RHO_METRIC >= 0.80, f"Spearman {RHO_METRIC:.2f}")
f6, r6, b6 = _get("ROI-m06-Transplant", "CV macro-F1 (%)"), _get("ROI-m06-Transplant", "Reliance (swap)"), _get("ROI-m06-Transplant", "BraTS AUC")
judge("S6a", None if np.isnan(f6) else f6 >= 96.0, f"CV macro-F1 {f6:.2f}%")
judge("S6b", None if np.isnan(r6) else r6 >= 0.60, f"reliance {r6:.2f}")
judge("S6c", None if np.isnan(b6) or np.isnan(a0) else b6 >= a0 - 0.01, f"BraTS AUC {b6:.4f} vs ROI-Crop {a0:.4f}")
try:
    rr = agg_hr.loc[("Internal test", "ROI-Loc")]; ratio = float(rr["heat in tumour / chance"])
    extra7 = (f"; info: ROI-m06-Transplant ratio {float(agg_hr.loc[('Internal test', 'ROI-m06-Transplant')]['heat in tumour / chance']):.2f}"
              if ("Internal test", "ROI-m06-Transplant") in agg_hr.index else "")
    judge("S7", ratio >= 2.0, f"heat in tumour {100 * rr['energy_in_mask']:.1f}% vs chance in crop "
                               f"{100 * rr['chance (inside what the model sees)']:.1f}% (ratio {ratio:.2f}){extra7}")
except Exception:
    judge("S7", None, "not computed")
try:
    rd = df_del.set_index("Model").loc["ROI-Loc"]; lo7 = float(rd["95% CI"].strip("[]").split(",")[0])
    judge("S8", lo7 > 0, f"faithfulness gap {rd['faithfulness gap (random - red)']:.3f} {rd['95% CI']}")
except Exception:
    judge("S8", None, "not computed")
df_check = pd.DataFrame(CRITERIA, columns=["ID", "Experiment", "Success if ..."]).merge(pd.DataFrame(res), on="ID")
print(df_check[["ID", "Result", "Experiment", "Measured", "Success if ..."]].to_string(index=False))
df_check.to_csv(OUT_DIR / "tables" / "table_success_criteria.csv", index=False)
print(f"\n{(df_check.Result == '✔').sum()} met | {(df_check.Result == '✘').sum()} not met | {(df_check.Result == '?').sum()} not testable yet")

# %%
# ─── 12.2 Main results at a glance (one figure for the paper / presentation) ───
GL = [n for n in ["ROI-Loc", "ROI-m06-Transplant", "ROI-Crop-m06", "CNX-Transplant", "CNX-Plain", "CTR-Net", "MTL-Seg",
                  "CAM-Guide", "ConvNeXt-Tiny", "MTK"] if n in SOLV]
if GL:
    panels = [("Reliance (swap)", "Tumour reliance (higher = uses the tumour)", (0, 1)),
              ("CV macro-F1 (%)", "Cross-validation macro-F1 (%)", (85, 100)),
              ("BraTS AUC", "BraTS AUC (new hospital)", (0.8, 1.0)),
              ("BDNeuro BA (%)", "BDNeuro balanced accuracy (%)", (75, 100)),
              ("Confident on tumour-free (%)", "Confident on TUMOUR-FREE slices (%) - lower is safer", (0, 70)),
              ("Pituitary correct when hidden (%)", "Pituitary correct with tumour HIDDEN (%) - lower = no shortcut", (0, 100))]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.2))
    for a, (col, ttl, lim) in zip(ax.ravel(), panels):
        v = [SOLV[n].get(col, np.nan) for n in GL]
        cols = [{"ROI-m06-Transplant": "#264653", "ROI-Crop-m06": "#f4a261"}.get(n, FAMILY_COLOR[family_of(n)]) for n in GL]
        a.barh(range(len(GL)), v, color=cols)
        for k_, vv in enumerate(v):
            if np.isfinite(vv): a.text(vv, k_, f" {vv:.3g}", va="center", fontsize=7)
        a.set(yticks=range(len(GL)), xlim=lim, title=ttl); a.set_yticklabels(GL, fontsize=7); a.invert_yaxis()
        a.title.set_fontsize(8.5)
    fig.suptitle("Main results at a glance: the solutions (top rows) use the tumour, transfer to a new hospital and stay "
                 "unsure without a tumour; standard models do not", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_results_at_a_glance.png", bbox_inches="tight", dpi=300); plt.show()

# %%
# ─── 12.3 ⬇ Download links again (same files as in cell 3.6, now including all tables and figures) ───
make_artifact_zip()
show_downloads()
