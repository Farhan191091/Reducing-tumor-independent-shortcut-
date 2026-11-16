# %% [markdown]
# ## 3. Training (identical protocol for every model)
# AdamW, 3-epoch warm-up + cosine decay, max 80 epochs, early stopping on validation macro-F1 (patience 15),
# class-weighted cross-entropy with label smoothing 0.1, MixUp/CutMix (p = 0.5), AMP, gradient clipping.
# No SWA and no test-time augmentation anywhere. Each (model, seed, fold) is saved and skipped if it already exists.

# %%
# ─── 3.1 Training engine ───
def run_paths(name, seed, fold):
    tag = f"{name}_s{seed}_f{fold}"
    return MODEL_DIR / f"{tag}.pt", OUT_DIR / "runs" / f"{tag}.json", OUT_DIR / "oof" / f"{tag}_logits.npy"

def run_done(name, seed, fold):
    ckpt, rec, oof = run_paths(name, seed, fold)
    return ckpt.exists() and rec.exists() and oof.exists()

def get_lr(name):
    choice = load_json(OUT_DIR / "lr_choice.json", {})
    name = LR_INHERIT.get(name, name)
    return choice.get(name, DEFAULT_LR["pretrained" if is_pretrained(name) else "scratch"])

def lr_factor(epoch, total, warmup=WARMUP_EPOCHS, floor=0.01):
    if epoch < warmup:
        return (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))

def mix_batch(x, y, m=None):
    """MixUp / CutMix. If a tumour mask m is given, it is mixed exactly like the image.
    Returns x, y_a, y_b, lam, m, idx (idx = None when no mixing was applied)."""
    if np.random.rand() >= MIX_PROB:
        return x, y, y, 1.0, m, None
    idx = torch.randperm(x.size(0), device=x.device)
    if np.random.rand() < 0.5:                                    # MixUp
        lam = float(np.random.beta(0.4, 0.4))
        m = None if m is None else lam * m + (1 - lam) * m[idx]
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam, m, idx
    lam = float(np.random.beta(1.0, 1.0))                         # CutMix
    H, W = x.shape[2:]
    ch, cw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2, x1, x2 = max(cy - ch // 2, 0), min(cy + ch // 2, H), max(cx - cw // 2, 0), min(cx + cw // 2, W)
    x = x.clone(); x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    if m is not None:
        m = m.clone(); m[:, :, y1:y2, x1:x2] = m[idx, :, y1:y2, x1:x2]
    return x, y, y[idx], 1 - (y2 - y1) * (x2 - x1) / (H * W), m, idx

def seg_loss_fn(seg_logit, mask, has):
    """BCE + soft Dice between the 28x28 tumour logits and the (area-downsampled) expert mask.
    Slices without a mask (has = 0) do not contribute."""
    tgt = F.adaptive_avg_pool2d(mask, seg_logit.shape[-2:])
    z = seg_logit.float()
    bce = F.binary_cross_entropy_with_logits(z, tgt, reduction="none").mean((1, 2, 3))
    pr = torch.sigmoid(z)
    dice = 1 - (2 * (pr * tgt).sum((1, 2, 3)) + 1) / (pr.sum((1, 2, 3)) + tgt.sum((1, 2, 3)) + 1)
    return ((bce + dice) * has).sum() / has.sum().clamp(min=1)

@torch.no_grad()
def predict_logits(model, arrs, mean, std, bs=128):
    model.eval(); out = []
    for i in range(0, len(arrs), bs):
        xb = to_batch(arrs[i:i + bs], mean, std)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out)

_SEG = {"key": None, "model": None}

def roi_seg_maps(arrs, fold, seed=0):
    """Tumour-probability maps (224x224) from ROI_SEG_SOURCE's model of the SAME fold (never saw this fold's
    validation patients). Separate model cache, so it does not disturb load_checkpoint's cache."""
    ckpt = run_paths(ROI_SEG_SOURCE, seed, fold)[0]
    if _SEG["key"] != ckpt:
        if _SEG["model"] is None:
            _SEG["model"] = build_model(ROI_SEG_SOURCE, pretrained=False).to(DEVICE)
        _SEG["model"].load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=True); _SEG["model"].eval()
        _SEG["key"] = ckpt
    mean, std = norm_for(ROI_SEG_SOURCE, fold); out = []
    with torch.no_grad():
        for i in range(0, len(arrs), 64):
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                _, sl = _SEG["model"](to_batch(arrs[i:i + 64], mean, std), return_seg=True)
            out.append(torch.sigmoid(F.interpolate(sl.float(), (IMG_SIZE, IMG_SIZE), mode="bilinear",
                                                   align_corners=False))[:, 0].cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, IMG_SIZE, IMG_SIZE), np.float32)

def roi_inputs(name, fold, arrs, maps=None):
    """(crops, found, locs): crop around the predicted tumour; found = False -> nothing to classify.
    locs = the 6 location numbers of the found tumour (ROI-Loc only, otherwise zeros)."""
    spec = MODEL_SPECS[name]
    maps = roi_seg_maps(arrs, fold) if maps is None else maps
    crops, found, locs = [], [], []
    for a, mp in zip(arrs, maps):
        b = box_from_mask(mp >= 0.5)
        found.append(b is not None)
        crops.append(crop_roi(a, b, spec["crop"], margin=spec.get("margin", ROI_MARGIN)) if b is not None else a)
        locs.append(loc_feats(a, b[4]) if (b is not None and spec.get("loc")) else np.zeros(N_LOC, np.float32))
    return crops, np.array(found, bool), np.stack(locs) if locs else np.zeros((0, N_LOC), np.float32)

@torch.no_grad()
def predict_logits_loc(model, arrs, locs, mean, std, bs=128):
    model.eval(); out = []
    for i in range(0, len(arrs), bs):
        xb = to_batch(arrs[i:i + bs], mean, std); lb = torch.from_numpy(np.asarray(locs[i:i + bs], np.float32)).to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            out.append(model(xb, loc=lb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_CLASSES), np.float32)

STRIP_SKIP_SETS = {"brats", "bfree"}      # already skull-stripped -> CNX-Strip uses them as they are
_CUR_SET = {"name": None}                 # set by the evaluation loops (5.1, 8.5) to the current dataset

def predict_any(name, fold, model, arrs, mean, std, maps=None):
    """Logits for any model. ROI models: when no tumour is found the logits are 0 (= uniform = 'unsure').
    CNX-Strip: inputs are skull-stripped first (except BraTS, which already is)."""
    spec = MODEL_SPECS[name]
    if spec.get("strip"):
        if _CUR_SET["name"] not in STRIP_SKIP_SETS:
            arrs = [skull_strip(a) for a in arrs]
        return predict_logits(model, arrs, mean, std)
    if not is_roi(name):
        return predict_logits(model, arrs, mean, std)
    crops, found, locs = roi_inputs(name, fold, arrs, maps)
    z = np.zeros((len(arrs), N_CLASSES), np.float32)
    if found.any():
        sel = [c for c, f in zip(crops, found) if f]
        z[found] = (predict_logits_loc(model, sel, locs[found], mean, std) if spec.get("loc")
                    else predict_logits(model, sel, mean, std))
    return z

def guide_penalty(model, x, region, y, kind):
    """Share of the model's evidence that lies OUTSIDE the tumour region (0 = all evidence inside).
    rrr: squared input gradients of the log-probabilities (Ross et al. 2017, normalised);
    cam: class-activation map of the true class on the last 7x7 features (CAM / attention guidance)."""
    with torch.autocast(device_type=DEVICE.type, enabled=False):
        if kind == "rrr":
            x = x.float().detach().requires_grad_(True)
            lp = F.log_softmax(model(x).float(), 1).sum()
            g = torch.autograd.grad(lp, x, create_graph=True)[0].pow(2).sum(1, keepdim=True)
            return ((g * (1 - region)).sum((1, 2, 3)) / (g.sum((1, 2, 3)) + 1e-12)).mean()
        logits, f3 = model(x.float(), return_feat=True)
        g = torch.autograd.grad(logits.float().gather(1, y[:, None]).sum(), f3, create_graph=True)[0]
        cam = F.relu((g.mean((2, 3), keepdim=True) * f3).sum(1, keepdim=True))
        r = F.adaptive_avg_pool2d(region, cam.shape[-2:])
        return ((cam * (1 - r)).sum((1, 2, 3)) / (cam.sum((1, 2, 3)) + 1e-6)).mean()

def close_loader(loader):
    """Stop the DataLoader worker processes. Without this, every finished model left 4 workers
    alive in RAM, which crashed the kernel ("allocate more memory than available")."""
    it = getattr(loader, "_iterator", None)
    if it is not None and hasattr(it, "_shutdown_workers"):
        it._shutdown_workers()
    loader._iterator = None

def train_one(name, seed, fold, lr=None, max_epochs=NUM_EPOCHS, patience=PATIENCE, save=True, show_every=10):
    lr = lr or get_lr(name)
    set_seed(seed * 100 + fold)
    mean, std = norm_for(name, fold)
    train_items, val_items = FOLDS[fold]["train"], FOLDS[fold]["val"]
    newp = new_pipeline(name) and bool(MASKS)
    spec = MODEL_SPECS[name]
    use_loc = bool(spec.get("loc"))
    if newp:                                      # Run-3 models: GPU augmentation + masks + hidden versions
        seg, cf, guide = bool(spec.get("seg")), bool(spec.get("cf")), spec.get("guide")
        ds = (TransplantSet(train_items) if spec.get("transplant")                       # Run 5, solution 2
              else GuidedSet(train_items, strip=bool(spec.get("strip"))))               # Run 5, skull control
    elif is_roi(name):                            # Run-4 remedy: train on expert-mask crops
        seg, cf, guide = False, False, None
        ds = RoiSet(train_items, train_transform(mean, std), spec["crop"],
                    margin=spec.get("margin", ROI_MARGIN), loc=use_loc, transplant=bool(spec.get("transplant")))
    else:
        seg, cf, guide = uses_masks(name) and bool(MASKS) and spec.get("use_seg", False), False, None
        ds = ImageMaskList(train_items, mean, std) if seg else ImageList(train_items, train_transform(mean, std))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, persistent_workers=True)
    val_raw = [load_u8(p) for p, _ in val_items]
    val_locs = None
    if is_roi(name):                              # early stopping on expert-mask crops (no test information)
        mg = spec.get("margin", ROI_MARGIN)
        val_arrs = [crop_roi(a, box_from_mask(MASKS[p]), spec["crop"], margin=mg) if p in MASKS and box_from_mask(MASKS[p]) else a
                    for a, (p, _) in zip(val_raw, val_items)]
        if use_loc:
            val_locs = np.stack([loc_feats(a, MASKS[p]) if p in MASKS else np.zeros(N_LOC, np.float32)
                                 for a, (p, _) in zip(val_raw, val_items)])
    elif spec.get("strip"):
        val_arrs = [skull_strip(a) for a in val_raw]
    else:
        val_arrs = val_raw
    val_y = np.array([y for _, y in val_items])
    model = build_model(name, pretrained=True).to(DEVICE)
    if is_pretrained(name) and USE_DP_FOR_PRETRAINED:
        model = nn.DataParallel(model)
    groups = model.param_groups(lr) if hasattr(model, "param_groups") else [{"params": list(model.parameters()), "base_lr": lr}]
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    w = class_weights(train_items)
    ce = lambda out, t: F.cross_entropy(out, t, weight=w, label_smoothing=LABEL_SMOOTHING)
    best = {"f1": -1, "epoch": -1, "state": None}
    hist = {"train_loss": [], "val_f1": [], "val_acc": [], "lr": []}
    wait, t_start = 0, time.time()
    try:
      for epoch in range(max_epochs):
          for g in opt.param_groups:
              g["lr"] = g["base_lr"] * lr_factor(epoch, max_epochs)
          model.train(); tot = 0.0; t_ep = time.time()
          for batch in loader:
              if newp:
                  st, has, yb = (t.to(DEVICE, non_blocking=True) for t in batch)
                  imgs, masks = gpu_augment(st)
                  x0, xh, xr = (norm3(imgs[:, k:k + 1], mean, std) for k in range(3))
                  mb, reg = (masks[:, :1] if seg else None), masks[:, 1:2]
                  hb = has
                  xb = x0
              elif seg:
                  xb, mb, hb, yb = (t.to(DEVICE, non_blocking=True) for t in batch)
              elif use_loc:
                  xb, lb, yb = (t.to(DEVICE, non_blocking=True) for t in batch); mb = None
              else:
                  (xb, yb), mb = (t.to(DEVICE, non_blocking=True) for t in batch), None
              xb, ya, yb2, lam, mb, idx = mix_batch(xb, yb, mb)
              if use_loc and idx is not None:
                  lb = lam * lb + (1 - lam) * lb[idx]       # location numbers are mixed like the images
              opt.zero_grad(set_to_none=True)
              with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                  if seg:
                      out, seg_logit = model(xb, return_seg=True)
                  elif use_loc:
                      out = model(xb, loc=lb)
                  else:
                      out = model(xb)
                  loss = lam * ce(out, ya) + (1 - lam) * ce(out, yb2) if lam < 1 else ce(out, ya)
                  if seg:
                      hb_m = hb * hb[idx] if idx is not None else hb
                      loss = loss + SEG_LOSS_WEIGHT * seg_loss_fn(seg_logit, mb, hb_m)
                  if cf:                          # counterfactual rules on half of the batch (unmixed images)
                      sel = torch.nonzero(has[: xb.shape[0] // 2] > 0).squeeze(1)
                      if len(sel):
                          lh = model(xh[sel]).float()
                          loss = loss + CF_HIDE_WEIGHT * (-F.log_softmax(lh, 1).mean(1)).mean()   # -> uniform
                          loss = loss + CF_KEEP_WEIGHT * ce(model(xr[sel]), yb[sel])              # -> same label
              if guide:                           # RRR / CAM-Guide penalty (fp32, double backprop, few images)
                  sel = torch.nonzero(has > 0).squeeze(1)[:GUIDE_BATCH]
                  if len(sel):
                      loss = loss + GUIDE_WEIGHT * guide_penalty(model, x0[sel], reg[sel], yb[sel], guide)
              scaler.scale(loss).backward()
              scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
              scaler.step(opt); scaler.update()
              tot += loss.item()
          logits = (predict_logits_loc(model, val_arrs, val_locs, mean, std) if use_loc
                    else predict_logits(model, val_arrs, mean, std))
          pred = logits.argmax(1)
          f1, acc = macro_f1(val_y, pred), float((pred == val_y).mean())   # (validation crops always contain a tumour)
          hist["train_loss"].append(tot / len(loader)); hist["val_f1"].append(f1)
          hist["val_acc"].append(acc); hist["lr"].append(opt.param_groups[0]["lr"])
          improved = f1 > best["f1"] + 1e-4
          if improved:
              m = model.module if isinstance(model, nn.DataParallel) else model
              best = {"f1": f1, "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}}
              wait = 0
          else:
              wait += 1
          if epoch == 0:
              print(f"    1 epoch = {time.time() - t_start:.0f}s  ->  max {(time.time() - t_start) * max_epochs / 60:.0f} min "
                    f"for this {'test' if not save else 'job'} (stops earlier if no progress for {patience} epochs)", flush=True)
          if show_every and ((epoch + 1) % show_every == 0):
              print(f"    epoch {epoch + 1:2d}: val F1 now {100 * f1:4.1f}% | best so far {100 * best['f1']:4.1f}% "
                    f"| {ram_text()}", flush=True)
          if wait >= patience:
              print(f"    stopped early at epoch {epoch + 1} (no progress for {patience} epochs)", flush=True)
              break
    finally:
        close_loader(loader)
    sec_per_epoch = (time.time() - t_start) / (epoch + 1)
    m = model.module if isinstance(model, nn.DataParallel) else model
    m.load_state_dict(best["state"])
    # final out-of-fold scores END-TO-END (ROI models: box from the predicted tumour map, not the expert mask)
    _CUR_SET["name"] = "figshare"
    logits = predict_any(name, fold, m, val_raw, mean, std)
    pred = hard(softmax_np(logits))           # -1 = no tumour found (abstained)
    rec = {"model": name, "seed": seed, "fold": fold, "lr": lr, "uses_masks": seg, "counterfactual": cf,
           "guide": guide, "best_epoch": best["epoch"] + 1,
           "epochs_run": epoch + 1, "sec_per_epoch": round(sec_per_epoch, 1),
           "val_f1": macro_f1(val_y, pred), "val_acc": float((pred == val_y).mean()),
           "val_auc": macro_auc(val_y, softmax_np(logits)), "n_val": len(val_y),
           "n_params": sum(p.numel() for p in m.parameters()), "history": hist}
    if save:
        ckpt, rec_path, oof_path = run_paths(name, seed, fold)
        # half precision on disk (2x smaller zips); weights are cast back to float when loaded
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in best["state"].items()}, ckpt)
        rec["ckpt"], rec["sha256"] = ckpt.name, sha256_file(ckpt)
        rec["val_paths"], rec["val_labels"] = [p for p, _ in val_items], val_y.tolist()
        np.save(oof_path, logits)
        save_json(rec, rec_path)
    del model, m, opt, loader
    gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return rec

# %%
# ─── 3.2 (Session 1 only) Short learning-rate search on fold 1 ───
# Same budget for every model: max 30 epochs, patience 8, fold 1, seed 0. Chosen LR = best val macro-F1.
if RUN_TRAINING and RUN_LR_SEARCH:
    choice = load_json(OUT_DIR / "lr_choice.json", {})
    for name in LR_SEARCH_MODELS:
        if name in choice:
            continue
        if (time.time() - SESSION_START) / 3600 > MAX_TRAIN_HOURS:
            print("Time budget reached during LR search - it continues in the next run."); break
        grid = LR_GRID["pretrained" if is_pretrained(name) else "scratch"]
        k = LR_SEARCH_MODELS.index(name) + 1
        banner(f"STEP 1/2  Picking the learning rate for {name}  (model {k} of {len(LR_SEARCH_MODELS)})",
               f"short test runs on fold 1, nothing saved | time used {hms(time.time() - SESSION_START)}")
        scores = {}
        for i, lr in enumerate(grid, 1):
            print(f"  Test {i}/{len(grid)}: learning rate {lr} ...", flush=True)
            scores[lr] = train_one(name, 0, 1, lr=lr, max_epochs=30, patience=8, save=False, show_every=0)["val_f1"]
            print(f"  Test {i}/{len(grid)}: learning rate {lr} -> score (val F1) {100 * scores[lr]:.1f}%", flush=True)
        choice[name] = max(scores, key=scores.get)
        print(f"\n  ✔ {name} will be trained with learning rate {choice[name]}  "
              f"(" + ", ".join(f"{l}: {100 * v:.1f}%" for l, v in scores.items()) + ")")
        save_json(choice, OUT_DIR / "lr_choice.json")
if RUN_TRAINING:
    print("\nLearning rates (ablations reuse their parent's LR):")
    for n in MODEL_ORDER:
        print(f"   {n:18s} {get_lr(n)}" + ("" if n in load_json(OUT_DIR / "lr_choice.json", {}) or n in LR_INHERIT
                                            else "   (not tested yet)"))

# %%
# ─── 3.3 Train the queue (skips finished runs; stops before the Kaggle time limit) ───
lr_done = all(m in load_json(OUT_DIR / "lr_choice.json", {}) for m in LR_SEARCH_MODELS) or not RUN_LR_SEARCH
if RUN_TRAINING and lr_done:
    jobs = [(n, s, f) for n, seeds in TRAIN_QUEUE for s in seeds for f in range(1, N_FOLDS + 1)]
    n_done0 = sum(run_done(*j) for j in jobs)
    banner("STEP 2 of 2 - TRAINING THE REAL MODELS (saved, used in the paper)",
           f"{n_done0} of {len(jobs)} jobs already finished. 1 job = 1 model x 1 seed x 1 fold (max {NUM_EPOCHS} epochs).\n"
           f"  Training stops by itself after {MAX_TRAIN_HOURS} h; the rest continues in the next run.\n"
           f"  elapsed {hms(time.time() - SESSION_START)} | {ram_text()}")
    for name, seeds in TRAIN_QUEUE:
        for seed in seeds:
            for fold in range(1, N_FOLDS + 1):
                if run_done(name, seed, fold):
                    continue                                   # already finished in an earlier run
                if (time.time() - SESSION_START) / 3600 > MAX_TRAIN_HOURS:
                    print("Time budget reached - remaining runs left for the next session."); break
                n_done = sum(run_done(*j) for j in jobs)
                print("\n" + "-" * 70 + f"\nJOB {n_done + 1}/{len(jobs)} | {name} | seed {seed} | fold {fold} of {N_FOLDS} "
                      f"| lr {get_lr(name)}\n  {role_of(name)} | elapsed {hms(time.time() - SESSION_START)}\n" + "-" * 70,
                      flush=True)
                r = train_one(name, seed, fold)
                print(f"  ✔ JOB {n_done + 1} SAVED: val F1 {100 * r['val_f1']:.1f}% | val acc {100 * r['val_acc']:.1f}% "
                      f"| best epoch {r['best_epoch']} of {r['epochs_run']} run")

# %%
# ─── 3.4 Which runs are complete? ───
todo = [(n, s, f) for n, seeds in TRAIN_PLAN for s in seeds for f in range(1, N_FOLDS + 1) if not run_done(n, s, f)]
total = sum(len(seeds) * N_FOLDS for _, seeds in TRAIN_PLAN)
print("=" * 70)
if not lr_done:
    print("STATUS: learning-rate search not finished -> run the notebook again.")
elif todo:
    print(f"STATUS: {total - len(todo)} of {total} training runs done -> run the notebook again.")
    print(f"        next up: {todo[0][0]} seed {todo[0][1]} fold {todo[0][2]}")
else:
    print("STATUS: ALL TRAINING DONE. The tables below are final.")
print("=" * 70)
def complete_seeds(name):
    return [s for s in range(10) if all(run_done(name, s, f) for f in range(1, N_FOLDS + 1))]

READY = {n: complete_seeds(n) for n in MODEL_ORDER}
READY = {n: s for n, s in READY.items() if s}
plan_seeds = defaultdict(set)
for n, seeds in TRAIN_PLAN:
    plan_seeds[n] |= set(seeds)
print(f"\n{'Model':15s} {'seed 0':>7s} {'seed 1':>7s} {'seed 2':>7s}   role")
for n in MODEL_ORDER:
    if n not in plan_seeds:
        continue
    cells = []
    for s in range(3):
        if s not in plan_seeds[n]:
            cells.append("  -  ")
        else:
            k = sum(run_done(n, s, f) for f in range(1, N_FOLDS + 1))
            cells.append("  ✔  " if k == N_FOLDS else f" {k}/5 ")
    print(f"{n:15s} {cells[0]:>7s} {cells[1]:>7s} {cells[2]:>7s}   {role_of(n)[:60]}")
print("(✔ = all 5 folds trained; only models with ✔ appear in the tables below)")

# The proposed model (pre-registered): ROI-Loc; until it is trained, ROI-Crop (then CTR-Net, MTK)
PROPOSED = next((n for n in ["ROI-Loc", "ROI-Crop", "CTR-Net", "MTK"] if n in READY), "MTK")
print(f"Main model in all comparisons: {PROPOSED}  ({role_of(PROPOSED)})")
XAI_MODELS = [n for n in ["CNX-Transplant", "CTR-Net", "CTR-noCF", "MTL-Seg", "RRR", "CAM-Guide", "CNX-Plain", "MTK", "ResNet-50"] if n in READY]
if RUN_XAI == "auto":
    main_needed = ["CTR-Net", "CTR-noCF", "MTL-Seg", "RRR", "CAM-Guide", "CNX-Plain"]
    RUN_XAI = all(n in READY for n in main_needed)
    print("Explainability (section 9) will run now (cached per model, so later runs are fast)." if RUN_XAI else
          "Explainability (section 9) waits until CTR-Net and the comparison methods are trained.")

# %%
# ─── 3.5 Save the artifact ZIP now (so training is safe even if a later cell fails) ───
make_artifact_zip()

# %%
# ─── 3.6 ⬇ DOWNLOAD NOW (training is finished; everything below only makes tables and figures) ───
from IPython.display import FileLink, display, HTML
def show_downloads():
    zips = sorted(WORK.glob(f"{ZIP_PREFIX}_*.zip"))
    display(HTML("<div style='padding:10px;border:3px solid #d1495b;border-radius:8px;font-size:15px'>"
                 "<b>⬇ Download these files NOW</b> (click each link; keep this tab open until all are saved).<br>"
                 "Then upload them as ONE new Kaggle dataset (e.g. <code>mtk-run5-artifacts</code>) and attach it next run, "
                 "together with <code>mtk-run3-part2</code> and <code>mtk_planA_artifacts</code>.</div>"))
    for z in zips:
        display(FileLink(str(z.relative_to(WORK)), result_html_prefix=f"{z.stat().st_size / 1e9:.2f} GB &nbsp; "))
    if not zips:
        print("No zip found - run cell 3.5 again.")
show_downloads()
