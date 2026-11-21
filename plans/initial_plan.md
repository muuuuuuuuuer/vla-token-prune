# Initial Plan: Spatio-Temporal Visual Token Pruning IP for VLA Policies

Repository: `<GitHub URL of this repo>`

## Project Team

- Hao Zhong (hz4324)
- Yaoqi Xia (yx3954)

## IP Definition

### What the IP does

Vision-Language-Action (VLA) policies (for example, OpenVLA-class models) turn every camera frame into a grid of visual tokens. A vision encoder processes those tokens, and then a language backbone does too. The policy runs at only a few Hz, while a robot needs actions at 10-50 Hz. In a manipulation episode seen from a fixed camera, most of the image (table, background, robot base) does not change from one frame to the next. Recomputing every token each frame is mostly wasted work.

Our IP is a **streaming front-end that decides, for every frame, which visual tokens must be recomputed**. It sits between the camera stream and the host model. For each 224x224 RGB frame it compares every 16x16 patch (14x14 = 196 tokens) against the version of that patch that was **last sent** to the model. It then applies spatial and temporal rules and emits a compact list of the token indices to recompute. The host recomputes only those tokens and reuses cached features or KV entries for the rest. The IP also emits a frame-level "nothing changed" flag that the host can use to reuse the previous action chunk.

The project has a fixed structure. We first build a **baseline that keeps all tokens**, with the same interfaces and test harness. Each pruning rule below is then added as a variant that can be switched on at runtime. All variants are measured with the same verification flow, and the best-performing rules are combined and optimized in the final design.

### Mathematical operations

All arithmetic is unsigned fixed-point integer. Notation: frame `t`, patch `p` in 0..195, pooled block `b` in 0..15 (each patch is pooled into 4x4 blocks of 4x4 pixels).

1. **Luma:** `Y = (77R + 150G + 29B) >> 8` (BT.601 integer approximation, 8-bit).
2. **Block pooling:** `Yhat[p,b] = (sum of Y over the 4x4 pixels of block b) >> 4` (8-bit).
3. **Patch difference against the last-sent reference:** `D[p] = sum_b |Yhat[p,b] - R[p,b]|` (12-bit, max 4080).
4. **Change map and 3x3 dilation:** `c[p] = D[p] > tau`. Then `m[p] = OR of c[q]` over the 3x3 neighborhood of `p` on the 14x14 grid, with zero padding at the borders.
5. **Age refresh:** `m[p] |= age[p] >= A_max`. This bounds how long a reused token can go stale.
6. **Token budget (optional):** if `sum(m) > K_max`, keep exactly `K_max` tokens. The priority is `s[p] = min(D[p] + (age[p] << w_age), 4095)`, and the top-K selection uses a 64-bin histogram.
7. **Commit:** for kept tokens, `R[p] <- Yhat[p]` and `age[p] <- 0`. For all other tokens, `age[p] <- min(age[p]+1, 255)`.
8. **Global reuse flag:** `G = sum_p D[p]` (20-bit). Set `reuse = (G < tau_g)`.

Python golden model (the reference for bit-exact testing):

```python
IMG, P, B = 224, 16, 4                    # image, patch, pooling block
G_ = IMG // P                             # 14 -> N = 196 tokens
NB = (P // B) ** 2                        # 16 pooled blocks per patch

def prune_frame(rgb, st, cfg):
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    y = (77*r + 150*g + 29*b) >> 8                                   # luma_pool
    blk = y.reshape(IMG//B, B, IMG//B, B).sum(axis=(1, 3)) >> 4      # (56,56)
    cur = blk.reshape(G_, P//B, G_, P//B).transpose(0, 2, 1, 3).reshape(-1, NB)
    D = np.abs(cur - st.ref).sum(axis=1)                             # patch_diff
    c = (D > cfg.tau).reshape(G_, G_)                                # mask_gen
    keep = (dilate3x3(c) if cfg.dilate else c).reshape(-1)
    if cfg.age_refresh:
        keep |= st.age >= cfg.a_max
    if cfg.budget and keep.sum() > cfg.k_max:                        # budget_select
        s = np.minimum(D + (st.age.astype(np.int32) << cfg.w_age), 4095)
        keep = hist_topk(keep, s, cfg.k_max)
    if st.first or cfg.keep_all:                                     # baseline / episode start
        keep[:] = True
    upd = np.ones_like(keep) if cfg.ref_update_all else keep         # ablation switch
    st.ref[upd] = cur[upd]                                           # ref_commit
    st.age = np.where(keep, 0, np.minimum(st.age + 1, 255))
    st.first = False
    return np.flatnonzero(keep), D, D.sum() < cfg.tau_g

def hist_topk(keep, s, K):        # exact count K, 64-bin approximate ranking
    q = s >> 6
    cum = np.cumsum(np.bincount(q[keep], minlength=64)[::-1])[::-1]  # cum[b] = #(bin >= b)
    bstar = max(i for i in range(64) if cum[i] >= K)
    take = keep & (q > bstar)
    for p in np.flatnonzero(keep & (q == bstar)):                    # fill in index order
        if take.sum() == K:
            break
        take[p] = True
    return take
```

### Why it suits hardware acceleration

- **Streaming, fixed-point, single pass.** Every operation consumes pixels in raster order at one pixel per cycle. The design needs line accumulators and about 10 KB of on-chip state, with no frame buffer and no floating point. This is the line-buffer structure FPGAs are built for, and it can run inline on the camera path.
- **Latency-critical.** The pruning decision has to be ready almost as soon as the last pixel of a frame arrives, or it delays the whole control loop. Our target is one patch row (3,584 cycles, about 36 us at 100 MHz) after the last pixel. Done on the PS, the same work would compete with the policy and control software for the two ARM cores, and its latency would vary.
- **Parallel and bit-level.** The 196 patch differences are independent. The 3x3 dilation and the masks are bit operations. The histogram top-K uses small counters. All of these map to LUTs and need almost no DSPs.
- **Leverage.** The IP's value is the downstream encoder work it removes. A ViT-B/16 forward pass at 224x224 costs about 17.5 GFLOPs, and the work skipped scales with the fraction of tokens pruned. That only pays off if token selection itself is essentially free in latency and CPU time, which is what a hardware front-end provides.

### How it interacts with the PS

The PS replays recorded robot episodes. It writes frames to DDR, sets the thresholds through AXI-Lite, and starts an AXI DMA transfer that streams the frame into the IP. The IP keeps its reference and age state on-chip across frames. The PS sets `reset_ref` at the start of each episode, which forces all tokens to be kept on the next frame. For each frame, the IP streams back one 32-bit record per kept token, followed by a trailer word. The PS reads per-frame statistics (token count, global difference, cycle count) from AXI-Lite status registers.

## IP Architecture

### Sub-modules

1. **`luma_pool`**: reads one RGB pixel per cycle, computes `Y`, and accumulates 4x4 block sums in a 56-entry line accumulator. Every 4 pixel rows it emits 56 block means. This stage sets the pixel rate (II = 1).
2. **`patch_diff`**: maps each block to its `(p, b)`, reads `R[p,b]` from `ref_mem`, and accumulates `|Yhat - R|` into 14 partial sums, one per patch in the current patch row. It emits 14 values of `D[p]` per patch row and writes `Yhat` into a staging buffer for the later commit.
3. **`mask_gen`**: keeps 3 rows of change bits (3x14 bits) for the 3x3 dilation, so its output lags its input by one patch row. It ORs in the age rule, attaches reason flags (changed, dilated, aged, forced), and accumulates `G`.
4. **`budget_select`**: when the budget is enabled, it buffers the 196 records, builds a 64-bin histogram, finds `bstar`, and outputs exactly `K_max` records in index order. This adds a frame-level barrier. When the budget is disabled, records pass straight through.
5. **`token_writer`**: packs the kept records into 32-bit output words, counts them, and appends a trailer word with `TLAST`. It also writes the 196-bit keep mask for `ref_commit`.
6. **`ref_commit`**: runs after the dataflow region of each frame. It copies staging into `ref_mem` for kept patches (or for all patches when `ref_update_all` is set) and updates `age`. With the reference partitioned 16 bytes wide, this takes 196 cycles.

Modules 1-5 run as an HLS `DATAFLOW` region connected by `hls::stream`, so each stage overlaps with the next (load, compute, decide, store). The commit step is placed after the region on purpose. `ref_mem` is read by `patch_diff` during frame `t`, but which entries get updated depends on the keep mask, which is only final at the end of frame `t`. Committing after the region avoids a feedback loop between dataflow stages at a cost of about 200 cycles per frame.

### Interfaces

| From → To | Interface | What it carries |
|---|---|---|
| PS DDR → `luma_pool` | AXI4-Stream, 32-bit, via AXI DMA MM2S | One pixel per beat `{0, B, G, R}` (8 bits each). 50,176 beats per frame, `TLAST` on the last pixel. |
| PS ↔ IP | AXI-Lite (`s_axilite`) | Control: `tau` (12b), `tau_g` (20b), `A_max` (8b), `K_max` (8b), `w_age` (3b), `mode` bits `{keep_all, dilate, age_refresh, budget, ref_update_all}`, `reset_ref`, `ap_start`. Status: `n_kept`, `G`, `frame_cnt`, `cycles`. |
| `luma_pool` → `patch_diff` | `hls::stream<ap_uint<8>>` | Block means in block-row-major order. 3,136 per frame, one every 16 cycles on average. |
| `ref_mem` → `patch_diff` | On-chip BRAM read port | `R[p,b]`, 8-bit. 196x16 B = 3,136 B. |
| `patch_diff` → staging | On-chip BRAM write | `Yhat[p,b]` for the current frame (3,136 B). |
| `patch_diff` → `mask_gen` | `hls::stream<ap_uint<12>>` | `D[p]`: 14 values per patch row, 196 per frame. |
| `age` table → `mask_gen` | On-chip BRAM/LUTRAM read | `age[p]`, 8-bit, 196 entries. |
| `mask_gen` → `budget_select` | `hls::stream<token_rec>` | All 196 records `{idx 8b, D 12b, age 8b, flags 4b}` with a keep bit, plus an end-of-frame record carrying `G`. |
| `budget_select` → `token_writer` | `hls::stream<token_rec>` | Kept records only, followed by an end marker. |
| `token_writer` → PS DDR | AXI4-Stream, 32-bit, via AXI DMA S2MM | Per kept token: `[7:0] idx, [19:8] D, [27:20] age, [31:28] flags`. Trailer: `[15:0] frame_id, [23:16] n_kept, [24] reuse`, with `TLAST`. At most 197 words. |
| `token_writer` → `ref_commit` | On-chip array (196 bits) | Keep mask for the reference update. |
| `ref_commit` ↔ `ref_mem`, `age` | On-chip BRAM read/write | Reference and age update after each frame. |

On-chip memory totals about 10 KB: 3,136 B reference, 3,136 B staging, 196 B age, 196 buffered records, and small line accumulators. Luma uses three constant multiplies, which can be built from shift-adds. The design therefore fits easily on a PYNQ-Z2 (XC7Z020), leaving room for unrolling. Performance targets: II = 1 on the pixel loop, about 50,200 cycles per frame (about 0.5 ms at 100 MHz), and at most 4,000 cycles from the last pixel to `TLAST`.

### Why this decomposition

The stages run at very different rates: one pixel per cycle, one block per 16 cycles, and 14 tokens per patch row. Splitting them into streaming stages lets each stage be pipelined at its own rate, and the front-end never stalls on decision logic. Each module corresponds to a few lines of `prune_frame`, so it can be unit-tested against the golden model on its own: block means, `D`, masks, selected indices, and post-commit `ref` and `age`.

The split also supports our development plan. The **baseline** (`keep_all`) uses the full I/O path, DMA, harness, and cycle and resource measurement with no pruning. Each pruning rule then lives in a single module and is switched on by one `mode` bit, so all variants share the same bitstream and test flow:

- **V1, temporal difference:** `tau` only.
- **V1-prev, ablation:** `ref_update_all` makes the IP compare against the previous frame instead of the last-sent patch, which shows slow-drift failures.
- **V2:** V1 plus 3x3 dilation.
- **V3:** V2 plus age refresh.
- **V4:** V3 plus the `K_max` budget.
- **V5:** the global reuse flag.

Every variant is scored on the same replayed episodes (third-person views from public datasets such as BridgeData V2 or LIBERO) with the same metrics:

- bit-exact agreement with the golden model, including the number of output words;
- cycles per frame and latency;
- LUT, FF, BRAM, and DSP usage;
- token keep ratio;
- feature fidelity: the cosine similarity between a frozen ViT's features on the true frame and on the frame the host effectively sees (unkept patches replaced by their last-sent version).

We then pick the Pareto-best combination of keep ratio versus fidelity, remove unused paths at compile time, and optimize the result (partitioning, unrolling, II) for the final design.

### Existing IP

AMD AXI DMA handles the MM2S and S2MM transfers. For a live camera later, the AMD Video Processing Subsystem could do color conversion and scaling in front of `luma_pool`. We found no existing AMD IP that does reference-based patch differencing, dilation, or histogram top-K, so we design those modules ourselves.
