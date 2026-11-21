# Initial Plan: Visual Token Differential Pruning IP for VLA Models

[Project repository](https://github.com/muuuuuuuuuer/vla-token-prune)

## Project Team

- Hao Zhong (hz4324)
- Yaoqi Xia (yx3954)

## IP Definition

**In one sentence:** our FPGA IP streams in each camera frame and tells the host which `K` of the 196 image patches changed most since they were last refreshed. The VLA model then re-encodes only those patches and reuses its cached features for the rest.

### Terms used in this plan

| Term | Meaning |
|---|---|
| VLA model | The vision-language-action network on the host that controls the robot. It is not part of the IP. |
| Position / token | One of the 14×14 = 196 patches (16×16 pixels each) of a 224×224 frame. This grid matches the ViT-B/16 encoder we use for evaluation. |
| Descriptor | A 16-byte summary of one patch, with one byte per 4×4 pixel block. |
| Score | How far a patch's current descriptor is from the descriptor saved the last time that position was refreshed. |
| Refresh | Selecting a position. The host re-encodes that patch, and the IP saves the patch's current descriptor as its new reference. |
| Selector | One hardware pruning design: a descriptor type combined with a selection scope. We compare four selectors (2×2). |
| Oracle | An offline, software-only choice that uses dense encoder features to find the best possible `K` positions. |

### What the IP does

- **Input:** a 224×224 RGB frame from the PS.
- **Output:** exactly `K_eff` distinct patch indices with their scores, followed by one trailer word.
- **Budget:** the host sets `K` (0–196) for each frame. `K_eff = K`, except after `reset_ref` (the first frame of each episode), when `K_eff = 196` and every position is refreshed.
- **Host side:** the host re-encodes the selected patches and keeps the cached token at every other position. Model execution and the feature cache stay on the host. [VLA-Cache](https://github.com/siyuhsu/vla-cache) is an existing example of cross-frame token reuse in a VLA system.

**Research question:** with the same refresh budget `K`, can a streaming FPGA selector pick better patches if it keeps a little spatial information, and how much selection quality does it lose by emitting indices earlier? Two cases motivate the question:

- **Moving edge:** a gripper edge can move inside a 4×4 block while the block's mean brightness stays the same, so a mean-only descriptor misses it.
- **Exposure shift:** a global brightness change shifts every background block's mean, so a mean-only descriptor spends the budget on patches whose content did not change.

A fixed `K` makes the comparison fair: every selector refreshes the same number of positions on every frame, including static frames.

### How it interacts with the PS

1. At the start of each frame, the PS writes `K` and, at the start of each episode, `reset_ref` through AXI-Lite.
2. The PS replays a recorded frame from DDR through AXI DMA (MM2S) into the IP's input stream.
3. The IP returns `K_eff` index words and one trailer through a second AXI DMA (S2MM) into DDR.
4. The IP keeps all reference descriptors on-chip across frames. It does not access DDR directly.

### Mathematical operations

All operations are integer. Each patch holds 16 blocks of 4×4 pixels.

1. **Luma:** `Y = (77R + 150G + 29B) >> 8`, giving 8 bits.
2. **Block statistics:** for each 4×4 block, compute the mean `m = sum(Y) >> 4`. For the edge descriptor only, also compute four half-block means of 8 pixels each (`sum >> 3`): left, right, top and bottom.
3. **Descriptor byte for each block.** We compare two options:

   | Descriptor | Byte layout | Per-block distance |
   |---|---|---|
   | Mean | `[7:0] = m` | `abs(Δm)` |
   | Mean + edge | `[7:4] = m >> 4`, `[3:2] = h`, `[1:0] = v` | `4·abs(Δmean4) + abs(Δh) + abs(Δv)` |

   For the edge descriptor, `h = tern(right − left)` and `v = tern(bottom − top)`. The function `tern(x)` returns +1 if `x > 16`, −1 if `x < −16`, and 0 otherwise. It is stored as a 2-bit code: `00` = 0, `01` = +1, `11` = −1. Fields are decoded to signed values before subtracting. The threshold 16 and the weight 4 are starting values that the software pilot will tune.
4. **Patch score:** the sum of the 16 per-block distances. This is at most 4,080 for the mean descriptor and 1,024 for the edge descriptor, so 12 bits are enough.
5. **Exact top-K by rank counting:** `rank[p] = #{q : s[q] > s[p], or s[q] = s[p] and q < p}`. Position `p` is selected when `rank[p] < K`. Ties therefore go to the lower raster index, and the selected indices come out in raster order.
6. **Selection scope:** we compare two options.
   - **Frame scope:** rank all 196 scores together.
   - **Band scope:** split the frame into 7 bands, each two patch rows (28 positions). Band `j`, counting from 0, gets `floor((j+1)·K/7) − floor(j·K/7)` slots. The quotas always add up to `K`, and no band gets more than 28.
7. **Reference update:** only selected positions save their current descriptor. At skipped positions, small changes keep adding to the score until that position is selected.

Integer golden model (the reference for bit-exact tests):

```python
def tern(x):  return np.where(x > 16, 1, np.where(x < -16, -1, 0))
def enc(t):   return t & 3                       # 0->00, +1->01, -1->11
def dec(c):   return np.where(c == 3, -1, c)

def descriptors(rgb, kind):                      # -> (196, 16) per-patch descriptors
    R, G, B = (rgb[..., i].astype(np.int32) for i in range(3))
    y = (77*R + 150*G + 29*B) >> 8
    blk = y.reshape(56, 4, 56, 4)                # [block_row, row, block_col, col]
    m = blk.sum(axis=(1, 3)) >> 4
    if kind == "mean":
        d = m
    else:
        half = lambda b: b.sum(axis=(1, 3)) >> 3
        h = tern(half(blk[:, :, :, 2:]) - half(blk[:, :, :, :2]))   # right - left
        v = tern(half(blk[:, 2:, :, :]) - half(blk[:, :2, :, :]))   # bottom - top
        d = ((m >> 4) << 4) | (enc(h) << 2) | enc(v)
    return d.reshape(14, 4, 14, 4).transpose(0, 2, 1, 3).reshape(196, 16)

def score(cur, ref, kind):                       # -> (196,) 12-bit scores
    if kind == "mean":
        return np.abs(cur - ref).sum(axis=1)
    dm = np.abs((cur >> 4) - (ref >> 4))
    dh = np.abs(dec((cur >> 2) & 3) - dec((ref >> 2) & 3))
    dv = np.abs(dec(cur & 3) - dec(ref & 3))
    return (4*dm + dh + dv).sum(axis=1)

def top_k(s, pos, k):                            # exact; ties -> lower index; raster order
    rank = np.array([(s > s[i]).sum() + (s[:i] == s[i]).sum() for i in range(len(s))])
    return pos[rank < k]

def frame_step(rgb, st, K, kind, scope):         # st.ref: (196, 16) int32, st.reset: bool
    cur = descriptors(rgb, kind)
    if st.reset:                                 # K_eff = 196, scores reported as 0
        sel, s = np.arange(196), np.zeros(196, np.int32)
    else:
        s = score(cur, st.ref, kind)
        if scope == "frame":
            sel = top_k(s, np.arange(196), K)
        else:                                    # 7 bands x 28 positions
            sel = np.concatenate([top_k(s[28*j:28*j+28], np.arange(28*j, 28*j+28),
                                        (j+1)*K//7 - j*K//7) for j in range(7)])
    st.ref[sel] = cur[sel]                       # commit selected positions only
    st.reset = False
    return sel, s[sel]
```

### Why it suits hardware acceleration

- **Latency-critical.** The host cannot start re-encoding until it knows which patches to refresh. In the band build, the IP can return the first indices while later pixels are still arriving. Our target is about 0.08 ms after the first pixel, compared with about 0.53 ms for the frame build. That gap in latency, set against any loss in selection quality, is exactly what we will measure. A software selector on the PS could only start after the whole frame was in memory, and it would compete with the host's control software for the two ARM cores.
- **Streaming and fixed-point.** Luma, block sums and reference comparisons consume one pixel per cycle in raster order, with no frame buffer. On-chip state is bounded at about 6.5 KB, and there is no floating point.
- **Parallel.** The 196 scores are independent. Rank counting is a grid of 12-bit comparisons that can be unrolled across any number of comparators: frame scope needs 196×196 comparisons, about 2,744 cycles with 14 comparators in parallel, and band scope needs 28×28.
- **Leverage.** The selector's value is the encoder work it lets the host skip. A ViT-B/16 forward pass at 224×224 costs about 17.5 GFLOPs. The selector itself must therefore add almost no latency or CPU load, which a pixel-rate hardware stage achieves.

## IP Architecture

### Sub-modules

1. **Descriptor builder.** Reads one pixel per cycle and computes `Y`. Its row accumulators, one per 4×4 block column (56 in all), hold the block sum and, for the edge build, the half-block sums. After every 16 pixel rows (one patch row) it emits 14 descriptor records. It depends only on the descriptor type.
2. **Patch scorer.** For each record, it reads the saved descriptor for that position from the reference RAM and computes the score. It also writes the current descriptor to the staging RAM for a possible refresh. After `reset_ref` it reports a score of 0.
3. **Budget selector.** Buffers scores and runs rank counting. The frame build waits for all 196 scores. The band build ranks each band of 28 as soon as that band's two patch rows are scored. Both builds emit selected `(position, score)` records in raster order. It depends only on the selection scope.
4. **Packet writer.** Packs each selected record into a 32-bit output word and appends a trailer with `TLAST`. It also logs the selected positions for the reference update.
5. **Reference commit.** After each frame's dataflow region finishes, it copies each selected position's staged descriptor into the reference RAM. With 128-bit RAM words this takes at most 196 cycles. All updates are visible before the IP accepts the next frame.

Modules 1–4 form an HLS `DATAFLOW` region connected by `hls::stream` FIFOs, so the stages overlap. The commit runs after the region so that the reference RAM never has a reader and a writer in the same dataflow region. In the band build, the commit may later move inside the frame, one band at a time, because later bands never read those positions. This would shrink staging from 3,136 B to 896 B (two bands), and we will try it only if memory becomes the limit.

### Interfaces

| From → To | Interface | What it carries (per frame) |
|---|---|---|
| PS DDR → Descriptor builder | AXI4-Stream, 32-bit, via AXI DMA MM2S | One pixel per beat as `{0, B, G, R}` (8 bits each). 50,176 beats, `TLAST` on the last pixel. |
| PS ↔ IP | AXI-Lite (`s_axilite`) | Control: `K` (8-bit, 0–196), `reset_ref` (1-bit), `ap_start`. Status: `frame_id` (16-bit), `count` (8-bit), `cycles` (32-bit). |
| Descriptor builder → Patch scorer | `hls::stream<ap_uint<136>>` | `{pos 8b, descriptor 128b}`. 14 records per patch row, 196 in total. |
| Reference RAM → Patch scorer | On-chip RAM read, 196 × 128-bit | Saved descriptor for `pos`. 3,136 B in total. |
| Patch scorer → Staging RAM | On-chip RAM write, 196 × 128-bit | Current descriptor for `pos`. |
| Patch scorer → Budget selector | `hls::stream<ap_uint<20>>` | `{pos 8b, score 12b}`. 196 records. |
| Budget selector → Packet writer | `hls::stream<ap_uint<21>>` | `{last 1b, pos 8b, score 12b}`. `K_eff` records in raster order. |
| Packet writer → PS DDR | AXI4-Stream, 32-bit, via AXI DMA S2MM | Token word: `[7:0]` index, `[19:8]` score. Trailer: `[15:0]` frame ID, `[23:16]` count. All other bits zero, and only the trailer asserts `TLAST`. `K_eff + 1` words in total. |
| Packet writer → Reference commit | On-chip array, 196 × 1-bit | Selected-position mask. |
| Staging RAM → Reference commit → Reference RAM | On-chip RAM, 128-bit words | Staged descriptors of selected positions. |

**Performance targets (XC7Z020, 100 MHz):**

| Metric | Frame build | Band build |
|---|---|---|
| Pixel-loop initiation interval (II) | 1 | 1 |
| Sustained frame interval | ≤ 51,000 cycles (about 0.51 ms) | ≤ 51,000 cycles (about 0.51 ms) |
| First input to first selected word | about 53,000 cycles | about 8,000 cycles (band 0 needs 7,168 pixels plus ranking) |
| Last input to trailer | ≤ 3,500 cycles | ≤ 1,000 cycles |
| DSP use | Near 0: luma uses constant multiplies, and the rest is adds and compares | Same |

### Why this decomposition

- **The modules match the 2×2 experiment.** The descriptor builder and scorer change only with the descriptor type. The budget selector changes only with the selection scope. The fixed record formats between them let each team member swap one side without touching the other.
- **Each stage runs at its own rate.** Pixels arrive at one per cycle, descriptors at 14 per 3,584 cycles, and selections in bursts per band or per frame. FIFOs between the stages keep the pixel loop at II = 1 while the slower stages are busy.
- **Each module can be tested on its own.** Every module matches one function in the golden model: `descriptors`, `score`, `top_k`, and the commit line in `frame_step`. We can unit-test each against its function before integrating.

## Development and Validation Plan

### Build and compare

1. **Keep-all baseline.** Build only the stream reader and packet writer, with no descriptor RAM and no pruning. Check that it outputs indices 0–195 plus one trailer, including when the output stream stalls. Synthesize it separately to measure the cost of the I/O path alone.
2. **Software pilot.** Run the golden model on two inputs: a synthetic clip in which an edge moves inside a 4×4 block at constant mean, and two or three real manipulation clips. At the same `K`, compare four choices: uniform random sampling (seeded, without replacement), the mean descriptor, the edge descriptor and the oracle. If the edge descriptor adds little on real clips, the final IP keeps the mean descriptor. If exposure changes cause many missed feature changes, try regional exposure compensation, in software only.
3. **Two parallel hardware tracks.**
   - Hao: compare the two descriptors, with frame scope fixed.
   - Yaoqi: compare frame and band scope, with the mean descriptor fixed.
4. **Combine.** Build all four selectors (2×2) with the same replay data and synthesis settings. Choose the final build from measured feature error, latency and area, then optimize its slowest stage.

### Checking selection quality against the oracle

1. **Freeze the setup.** Before held-out testing, fix the encoder checkpoint, the preprocessing and the layer whose patch features we compare.
2. **Encode densely.** Run the encoder offline on every original frame.
3. **Give each selector its own cache.** Before each frame's decision, compute the staleness at each position: `e[p]` = cosine distance between the frame's dense feature at `p` and the selector's cached feature at `p`.
4. **Compute the oracle.** From that same cache, the oracle refreshes the `K` positions with the largest `e[p]`. No other choice of `K` positions leaves less total error.
5. **Score the selector.** Its excess error is its residual error minus the oracle's residual error, where residual error is the sum of `e[p]` over unrefreshed positions.
6. **Advance the cache.** Copy the dense features of the positions the selector actually chose (not the oracle's) into its cache, then move to the next frame.

This measures how stale the visual features become. Task success and end-to-end speedup need separate, end-to-end tests.

### Replay data and plots

- **Data:** a fixed third-person camera view from a public manipulation dataset. Record the episode IDs and frame sampling, and hold out whole episodes for testing.
- **Sweep:** sweep `K` from 0 to 196.
- **Plots:** residual error per frame (unrefreshed error divided by 196) and the largest missed error, each against `K`.
- **Breakdown:** report static scenes, gripper motion, exposure changes and camera motion separately where the data allows.
- **Demo:** show the original video, the selected patches and a heatmap of missed feature changes side by side.

### Hardware evidence

- **Bit-exact checks:** compare packet count, indices, scores and the post-frame reference RAM against the golden model, in both HLS C simulation and RTL co-simulation.
- **Test cases:** reset between episodes, repeated frames, tied scores, slow drift, `K = 0`, `K = 196`, uneven band quotas and stalls on the output stream.
- **Reports:** LUTs, FFs, BRAMs, DSPs, timing and sustained frame interval on the same XC7Z020 at 100 MHz.
- **Latency:** first input to first selected word (when one exists), last input to trailer, and time until the IP is ready for the next frame.
- **Reproducibility:** we commit the replay script, test vectors, co-simulation logs and synthesis tables so that another team can reproduce the comparison.
