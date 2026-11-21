# Initial Plan: Visual Token Differential Pruning IP for VLA Policies

[Project repository](https://github.com/muuuuuuuuuer/vla-token-prune)

## Project Team

- Hao Zhong (hz4324)
- Yaoqi Xia (yx3954)

## IP Definition

**Given a budget of K token refreshes per frame, which patches should the host update?** A moving gripper edge can change position inside a small image block while its mean brightness stays constant. An exposure shift can change the mean of every background block at once. We will test whether a streaming FPGA selector can spend the same refresh budget more effectively by retaining a little spatial information, and how much selection quality it gives up to produce indices earlier.

The IP accepts a 224×224 RGB frame and emits indices on a 14×14 grid of 16×16-pixel patches. The first frame of each episode refreshes all 196 positions. On subsequent frames the host sets `K` from 0 through 196, and the IP emits exactly `K` distinct indices. This fixed-budget experiment gives every rule the same opportunity to refresh tokens, including on static frames. The host retains the cached token at every omitted position. Our grid matches the frozen ViT-B/16 used in evaluation. VLA manipulation supplies the video workload. Model execution and the feature cache reside on the host. [VLA-Cache](https://github.com/siyuhsu/vla-cache) provides an example of cross-frame reuse in a VLA system.

### What the selector computes

Divide each patch into sixteen 4×4 blocks. Compute integer luma `Y = (77R + 150G + 29B) >> 8`. For each patch, compare its current descriptor with the descriptor saved **the last time that position was selected**. Both candidate descriptors use 16 bytes per patch, or 3,136 reference bytes per frame.

| Descriptor | One byte per 4×4 block | Patch score |
|---|---|---|
| Mean luma | Store the 8-bit mean `m = sum(Y) >> 4`. | Sum the 16 absolute differences between current and saved block means. |
| Mean and edge direction | Store the upper four bits of `m` and two bits each for horizontal and vertical contrasts. For each axis subtract the mean of one half-block from the other (right minus left, bottom minus top, each half mean rounded down). Encode the signed contrast as −1 below −16, +1 above +16, or 0 otherwise. | For each block use `4·abs(Δmean4) + abs(Δhorizontal) + abs(Δvertical)`, then sum across the patch. Decode contrast fields to signed values before subtraction. |

The edge rule trades mean precision for sensitivity to a boundary moving inside a block. Its thresholds and weights above are starting values for the pilot. Rank patches by descending score and resolve ties by lower raster index. Only the selected patches update their references. This lets small changes accumulate at skipped positions. Luma, block sums and reference comparisons run as pixels arrive, making the selector a candidate for a fixed-point pipeline with bounded on-chip storage.

## IP Architecture

### Dataflow and host interfaces

| Block | Inputs, outputs and state |
|---|---|
| Pixel input and descriptor builder | The PS replays frames from DDR through AXI DMA. The 32-bit AXI4-Stream input carries one pixel per beat as `{0,B,G,R}`, with `TLAST` on pixel 50,176. Row accumulators form block statistics in scan order and feed (position, descriptor) records to the scorer through a FIFO. |
| Patch scorer | Reads the saved descriptor for the same position from on-chip RAM, stages the current descriptor for a possible refresh, and emits (patch index, 12-bit score) records. On initialization it reports score zero and stages all descriptors. |
| Budget selector | The **frame selector** chooses the highest K scores across all 196 positions. The **band selector** decides after each pair of patch rows (7 bands of 28 patches). Zero-based band `j` receives `floor((j+1)K/7) − floor(jK/7)` slots. Both use exact ranking within their scope and issue selected indices in raster order. The band build can release staging space after each band's selections are committed. |
| Packet writer and reference update | The PS sets `K` and `reset_ref` over AXI-Lite at frame start. Each selected token produces a 32-bit AXI4-Stream word with index in bits 7–0 and score in bits 19–8. A trailer carries frame ID in bits 15–0 and count in bits 23–16. Other bits are zero. Only the trailer asserts `TLAST`. Commit each selected descriptor once and make all updates visible before accepting the next frame. |

The decision scope determines when records can leave the IP. A frame-wide selector must see the last score before it can guarantee the winners. A band selector can emit from the first band while later pixels arrive, but a band with a fast-moving gripper cannot borrow slots allocated to a quiet band. Motion confined to one band will test that trade-off directly.

### Build and compare

1. **Bring up a keep-all baseline.** Build the stream reader and writer with no descriptor RAM or pruning logic. Verify indices 0–195 and one trailer, including under backpressure. Synthesize this build separately to establish the I/O cost.
2. **Run a small software pilot.** Use a synthetic sequence in which an edge moves within a 4×4 block at constant mean, plus two or three real manipulation clips. Compare uniform sampling (seeded, without replacement), mean luma, the edge descriptor and the offline reference below at identical K. If edge information adds little on real clips, keep mean luma for the final IP. Try regional exposure compensation in software only if exposure changes account for substantial missed feature changes.
3. **Develop the two hardware choices independently.** One member compares descriptors with frame-wide selection fixed. The other compares frame and band selection with the mean descriptor fixed. Evaluate the resulting 2×2 combinations under the same replay and synthesis settings. Use the measured error, latency and area to choose the final build, then optimize its limiting stage.

### How we will validate the pruning decisions

**Offline reference.** Fix the encoder checkpoint, preprocessing and patch-feature layer before held-out evaluation. Encode each original frame densely offline. For each policy, let `e[p]` be the cosine distance between the current dense feature at position `p` and that policy's cached feature there. Refreshing a position copies its current dense feature into the evaluation cache. From the **same pre-decision cache**, selecting the K largest errors minimizes the sum of errors left at the omitted positions for that frame. Measure each selector's excess residual error against this reference, then advance its cache using its own actual selections. This reference measures visual feature staleness. Policy success and inference speedup require separate end-to-end measurements.

**Replay and demonstration.** Use a fixed third-person view from a public manipulation dataset, record episode IDs and frame sampling, and reserve whole episodes for testing. Sweep K. Plot the residual error per frame (`sum of unrefreshed errors / 196`) and the largest missed error against refresh budget. Report static scenes, gripper motion, exposure changes and camera motion separately where available. Show the original video, the selected patches and a heatmap of missed feature changes side by side.

**Hardware evidence.** Compare packet counts, indices, scores and post-frame reference state against an integer golden model in HLS simulation and RTL co-simulation. Exercise reset between episodes, repeated frames, ties, slow drift, `K = 0`, `K = 196`, uneven band quotas and receiver stalls. On the same XC7Z020 and 100 MHz target, report LUTs, FFs, BRAMs, DSPs, timing and sustained frame interval. Measure first input to first selected word (when one exists), last input to trailer, and readiness for the next frame. The replay script, test vectors, co-simulation logs and synthesis tables will let another team reproduce the comparison.
