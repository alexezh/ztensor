# FocusCross

`focuscross.h`: [LowResCross](lowrescross.md) with a workspace that is
refined at every level, focused by what the queries found relevant.

The rows, the Layout and the queries' layers are LowResCross's. In
LowResCross the memory is fixed through the stack. Here the memory (the
workspace) changes: every level first lets the workspace rows attend to each
other, then runs the queries over the workspace as it now stands.

```
workspace  w_j += r_j · mask · O_w(attn_j(Q_w ŵ, K_w ŵ + log(r + eps), V_w ŵ))
self       h   += mask · O_s(attn(Q_s x̂, K_s x̂, V_s x̂))        among the queries
cross      h   += mask · O_c(attn(Q_c x̂', K_c ŵ', V_c ŵ'))     into the workspace
           r_j  = 1 - exp(-s_j)
ffn        h   += mask · W_2 relu(W_1 x̂'')
```

## Relevance

- **What it is.** `s_j` is the attention workspace row `j` got in a level's
  cross attention, summed over the pass's query rows and averaged over heads.
  `r_j` is in [0, 1).
- **What it does.** Each level's relevance sets the next level's workspace
  attention:
  - Relevance sets which rows take part as keys, through the bias
    `log(r + eps)` (eps 0.01): a row nobody looked at gets about 1/100th the
    weight.
  - It also sets which rows the workspace attention changes, through the
    gate `r_j`. A row nobody looked at stays as it was.
- **It's continuous,** so the loss trains it: the gradient reaches the cross
  attention that produced it.
- **It's recomputed at every level,** so a relation found at one level can
  make another row matter at the next.
- **At the first level every row counts** (r = 1).

A pass with few query rows gives every row less relevance, because `s_j`
sums over the pass's query rows.

## The workspace as working memory

The workspace rows evolve in their state slots, only where their write mask
allows (`Layout::put_write_mask`). They're an internal working memory, kept
apart from the caller's rows. `memory_rows()` returns the rows after the last
level, which is what a `LowResPointer` scores against.

```cpp
zf::FocusCross<float, MyLayout, Q, M, /*Heads=*/4, /*HeadDim=*/16, /*Ffn=*/128> focus(/*levels=*/4);
focus.randomize(rng);
focus.forward_batch(passes);                 // as LowResCross
const auto h = focus.readout_of(b);
const float* r = focus.relevance(b);         // the last level's, per workspace row
const float* r1 = focus.relevance(b, 1);     // level 1's
focus.backward_batch(d);
focus.apply_sgd(lr);
```

## Where it runs

As LowResCross:
- `set_device()` picks the CPU (`focuscross_cpu.h`) or the GPU
  (`apple/focuscross_apple.h`). `kAuto` sends batches of 64 passes or more to
  the GPU when it's on.
- `step_batch(batch, heads)` runs a whole step in one GPU run: the stack, the
  model's heads and their loss, and every gradient. The heads are the same
  kind LowResCross takes ([Apple GPU](apple_gpu.md#a-whole-step-lowrescrossstep_batch)).
  `hg.rows` is the workspace after the last level.
- `forward()`, a batch of one, always runs on the CPU.

## Compared with LowResCross

- **Weights:** LowResCross's, plus each level's workspace attention: a norm,
  Q, K, V and an output map into the state. That's about 35% more weights at
  the same widths.
- **Cost:** about 1.5× LowResCross's on the CPU at a policy's size (rows 426
  wide, 11 queries, 12 workspace rows, 4 levels).
- **No memories' cache.** The workspace depends on the pass's queries, so
  there's no `set_memories` / `forward_shared`.
- **Smaller steps.** A pointer's gradient now moves the rows it scores,
  which feeds back into itself. In the tests, per-example SGD at 0.05
  diverges where LowResCross trains, so use a smaller rate or clip the
  gradient.

The tests are in `tests/focuscross_test.cpp`:
- every parameter's gradient against finite differences, including through
  the relevance;
- the GPU's forward, backward and fused step against the CPU;
- the workspace writes only its state;
- passes are independent;
- a pointer learns through the workspace.
