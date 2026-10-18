# KVCacheManagerV2 sparse attention — handover

Branch `kvcm2-sparse-attention-api`, pushed to `fork`. Everything below is on top
of your `fd41f2f13c` merge.

Two parts, in the order they were written: a **breaking config change** plus the
fixes it enabled, and then a **Batch-centred rework of the sparse step** — the
fetch and the write-back both address the cold side by converted page index, and
the write-back became an explicit batch-wide call. Read *The config change*
first if you are picking this up cold; read *The sparse step is the Batch's* if
you are looking at the current API.

---

## The config change

### Before

```python
BufferConfig(
    role="key",
    size=row_bytes,                        # bytes of ONE selection unit
    tokens_per_block_override=4,           # also stretched the page index space
    is_sparse=True,
    num_selected_blocks=topk,
    max_cached_units=2 * topk,
    sparse_unit_layout=SelectionUnitLayout(1, row_bytes, 0, row_bytes),
)
```

### After

```python
AttentionLayerConfig(
    layer_id=...,
    buffers=[
        BufferConfig(role="key", size=page_bytes, is_sparse=True),   # whole page
        BufferConfig(role="index", size=...),                        # dense, same layer
    ],
    sparse_cache=SparseCacheConfig(
        num_selected_blocks=topk,
        max_cached_units=2 * topk,
        unit_tokens=4,
    ),
)
```

### What moved, and why

| was | is | why |
| --- | --- | --- |
| `BufferConfig.num_selected_blocks` | `AttentionLayerConfig.sparse_cache.num_selected_blocks` | every sparse buffer of a layer shares one resolve, one resident set, one recency order |
| `BufferConfig.max_cached_units` | `…sparse_cache.max_cached_units` | same |
| `BufferConfig.tokens_per_block_override` (for sparse) | `…sparse_cache.unit_tokens` | the override also stretched the buffer's **public page index space**; the new field names the sparse cache's granularity only |
| `BufferConfig.sparse_unit_layout: SelectionUnitLayout` | `BufferConfig.sparse_unit_chunks: int` | the other three fields are derivable |
| `BufferConfig.size` = bytes per **unit** | bytes per **page** | same meaning as every dense buffer |
| `KVCacheManager.get_sparse_cache_expansion` | `…get_sparse_cache_units_per_page` | `expansion` means page-index-space stretch, which is a different quantity |
| `AttnLifeCycle.is_sparse` | *(removed)* | see below |

These were previously declared per buffer and then collapsed with `std::max`
across the whole life cycle — two buffers, or two layers, could disagree and the
larger silently won. They are now declared once, where they belong, and
disagreement between layers of one life cycle is **rejected**.

### `sparse_unit_chunks`

Replaces the four-field `SelectionUnitLayout`. It is the page's dimension order
expressed as a count:

- **1** for a sequence-major page (`[token][head][dim]`), or for any single-head
  page (a compressed or pooled latent) — the unit is one contiguous range.
- **numHeads** for a head-major page (`[head][token][dim]`) — the unit is that
  many strided pieces.

Everything else is derived, given the page is densely packed:

```
unitsPerPage = tokens_per_block / sparse_cache.unit_tokens
chunkStride  = size / sparse_unit_chunks
chunkBytes   = size / (sparse_unit_chunks * unitsPerPage)
unitStride   = chunkBytes
```

DeepSeek V4 asserts `num_kv_heads == 1`, so **1 is correct** and is the default.
A survey of every sparse backend in the repo (`deepseek_v4`, `dsa`, `glm_kpool`,
`qsa`, `minimax_m3`, `rocket`, `skip_softmax`, `csa2`) found all of them either
single-head or sequence-major on the buffer selection reads. The field exists for
QSA, whose selection is shared across heads while the buffer attention gathers
from is head-major GQA.

Padded pages (alignment gaps between heads or between units) are no longer
expressible. Nothing in the repo has one.

### `sparse_cache` presence is the switch

`sparse_cache=None` on a layer with sparse buffers means **sparse storage and
locking, no GPU fetch path** — the documented case that `max_cached_units=0` used
to express. Setting `sparse_cache` on a layer with no sparse buffer is rejected.

---

## One layer, both dense and sparse buffers

Previously `AttnLifeCycle` carried `isSparse`, derived from
`buffers.front().isSparse` — so a layer with mixed buffers silently took its
*first* buffer's sparsity, and a layer had to be wholly one or the other.

Sparsity is now a property of a **pool**:

- buffer coalescing keys on `(size, isSparse)`, so a pool is uniformly sparse or
  uniformly dense — **sparse buffers never coalesce with dense ones**, even at
  equal size, because a pool is allocated from one medium;
- the pool-group key carries a per-pool sparsity vector, so two life cycles share
  a group only when their pools match position-wise on size *and* sparsity;
- the GPU tier allocates sparse pools from **host** memory (device-mapped) and
  dense ones from device memory, in the same group, and reports `GPU_MEM` for the
  group. `isSparsePool(level, pg, pool)` names the host-backed ones.

Cardinality is unchanged: a life cycle still belongs to exactly one pool group,
and a pool group may hold several life cycles.

**This is why `90e9a9c39a` collapses your `if self._enable_kv_cache_offload:`
split.** `cc71759303` (#19502) separated `COMPRESS` and `INDEXER_COMPRESS` into
two KVCM2 layers purely because `is_sparse` had to be uniform per layer. Without
offload they were already one layer. They are one attention computation, so they
are one layer again, with only `COMPRESS` marked sparse — and the dense indexer
pool is what sizes the group's slots.

Two checks inverted with it:

- a dense buffer beside a sparse one in a layer is the supported shape, not an
  error (`get_sparse_offload_descriptors` now takes only the sparse buffers);
- sharing a layer group with a sparse layer is expected.

### Pool-group sizing

A slot is one row spanning **every** pool of its group, so once a group holds a
device-backed pool, the sparse pools inherit its row count for free. The GPU quota
therefore charges only the non-sparse pools. **A pool group must hold at least one
non-sparse pool** — a wholly sparse group has no chargeable bytes per slot and so
no derivable slot count. The natural configuration (indexer + data in one layer)
satisfies this.

This removed the previous `kMaxOffQuotaHeadroom` heuristic entirely.

---

## New API: `prepare_sparse_write_indices`

```python
manager.prepare_sparse_write_indices(batch, buffers, out, stream)
# out: int32[batch.max_batch_size, len(buffers)]
```

Gives each request's **in-flight** unit a cell and reports where to write it. The
resolve already made that unit resident and pinned it; it simply never reported
where. Separate from `fetch_sparse_pages` because the orders differ — a fetch
needs the indexer's selection, a write needs only the watermarks, so it can run
first, which is the order the model requires.

- Idempotent while the watermarks hold; the fetch's own resolve reuses the entry.
- Nothing is fetched: an in-flight unit has no host copy yet.
- `BAD_PAGE_INDEX` is an **ordinary outcome**, not an error. With 4-token units a
  unit completes one step in four, which is also the only step a ratio-4
  compressor has produced anything to write.
- `out` is `[batch, buffers]` — one entry per request, since `resize` bounds
  `capacity - history_length` to a single token.

Note the shape differs from your declaration (`[batch, buffers, 1]`); the trailing
extent was indexed away at its only consumer, which then rebuilt `[batch, 1]`
explicitly for the scatter's block table.

---

## Indices and base addresses

`fetch_sparse_pages` and `prepare_sparse_write_indices` return indices relative to
**`get_sparse_cache_base_address(layer, role)`** — *not* `get_mem_pool_base_address`,
and not to a cache level.

- `get_mem_pool_base_address` names the buffer's **pages**, which for a sparse
  life cycle are host memory. It is what the transceiver and the KV transfer path
  address, and keeps one meaning for every buffer.
- `get_sparse_cache_base_address` names the **arena region** of that buffer's pool.
  It is role-dependent: the arena mirrors the hot pool group, so buffers of
  different sizes (data and its block scales) sit in regions of their own.

Sparse buffers may now span more than one pool, so the quantised shape
(data + block scales) works.

---

## Commits

| commit | what |
| --- | --- |
| `6d10dee2ab` | sparse buffers across more than one pool (quantised shape) |
| `6d69f8749d` | `prepare_sparse_write_indices` |
| `5f8b37c900` | name the base a sparse fetch's indices are relative to |
| `269e74d14d` | sparsity becomes a pool attribute, not a life cycle's |
| `623f684400` | keep host-backed pool bytes off the GPU quota on every path |
| `1439b4f602` | move the fetch path from `BufferConfig` to `AttentionLayerConfig` |
| `90e9a9c39a` | DeepSeek V4: compressed KV and its indexer back in one layer |
| `87554e22b5` | describe the converted index tables a Batch will build |
| `1790111fd1` | convert page indices inside the Batch |
| `93720c1002` | substitute scratch slots when converting indices |
| `4fbb2b4643` | a Batch owns the stream its members run on |
| `c5dfa2a4bd` | `Batch::resize` checks every row before touching any |
| `db36791a8e` | address the sparse cold side by converted page index |
| `d3f1c963d6` | write back sparse units through an explicit Batch call |

`623f684400` fixes two bugs a review found in `269e74d14d`: the runtime rebalance
path re-charged sparse pools against the GPU quota (silently under-sizing the hot
level, and able to reject a quota the device could hold), and a hot-to-hot defrag
migration dispatched a device-to-device copy kernel over host pointers.

---

## Verification

Run **in the `tekit` container on `dev-b200`**, not locally:

```
myssh dev-b200 "docker exec tekit bash -lc '<cmd>'"       # myssh is ~/bin/myssh
```

- KVCM2 suite: **251 pass / 12 skipped**
- C++ unit tests: 80 pass across 10 targets
- DSv4 + integration, as one run of
  `test_deepseek_v4_cache_manager.py`, `test_deepseek_v4_offload.py` and
  `test_kvcm2_integration.py`: **49 fail / 633 pass**, split 13 and 36. That is
  the pre-existing baseline, not a regression — the classes are
  `_FakeManagerConfig has no attribute 'constraints'` (20), the offload
  host-memory `ValueError` (12), `_enable_kv_cache_offload` (2), and three
  singletons. Check a change against the split and the error classes, not just
  the total.

Running tests: use `PYTHONPATH=$REPO_ROOT` with `python -m pytest`. Mixing the
fast-mode `PYTHONPATH=$REPO_ROOT/tensorrt_llm/runtime/` with a repo-root pytest
run makes the module take its fast-mode import branch and fail to find
`bindings.internal`.

Build notes, learned the hard way:

- use `-j12`, not `-j$(nproc)` — the sm100 CUTLASS instantiations OOM at 32-way
  and fail as `Error 9` with no `error:` line;
- `/home/yaoy/tekit/cpp/build` is shared with other containers running a
  **different PyTorch** (nv26.08 vs nv26.09). Objects compiled elsewhere link
  here and fail with undefined `c10::cuda::*` symbols. Only torch-dependent
  objects are affected (123 of 1881 when I hit it);
- the `msa`/`xgrammar`/`deepgemm`/`deep_ep_download` FetchContent patch step can
  fail after a reconfigure; remove all four `_deps/*-src` and `*-subbuild` at
  once, since fixing one exposes the next.

---

## Handed over: 10 failing tests

Both in `tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py`.
Neither is believed to be a product bug.

> That file's failure count has since drifted to 13 for reasons unrelated to
> these two — see *Verification* for the current baseline and how to read it.
> The two described below are still among them.

### 1. `test_sparse_warmup_cleanup_covers_storage_and_shared_arenas` (8 params)

`assert wrapper.numel() == buffer.numel()` fails, e.g. `884736 != 27648` — a
factor of `tokens_per_block // 4`.

The fixture (~line 500) mocks `get_page_index_upper_bound` with
`host_rows = 3 * (tokens_per_block // 4)`, the compressed-**row** count. That was
right when a sparse buffer stretched its index space; now `expansion == 1` and the
bound counts **pages**, with `get_buffers` shaping
`(pages, compressed_block_size, width)`.

**Trap:** changing the mock to `return_value=3` fixes the storage view but
triggers a **CUDA device-side assert** that poisons the context and cascades into
~17 unrelated tests in the same process. Something in the arena path
(`_sparse_arena_buffers` / `_initialize_sparse_arena_buffers`, 11 rows in this
fixture) also consumes that mock and under-sizes. Trace that consumer before
changing it, and run a single param with `-x` — the cascade hides the real
failure.

Likely fix: give the two consumers separate values rather than one `host_rows`.

### 2. `test_sparse_offload_rejects_incompatible_layout[width]` (2 params)

`DID NOT RAISE`. The test mutates the selection width to disagree with
`index_topk` and expects rejection:

```python
cache = config.sparse_cache
cache.num_selected_blocks += 1
config.sparse_cache = cache
```

Two candidates, not distinguished:

1. the write doesn't persist — `sparse_cache` is a nanobind `def_rw` over
   `std::optional<SparseCacheConfig>`, so reading returns a copy; confirm with
   `assert manager.impl.layers[41].sparse_cache.num_selected_blocks == 9`;
2. the check isn't on that path for this layer — `layer_configs` is built from
   `impl.init_config.layers` and the loop iterates `local_layers`; if layer 41
   isn't among them the check never runs.

The `unconfigured` case (`sparse_cache = None`) *does* raise, which argues the
plumbing works and points at (2).

Also note: the `"mixed"` parameter was removed from that test's parametrisation
on purpose — a dense buffer beside a sparse one is now supported, and
`test_sparse_offload_accepts_a_dense_buffer_in_a_sparse_layer` asserts it.

---

## Landed since this document was written

- **`KVCacheManager.is_sparse(layer_id, data_role)`** is implemented and bound
  (`3362c53bf6`). It previously existed only in the stub, so the first call
  against a real manager would have raised `AttributeError`.
- **`KVCacheManager.get_sparse_host_bytes()`** reports the host bytes the hot
  level holds for sparse pools, across every pool group — a figure drawn from no
  quota and visible in no other total. `kv_cache/sparse_offload_memory.py`
  currently mirrors the native arena layout to derive this; reading it instead
  would remove that duplication.

## Still open, beyond those tests

- **Compression** is reachable but not turned on. The codec is the hot↔cold
  transform; sparse history is hot, so it crosses that boundary like any other
  page and needs nothing sparse-specific.
- **Per-layer selection unit size.** Layers of one life cycle must currently agree
  on `sparse_cache`. Not fundamental — size a cell from the largest unit (or a
  whole page) and let a smaller-unit layer subdivide it, which is the mechanism a
  cell already uses; power-of-two unit sizes make every division exact. Sketched
  in `SPARSE_ATTENTION_DESIGN.md`.
- **Per-KV-head selection** is excluded by design: one resident set per
  `(request, layer)`. Rocket (`[num_kv_heads, total_tokens]`) and MiniMax-M3 under
  MSA (`[tokens, num_kv_heads, topk]`) both select per head and cannot be served.
- **Multi-beam** is refused at `batch.cpp`.
- **VA sizing** includes sparse pools in its `maxSlotSize` ratio, costing growth
  headroom. `@TODO` at `storage/core.cpp`.
- **Dirty-row publication** is planned, not built: `publish()` restages every row.

### The two DeepSeek-V4 workarounds for the quota bug

I suggested earlier that both might now be removable. Having checked, **only one
is, and the other should stay** — the reason I gave for it was wrong.

**`allocated_bytes` — already fine, nothing to do.**
`kv_cache_manager_v2.py:5294` reads `self.impl.get_quota(GPU_LEVEL)` with no
subtraction. Since `623f684400` that call excludes host-backed pools by
construction, so the line is correct as it stands.

**`enable_kv_pool_rebalance` rejection — keep it.** It is *not* a workaround for
the quota accounting. Its own stated reason (`kv_cache_manager_v2.py:5310`) is
structural:

> host-backed history pools take their pool group's slot count, so a rebalance
> would resize pinned host memory outside the host budget

That follows from the design — a slot spans every pool of its group, so a sparse
pool's row count is not independently controllable — and fixing the quota paths
does nothing to change it. It is the same constraint as the open question in #3,
and it compounds with the `mremap(MREMAP_MAYMOVE)` hazard: a host pool resize can
move the base under captured CUDA graphs.

So treat these three as one thread, not three items: **a sparse pool cannot
currently be sized, or resized, independently of its pool group.** Resolving that
is what would let the rebalance guard come out.

---

## The sparse step is the Batch's

The per-step work moved out of `KvCache` and into `Batch`. A lease is the
request's reservation again and carries no per-step state; everything a step
scans, converts or writes back belongs to the Batch, because every one of those
is a single launch spanning the rows.

**The cold side is addressed by converted page index.** A copy reaches a host
page as `poolBase + converted * getPageStride(layer, role)`. The converted value
already carries the buffer's offset within the coalesced page, so the pool's slot
size plays no part — what a converted index counts is whole buffers.
`pageBufferOffset` is gone from `SparseCacheView`, the arena metadata and the
cache shape.

Both producers of page indices read **one** table at different extents: the fetch
from row zero across the batch, the write-back positioned at its own row. The
private slice the write-back used to stage, and the synchronous upload that
filled it, are gone.

Two things are refused when a Batch is built, because a run that cannot be
addressed would otherwise produce wrong page addresses at step time rather than
an error: a sparse buffer no `IndexTableRequest` names, and a `(layer, pool)` run
split across two requests, which a single launch could not address from one base.
`KVCacheManager.sparse_index_table_requests()` returns that shape; a Batch still
has to be given it.

**The write-back is an explicit call.** `Batch.flush_sparse_units(new_history)`,
one entry per row, **before** advancing history. `resize()` then checks
`new_history <= flushed` rather than flushing anything. The check fires only when
the move crosses a unit that was in flight, clamped to the capacity the resize
started from — so a resumed request, whose units reached their host pages
directly, crosses nothing and needs no flush.

It reads the index table **as of the last publish**, which is current enough for
the same reason: the blocks in range were allocated by an earlier step. A member
resized directly with no publish in between would write back through stale
indices.

**A step's order, which the caller owns:**

1. `flush_sparse_units(new_history)`
2. `resize(...)`, or `Batch.resize`
3. `publish()`
4. the fetch, inside the captured graph

**A Batch owns its stream**, taken at construction. Joining binds the member to
it and carries the cross-stream sync a direct move would; `set_cuda_stream` is
refused while batched, and `resume()` refuses a stream that is not the Batch's.
That is what makes the write-back's ordering against a member's own events
structural rather than checked.

**`Batch.resize` validates every row before touching any.** A history may not
fall, nor outrun the capacity its row already had — capacity granted by a call
makes room for tokens written *after* it. A row whose growth fails still takes
its history advance, which needs no slots and is what frees the cells the growth
wanted; its result stays `0`.

### What this means for a caller

- **Seed a resumed request before it joins a Batch.** Setting capacity and
  history together from nothing is the one shape `Batch.resize` excludes. It
  costs nothing to do on the way in: nothing was in flight, so there is no
  write-back and no published table is needed.
- **Flush while the members are still seated.** DSv4 was releasing its Batch and
  *then* advancing history (`update_forward_resources`). The implicit flush
  tolerated that; this does not. Its own tests did not catch it, because their
  moves never cross a unit boundary.
- **Per-step membership churn is no longer forced.** The Batch borrows the
  runtime's page-index rows rather than keeping its own, so the single-lessee
  conflict that made DSv4 detach and re-attach every member each step is gone.
  `kernels.py:114` still assumes live rows are a contiguous prefix:

  ```python
  live = (token_id < active) & (req >= 0) & (req < active)
  ```

  That holds only while every member is re-added into a freshly emptied Batch.
  Once membership persists for a request's decode lifetime, finished requests
  leave holes and this mask marks a hole live and a high-numbered live row dead.
  It needs a per-row live signal; KVCM2 exposes one as
  `residentOrdinal == nullptr`, which the resolve and index kernels already use.

### Open, on this thread

- **The write-back trigger is per step, not per `resize`.** That is the point,
  but it means a caller resizing a member twice in a step must have flushed for
  the furthest history, not the nearer one. The `<=` check permits it; nothing
  reminds them.
- **`Batch::commit` / `stop_committing` were considered and not built.** They
  existed in the plan only to carry a watermark sync that the explicit flush
  removed; `commit`'s own history advance goes through `resize`, where the check
  already fires.

---

`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/SPARSE_ATTENTION_DESIGN.md`
is current and is the design of record.
