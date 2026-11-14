# Resource and artifact boundaries

<ol class="diagram-flow" role="list" aria-label="Artifact lifecycle">
  <li>
    <strong>Prepare inputs</strong>
    <code>inputs → prepare_dataset → Dataset</code>
    <svg class="diagram-flow__arrow" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M4 12h15m-5-5 5 5-5 5" /></svg>
  </li>
  <li>
    <strong>Fit a model</strong>
    <code>Dataset → fit(method) → ModelArtifact</code>
    <svg class="diagram-flow__arrow" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M4 12h15m-5-5 5 5-5 5" /></svg>
  </li>
  <li>
    <strong>Propose alternatives</strong>
    <code>ModelArtifact + Dataset → propose → ProposalArtifact</code>
    <svg class="diagram-flow__arrow" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M4 12h15m-5-5 5 5-5 5" /></svg>
  </li>
  <li>
    <strong>Assess quality externally</strong>
    <code>ProposalArtifact → export chosen IDs → external tools</code>
    <svg class="diagram-flow__arrow" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M4 12h15m-5-5 5 5-5 5" /></svg>
  </li>
  <li>
    <strong>Reconcile approvals</strong>
    <code>External approved IDs + proposals → reconcile → SelectionArtifact</code>
    <svg class="diagram-flow__arrow" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M4 12h15m-5-5 5 5-5 5" /></svg>
  </li>
  <li>
    <strong>Export compatible bins</strong>
    <code>SelectionArtifact → export → changed FASTAs + unchanged-input manifest</code>
  </li>
</ol>

Resource defaults depend on the entry point:

| Entry point | Allocation budget | Numerical threads | Rebuildable cache | Scoring block |
| --- | --- | --- | --- | --- |
| Low-level API and `prepare`/`fit`/`propose`/`run` CLI | 4 GiB | 4 | 10 GiB | 4,096 rows |
| Standard compact or materialized experiment | 6 GiB | 2 | 10 GiB | 4,096 rows |

Shared configuration or explicit resource settings may override these values.
Declared buffers and conservative fit estimates are
bounded; these are not an OS RSS reservation or page-cache cap. Oversized fits
fail explicitly without silent sampling. Individual sequence buffers are bounded.

Read-only memory maps and row indirection avoid array copies. Retrieval,
hard-negative mining and scoring use candidate blocks. Only top donors stay in
RAM; full saved score vectors remain proposal evidence. Neural queries are computed
once per recipient. A byte-limited feature LRU avoids per-proposal disk cache files.
Methods run serially; contexts release maps, leases and thread settings.

Only rebuildable arrays live in `arrays/<content identity>` in the derived cache.
Entry leases prevent active eviction; a quota lock serializes admission. Concurrent
requests for a leased entry fail rather than duplicating work. Configure `--cache`
or `BINCOMPLETE_CACHE`; default is the platform user cache under `bincomplete`.
Keep the cache separate from inputs and artifacts. Models and scientific evidence
are never evicted.
Eviction requires a completed entry with a matching content identity and only
recognized files. Automatic eviction leaves interrupted entries available for
resumption. Run `bincomplete cache clean` to preview removal of unused completed
arrays in the configured cache. Add `--apply` to remove the selected entries.
Pass `--cache PATH` to select another cache. Add `--include-incomplete` to
discard inactive interrupted entries with a matching ownership receipt.
Active entries and unrecognized files are skipped. Bundles written by
`bincomplete encode` are stored outside the cache.
Resuming encoding removes stale temporary progress files. Cleanup can also
remove completed encoder entries that still contain those files.
For a compatible encoding checkpoint, BinComplete rechecks stored rows marked
done and re-encodes any that fail validation. Fresh batches are checked at their
storage precision before they are marked done. Completed encoder caches and
embedding bundles must pass the same finite-value and L2 checks as preflight.
An invalid completed bundle is preserved. Use a new output directory to encode
again.
If another operation holds the cache lock, cleanup reports that the cache is
busy. If deletion fails after invalidating an entry, retry cleanup with
`--include-incomplete` after resolving the file error.

```shell
bincomplete cache clean --cache project-cache
bincomplete cache clean --cache project-cache --apply
```

Prepared metadata/index files reference original assembly and embedding inputs.
Fit matrices are scratch removed on success; episodes, group decisions, history,
configuration and weights remain model evidence. Completed outputs pin hashes
and reuse only matching identities. Failed partial outputs remain for inspection.

Memberships stay records until export; FASTAs are content-addressed. Default
final export references unchanged bins. Complete directories prefer hard links
and record any copy fallback. Assemblies are never copied into output folders.

The structural solver uses one thread, maximizes recipient count, then prefers
lexical proposal IDs. A whole-operation budget includes tie breaking. Receipts
distinguish feasible witnesses, proven maxima and canonical tie breaks. No external
quality metrics are parsed or interpreted here.
