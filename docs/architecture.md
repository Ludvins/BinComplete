# Resource and artifact boundaries

```text
inputs -> prepare_dataset -> Dataset -> fit(method) -> ModelArtifact
ModelArtifact + Dataset -> propose -> ProposalArtifact
ProposalArtifact -> export chosen IDs -> external tools
external approved IDs + proposals -> reconcile -> SelectionArtifact
SelectionArtifact -> export -> changed FASTAs + unchanged-input manifest
```

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
its expected array and receipt files. Incomplete entries and unrelated files
are preserved; if they prevent admission, the cache reports a budget error.

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
