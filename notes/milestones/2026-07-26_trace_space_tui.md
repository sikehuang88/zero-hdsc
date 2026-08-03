# Trace Space Runtime and TUI

Date: 2026-07-26

## Delivered

- Schema v10: append-only `traces`, semantic `trace_links`, activation audit,
  and optional sqlite-vec `trace_vec`.
- `TraceSpaceService.write_turn`: every completed interaction becomes one
  immutable episode trace.
- `TraceSpaceService.activate`: semantic recall, freshness/importance product,
  top-K selection, and one bounded radiation hop.
- Historical user/agent turns are indexed idempotently on TUI startup.
- The DeepSeek prompt receives activated trace content with source, score, and
  activation kind before the current reply is generated.
- The `SPACE` inspector renders embedding PCA geometry, links, activation heat,
  a selectable trace list, and per-node evidence details.

## Visual Legend

```text
@ main activation
+ radiation activation
o newest trace
. inactive trace
: semantic link
```

## Engineering Boundary

Trace space stores the append-only experiential substrate. The existing
`memories` table stores extracted and deduplicated long-term facts. Keeping the
two separate prevents memory consolidation from destroying episode geometry.
