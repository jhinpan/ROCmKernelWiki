---
id: doc-cdna3-whitepaper
title: AMD CDNA 3 Architecture White Paper
url: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
source_category: official-doc
architectures:
- gfx942
tags:
- xcd
- l2-cache
- infinity-cache
- hbm3
- cu
- cdna
retrieved_at: '2026-05-15'
---

# AMD CDNA 3 Architecture White Paper

Architecture white paper for CDNA3 / MI300. Describes the chiplet
design: each XCD (Accelerator Complex Die) has 40 physical CUs (38 active),
sharing a 4 MB L2; up to 8 XCDs → 304 CUs on MI300X. A 256 MB Infinity Cache
(memory-side LLC, 16-way) sits on the IO dies; MI300X ships 192 GB HBM3 at
5.3 TB/s. L2 coherence is per-XCD, making the XCD an effective NUMA domain.

## Reference

- Upstream: <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf>
