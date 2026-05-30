---
id: doc-cdna4-whitepaper
title: AMD CDNA 4 Architecture White Paper
url: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
source_category: official-doc
architectures:
- gfx950
tags:
- xcd
- fp4
- fp6
- mxfp
- infinity-cache
- hbm3
- cdna
retrieved_at: '2026-05-15'
---

# AMD CDNA 4 Architecture White Paper

Architecture white paper for CDNA4 / MI350-MI355X. Each XCD has 36
physical CUs (32 active) on TSMC N3P, 4 MB L2; up to 8 XCDs → 256 CUs. MI355X
ships 288 GB HBM3E at up to 8 TB/s. Table 1 peak matrix throughput: FP16/BF16
2.5 PF, OCP-FP8 5.0 PF, INT8 5.0 POPS, MXFP6/MXFP4 10 PF (dense). FP8 is the
OCP encoding (labeled "OCP-FP8"), distinct from CDNA3's FNUZ FP8.

## Reference

- Upstream: <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf>
