# Diff summary

- **files changed:** 17 (diff was byte-capped; summary is partial)
- **lines:** +3840 / -0
- **kernel-ish files:** 2

## Files (by churn)

- `docs/conceptual/ck_tile/coordinate_systems.rst`  (+612/-0)
- `docs/conceptual/ck_tile/convolution_example.rst`  (+567/-0)
- `docs/conceptual/ck_tile/coordinate_movement.rst`  (+532/-0)
- `docs/conceptual/ck_tile/buffer_views.rst`  (+462/-0)
- `docs/conceptual/ck_tile/adaptors.rst`  (+391/-0)
- `docs/conceptual/ck_tile/cache_flushing_benchmarking.rst`  (+390/-0)
- `docs/conceptual/ck_tile/descriptors.rst`  (+383/-0)
- `docs/conceptual/ck_tile/convert_mermaid_to_svg.py`  (+224/-0)
- `docs/conceptual/ck_tile/MERMAID_DIAGRAMS.md`  (+156/-0)
- `docs/conceptual/ck_tile/convert_raw_html_to_commented.py`  (+84/-0)
- `docs/conceptual/ck_tile/CK-tile-index.rst`  (+33/-0)
- `docs/conceptual/ck_tile/diagrams/adaptors_1.svg`  (+1/-0)
- `docs/conceptual/ck_tile/diagrams/adaptors_2.svg`  (+1/-0)
- `docs/conceptual/ck_tile/diagrams/buffer_views_1.svg`  (+1/-0)
- `docs/conceptual/ck_tile/diagrams/buffer_views_2.svg`  (+1/-0)

## Key added lines (kernel files)

**`docs/conceptual/ck_tile/convert_mermaid_to_svg.py`**
```
Script to convert all mermaid diagrams in CK Tile docs to SVGs.
This script:
1. Finds all mermaid blocks in RST files
2. Converts them to SVG using mmdc
```

**`docs/conceptual/ck_tile/convert_raw_html_to_commented.py`**
```
"""Convert raw HTML mermaid blocks to commented format for SVG conversion."""
import os
import re
def convert_raw_html_to_commented(content):
```
