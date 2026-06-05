"""Coordinate transform sanity test (tools).

这是排障脚本：用于在你怀疑 object_pos/skeleton_pos 坐标口径不一致时做快速对照。
备注：该仓库的主逻辑已在 `joint_attention_evaluator.py` 中处理多种口径/候选。
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    a = np.array([1.0, 2.0, 3.0])
    print("[tools] sample vec=", a.tolist(), "norm=", float(np.linalg.norm(a)))


if __name__ == "__main__":
    main()
