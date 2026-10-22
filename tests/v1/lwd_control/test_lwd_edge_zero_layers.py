#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm project (LWD line).
#
"""Source-scan guard: make_layers pins the LWD edge at zero decoder layers.

The seam is the single chokepoint every transformer model builds its
decoder stack through (89+ model files plus VL compositions). The guard
locks: (1) the edge branch exists and keys off lwd_config.is_edge;
(2) the stock PP indices stay untouched for cloud / non-LWD processes;
(3) LwdConfig keeps the is_edge predicate.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UTILS = _REPO_ROOT / "vllm/model_executor/models/utils.py"
_ENTRY = _REPO_ROOT / "vllm/config/lwd/entry.py"


def test_make_layers_pins_edge_at_zero_layers() -> None:
    src = _UTILS.read_text(encoding="utf-8")
    start = src.index("def make_layers(")
    end = src.find("\ndef ", start + 1)
    block = src[start:] if end < 0 else src[start:end]
    assert "get_current_vllm_config_or_none()" in block
    assert 'getattr(vllm_config, "lwd_config", None)' in block
    assert "lwd_config.is_edge" in block
    assert "start_layer = end_layer = 0" in block
    # The placeholder assembly below must stay the stock three-way build.
    assert "[PPMissingLayer() for _ in range(start_layer)]" in block
    assert "[PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]" in block


def test_lwd_config_keeps_is_edge() -> None:
    src = _ENTRY.read_text(encoding="utf-8")
    assert "def is_edge(self) -> bool:" in src
    assert 'return self.role == "edge"' in src
