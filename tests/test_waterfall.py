"""The waterfall's colour scaling (web/waterfall.js), run through node. Skipped without node."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parent.parent / "web" / "waterfall.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def run_js(expr: str):
    script = f"const W = require({json.dumps(str(JS))});\nconsole.log(JSON.stringify({expr}));"
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def rows_js(body: str) -> str:
    """JS that builds `rows` ({t, a: Uint8Array(52)}) with `body` giving a[i] for frame k."""
    return (
        "Array.from({length: 1000}, (_, k) => ({t: k / 100, "
        f"a: Uint8Array.from({{length: 52}}, (_, i) => {body})}}))"
    )


def test_amp_range_ignores_a_single_spike():
    # values spread 7..17 like the real board, plus one spike of 60
    rows = rows_js("k === 500 && i === 3 ? 60 : 7 + (k * 7 + i * 3) % 11")
    lo, hi = run_js(f"W.ampRange({rows}, 52)")
    assert 7 <= lo <= 8
    assert 16 <= hi <= 17


def test_amp_range_never_collapses():
    lo, hi = run_js(f"W.ampRange({rows_js('12')}, 52)")
    assert hi - lo >= 1


def test_change_scale_constant_signal():
    s = run_js(
        f"(s => [Array.from(s.med), s.noise, s.range])(W.changeScale({rows_js('i + 5')}, 52))"
    )
    med, noise, rng = s
    assert med == [i + 5 for i in range(52)]
    assert noise == 0
    assert rng == 6


# |a - median| in a still room, measured on the real board (1083 frames x 52 subcarriers):
# 0: 32%  1: 42%  2: 15%  3: 5.5%  4: 2.5%  5: 1.4%  6+: 1.5%  -> the same mix in 200 steps
_COUNTS = {0: 64, 1: 84, 2: 30, 3: 11, 4: 5, 5: 3, 6: 3}
_DEVS = [d * (1 if n % 2 else -1) for d, c in _COUNTS.items() for n in range(c)]
STILL = f"10 + {_DEVS}[(k * 7 + i * 13) % {len(_DEVS)}]".replace(" ", "")


def test_change_still_room_is_mostly_dark():
    rows = rows_js(STILL)
    frac = run_js(
        f"(() => {{ const rows = {rows}, s = W.changeScale(rows, 52); let bright = 0, n = 0;"
        "for (const r of rows) for (let i = 0; i < 52; i++) {"
        " n++; if (W.changeT(Math.abs(r.a[i] - s.med[i]), s) >= 0.25) bright++; }"
        "return bright / n; })()"
    )
    assert frac < 0.06


def test_change_scale_follows_motion():
    # 80% of frames still at 10, 20% jump to 20 -> deviation 10 dominates the top 2%
    rows = rows_js("k % 5 === 0 ? 20 : 10")
    med, t_move, t_still = run_js(
        f"(s => [Array.from(s.med), W.changeT(10, s), W.changeT(0, s)])(W.changeScale({rows}, 52))"
    )
    assert med == [10] * 52
    assert t_move >= 0.9
    assert t_still == 0


def test_turbo_runs_dark_to_cyan_green_red():
    lo, q1, mid, hi = run_js("[W.turbo(0), W.turbo(0.25), W.turbo(0.5), W.turbo(1)]")
    for c in (lo, q1, mid, hi):
        assert all(0 <= v <= 255 for v in c)
    assert max(lo) < 60  # near black: a still room looks dark
    assert q1[2] > q1[0]  # blue/cyan at the low quarter
    assert mid[1] > mid[0] and mid[1] > mid[2]  # green in the middle
    assert hi[0] > hi[1] and hi[0] > hi[2]  # red at the top
