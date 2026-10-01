#!/usr/bin/env python3
"""Phase 2 validation visualization (pose-fusion filter plan, see
/home/nikitakarpuks/.claude/plans/wild-fluttering-dolphin.md) -- shows the
per-detection quality signals (confidence, mean reprojection error) that are
now correctly plumbed all the way to the committed solution dict for EVERY
search tier, including brute-force cold recovery (previously silently
defaulted to confidence=1.0, see src/pose_search.py's brute_search_tier fix).

Data: a real run of the actual pipeline (main.py, unmodified behavior --
fusion filter not wired in yet, this is Phase 2, plumbing only) over
config/config.yml's own frame_range (2500-3000), captured via a monkeypatch
on ControllerTracker._commit_fused_solution (see scratchpad/
phase2_signal_capture.py) -- no repo output files were touched, no filter
logic ran. This window contains the real occlusion + identity-swap-prone
re-acquisition case (frame_idx ~275-440 here) documented in prior sessions.

Usage: python visualize_phase2_quality_signals.py
"""
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

_DATA_PATH = Path(__file__).parent / "visualization" / "phase2_chart_data.json"

# Categorical identity colors -- same validated-palette hexes already used for
# the two-series case elsewhere in this repo (visualize_position_orientation.py's
# vision/imu pair), reused here for left/right controller identity, fixed order.
_COLOR_LEFT = "#2a78d6"   # blue
_COLOR_RIGHT = "#eb6834"  # orange
_COLOR_GAP_SHADE = "rgba(120,120,120,0.10)"  # same "no track" shade convention
                                              # as visualize_position_orientation.py

_METHOD_SYMBOL = {
    "proximity": "circle",
    "prior_constrained_p2p": "circle",
    "prior_constrained_p1p": "circle",
    "p3p_systematic": "diamond",  # brute-force cold recovery -- the case that
                                   # previously reported confidence=1.0 regardless
                                   # of how weak the recovery actually was
}
_BRUTE_METHOD = "p3p_systematic"


def _no_track_gaps(all_frame_idxs: set, lo: int, hi: int):
    """Contiguous frame_idx ranges in [lo, hi] where NEITHER controller has a
    committed solution -- i.e. both controllers are fully lost."""
    gaps = []
    run_start = None
    for i in range(lo, hi + 1):
        if i not in all_frame_idxs:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                gaps.append((run_start, i - 1))
                run_start = None
    if run_start is not None:
        gaps.append((run_start, hi))
    return gaps


def main():
    data = json.loads(_DATA_PATH.read_text())
    lo = min(r["frame_idx"] for recs in data.values() for r in recs)
    hi = max(r["frame_idx"] for recs in data.values() for r in recs)
    all_committed = {r["frame_idx"] for recs in data.values() for r in recs}
    gaps = _no_track_gaps(all_committed, lo, hi)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Solution confidence (higher = more trustworthy)",
                         "Mean reprojection error (px, lower = better fit)"),
    )

    color = {"left_controller": _COLOR_LEFT, "right_controller": _COLOR_RIGHT}
    label = {"left_controller": "left controller", "right_controller": "right controller"}

    for ctrl_name in ("left_controller", "right_controller"):
        recs = data[ctrl_name]
        xs = [r["frame_idx"] for r in recs]
        confs = [r["confidence"] for r in recs]
        errs = [r["error"] for r in recs]
        symbols = [_METHOD_SYMBOL.get(r["method"], "circle") for r in recs]
        is_brute = [r["method"] == _BRUTE_METHOD for r in recs]
        hover = [f"frame {r['frame_idx']}<br>{label[ctrl_name]}<br>method={r['method']}"
                 f"<br>confidence={r['confidence']:.3f}<br>error={r['error']:.3f}px"
                 for r in recs]

        fig.add_trace(go.Scatter(
            x=xs, y=confs, mode="lines+markers", name=label[ctrl_name],
            legendgroup=ctrl_name, showlegend=True,
            line=dict(color=color[ctrl_name], width=1.5),
            marker=dict(size=[9 if b else 6 for b in is_brute], symbol=symbols,
                        color=color[ctrl_name],
                        line=dict(width=[1.5 if b else 0 for b in is_brute], color="white")),
            text=hover, hoverinfo="text",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=xs, y=errs, mode="lines+markers", name=label[ctrl_name],
            legendgroup=ctrl_name, showlegend=False,
            line=dict(color=color[ctrl_name], width=1.5),
            marker=dict(size=[9 if b else 6 for b in is_brute], symbol=symbols,
                        color=color[ctrl_name],
                        line=dict(width=[1.5 if b else 0 for b in is_brute], color="white")),
            text=hover, hoverinfo="text",
        ), row=2, col=1)

    # A dummy legend-only trace to explain the diamond/circle secondary encoding
    # (method), since color already carries controller identity.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="brute-force cold recovery",
        marker=dict(size=9, symbol="diamond", color="#888", line=dict(width=1.5, color="white")),
        showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="proximity / constrained (warm)",
        marker=dict(size=6, symbol="circle", color="#888"),
        showlegend=True,
    ), row=1, col=1)

    for r0, r1 in gaps:
        for row in (1, 2):
            fig.add_vrect(x0=r0 - 0.5, x1=r1 + 0.5, fillcolor=_COLOR_GAP_SHADE,
                          line_width=0, row=row, col=1)
    if gaps:
        r0, r1 = gaps[-1] if (gaps[-1][1] - gaps[-1][0]) > (gaps[0][1] - gaps[0][0]) else gaps[0]
        long_gap = max(gaps, key=lambda g: g[1] - g[0])
        fig.add_annotation(
            x=(long_gap[0] + long_gap[1]) / 2, y=1.0, yref="y1 domain" if False else None,
            text=f"both controllers lost<br>({long_gap[1]-long_gap[0]+1} frames)",
            showarrow=False, row=1, col=1, font=dict(size=10, color="#666"),
        )

    fig.add_annotation(
        x=390, y=1.05, xref="x2", yref="paper",
        text="re-acquisition window with known left/right identity-swap risk<br>"
             "(vision alone can't disambiguate near-mirror-image constellations here --"
             " this is exactly the gap the pose-fusion filter (Phase 3/4) targets)",
        showarrow=False, font=dict(size=11, color="#444"), align="center",
    )

    fig.update_xaxes(title_text="frame index (within config's frame_range 2500-3000)", row=2, col=1)
    fig.update_yaxes(title_text="confidence", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="error (px)", row=2, col=1)
    fig.update_layout(
        title="Phase 2: quality signals now available on every committed solution "
              "(including brute-force cold recovery)",
        height=720, width=1180,
        legend=dict(orientation="h", yanchor="bottom", y=1.09, xanchor="left", x=0),
        margin=dict(t=140),
        plot_bgcolor="white",
    )

    out_path = Path(__file__).parent / "visualization" / "pose_fusion_phase2_quality_signals.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
