"""
Interactive saccade curation GUI for Jupyter notebooks.

Provides a Plotly FigureWidget + ipywidgets interface for:
- Reviewing auto-detected saccade events overlaid on position/velocity traces
- Adding missed saccades via box-select on the position trace
- Deleting false-positive events via click on event markers
- Saving curated events to CSV + metadata JSON

Usage (from notebook cell)::

    from sleap.saccade_curation import build_curation_gui

    curation_widget, curation_state = build_curation_gui(
        df_work=df_work,
        auto_events=all_saccades_df,
        vel_thresh=vel_thresh,
        cell2_params=cell2_params,
        metadata=metadata,
        metadata_path=metadata_path,
        save_dir=downsampled_output_dir,
    )
    display(curation_widget)

    # In a later cell, after further analysis:
    # curation_state.save(save_dir, metadata, metadata_path, cell2_params)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import ipywidgets as widgets

    _WIDGETS_OK = True
except ImportError:
    _WIDGETS_OK = False

# ---------------------------------------------------------------------------
# Event schema shared between auto and manual saccades
# ---------------------------------------------------------------------------
EVENT_COLUMNS = [
    "direction",
    "time",
    "velocity",
    "start_time",
    "end_time",
    "duration",
    "start_position",
    "end_position",
    "amplitude",
    "displacement",
    "start_frame_idx",
    "peak_frame_idx",
    "end_frame_idx",
    "source",
]


def quantify_manual_saccade(
    start_time: float,
    end_time: float,
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
) -> dict:
    """Quantify a manually selected saccade from start/end times.

    Uses the same measurement schema as auto-detected events so the two
    sources merge cleanly.
    """
    if end_time <= start_time:
        start_time, end_time = end_time, start_time

    start_idx = int(np.clip(np.searchsorted(t, start_time), 0, len(t) - 1))
    end_idx = int(np.clip(np.searchsorted(t, end_time), 0, len(t) - 1))
    if start_idx == end_idx:
        end_idx = min(end_idx + 1, len(t) - 1)

    window_vel = v[start_idx : end_idx + 1]
    if len(window_vel) == 0 or np.all(np.isnan(window_vel)):
        peak_rel = 0
    else:
        abs_vel = np.abs(np.nan_to_num(window_vel, nan=0.0))
        peak_rel = int(np.argmax(abs_vel))
    peak_idx = start_idx + peak_rel

    displacement = float(x[end_idx] - x[start_idx])

    return {
        "direction": "upward" if displacement >= 0 else "downward",
        "time": float(t[peak_idx]),
        "velocity": float(v[peak_idx]) if np.isfinite(v[peak_idx]) else 0.0,
        "start_time": float(t[start_idx]),
        "end_time": float(t[end_idx]),
        "duration": float(t[end_idx] - t[start_idx]),
        "start_position": float(x[start_idx]),
        "end_position": float(x[end_idx]),
        "amplitude": float(abs(displacement)),
        "displacement": displacement,
        "start_frame_idx": int(f[start_idx]),
        "peak_frame_idx": int(f[peak_idx]),
        "end_frame_idx": int(f[end_idx]),
        "source": "manual",
    }


# ---------------------------------------------------------------------------
# Curation state
# ---------------------------------------------------------------------------
class CurationState:
    """Non-destructive curation state for saccade events.

    Auto-detected events are never mutated; deletions are tracked as a mask.
    Manual additions are stored separately.  Both are merged on demand via
    :meth:`get_final_events`.
    """

    def __init__(
        self,
        auto_events: pd.DataFrame,
        t: np.ndarray,
        x: np.ndarray,
        v: np.ndarray,
        f: np.ndarray,
        *,
        deleted_auto_indices: set[int] | None = None,
        manual_events: list[dict] | None = None,
    ):
        self._auto = auto_events.copy().reset_index(drop=True)
        if "source" not in self._auto.columns:
            self._auto["source"] = "auto"
        self._deleted_auto: set[int] = (
            deleted_auto_indices if deleted_auto_indices is not None else set()
        )
        self._manual: list[dict] = (
            manual_events.copy() if manual_events is not None else []
        )
        self._undo_stack: list[tuple[str, Any]] = []

        self._t = t
        self._x = x
        self._v = v
        self._f = f

    # -- mutators ----------------------------------------------------------

    def add_manual_saccade(self, start_time: float, end_time: float) -> dict:
        """Quantify and record a manually added saccade. Returns the event."""
        event = quantify_manual_saccade(
            start_time, end_time, self._t, self._x, self._v, self._f
        )
        self._manual.append(event)
        self._undo_stack.append(("add_manual", len(self._manual) - 1))
        return event

    def delete_auto_event(self, auto_idx: int) -> None:
        if auto_idx in self._deleted_auto:
            return
        self._deleted_auto.add(auto_idx)
        self._undo_stack.append(("delete_auto", auto_idx))

    def delete_manual_event(self, manual_idx: int) -> None:
        if 0 <= manual_idx < len(self._manual):
            removed = self._manual.pop(manual_idx)
            self._undo_stack.append(("remove_manual", (manual_idx, removed)))

    def undo(self) -> str | None:
        """Undo the last action. Returns a description or None if empty."""
        if not self._undo_stack:
            return None
        action, payload = self._undo_stack.pop()
        if action == "add_manual":
            idx = payload
            if 0 <= idx < len(self._manual):
                self._manual.pop(idx)
            return "Undid manual add"
        if action == "delete_auto":
            self._deleted_auto.discard(payload)
            return f"Restored auto event {payload}"
        if action == "remove_manual":
            idx, event = payload
            self._manual.insert(idx, event)
            return f"Restored manual event at {idx}"
        return None

    # -- queries -----------------------------------------------------------

    @property
    def n_auto_kept(self) -> int:
        return len(self._auto) - len(self._deleted_auto)

    @property
    def n_auto_deleted(self) -> int:
        return len(self._deleted_auto)

    @property
    def n_manual(self) -> int:
        return len(self._manual)

    @property
    def n_total(self) -> int:
        return self.n_auto_kept + self.n_manual

    def get_kept_auto_events(self) -> pd.DataFrame:
        mask = ~self._auto.index.isin(self._deleted_auto)
        return self._auto.loc[mask].reset_index(drop=True)

    def get_manual_events_df(self) -> pd.DataFrame:
        if not self._manual:
            return pd.DataFrame(columns=EVENT_COLUMNS)
        return pd.DataFrame(self._manual)

    def get_final_events(self) -> pd.DataFrame:
        kept = self.get_kept_auto_events()
        manual = self.get_manual_events_df()
        if len(kept) == 0:
            return manual
        if len(manual) == 0:
            return kept
        merged = pd.concat([kept, manual], ignore_index=True)
        if len(merged) > 0:
            merged = merged.sort_values("time").reset_index(drop=True)
        return merged

    def summary_text(self) -> str:
        return (
            f"auto_kept={self.n_auto_kept}  "
            f"manual_deleted={self.n_auto_deleted}  "
            f"manual_added={self.n_manual}  "
            f"total={self.n_total}"
        )

    # -- persistence -------------------------------------------------------

    def save(
        self,
        save_dir: Path,
        metadata: dict,
        metadata_path: Path,
        cell2_params: dict,
    ) -> str:
        """Save curated events CSV and update metadata JSON."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        final = self.get_final_events()
        video = metadata.get("eye_with_least_low_confidence", "")
        # VideoData1: upward→NT, downward→TN; VideoData2: upward→TN, downward→NT
        if video == "VideoData1":
            final["TNT_direction"] = final["direction"].map(
                {"upward": "NT", "downward": "TN"}
            )
        elif video == "VideoData2":
            final["TNT_direction"] = final["direction"].map(
                {"upward": "TN", "downward": "NT"}
            )
        else:
            final["TNT_direction"] = ""
        final["TNT_direction"] = final["TNT_direction"].fillna("")
        csv_path = save_dir / "curated_saccade_events.csv"
        final.to_csv(csv_path, index=False)

        # Save parquet with ±5 s snippets centered on each saccade
        SNIPPET_RAD_S = 5.0
        snippet_rows: list[dict] = []
        for ev_idx, (_, row) in enumerate(final.iterrows()):
            center = float(row["time"])
            lo, hi = center - SNIPPET_RAD_S, center + SNIPPET_RAD_S
            mask = (self._t >= lo) & (self._t <= hi)
            t_slice = self._t[mask]
            x_slice = self._x[mask]
            f_slice = self._f[mask]
            for i in range(len(t_slice)):
                snippet_rows.append(
                    {
                        "event_idx": ev_idx,
                        "time_rel": float(t_slice[i] - center),
                        "time_abs": float(t_slice[i]),
                        "X_raw": float(x_slice[i]),
                        "frame_idx": int(f_slice[i]),
                    }
                )
        snippets_df = pd.DataFrame(snippet_rows)
        snippets_path = save_dir / "curated_saccade_snippets.parquet"
        snippets_df.to_parquet(snippets_path, index=False)

        metadata["saccade_detection_parameters"] = cell2_params
        metadata["curation_summary"] = {
            "auto_kept": self.n_auto_kept,
            "manual_deleted": self.n_auto_deleted,
            "manual_added": self.n_manual,
            "total_final": self.n_total,
            "curation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(metadata_path, "w") as fh:
            json.dump(metadata, fh, indent=2)

        return (
            f"Saved {len(final)} events to {csv_path.name}, "
            f"{len(snippet_rows)} snippet rows to {snippets_path.name}, "
            f"and updated {metadata_path.name}"
        )


# ---------------------------------------------------------------------------
# GUI builder  (event-centred windowed view with W/Q keyboard navigation)
# ---------------------------------------------------------------------------

# Trace indices in the FigureWidget (fixed layout).
_IDX_POS = 0
_IDX_UP_AUTO = 1
_IDX_DOWN_AUTO = 2
_IDX_MANUAL = 3

HALF_WINDOW_S = 2.0  # +/- seconds shown around the current event


def _build_figure(eye_label: str) -> go.FigureWidget:
    """Build the single-panel FigureWidget with empty traces (filled by navigate)."""
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=("X position",),
    )

    # Trace 0: position (empty, filled per-window).
    # mode="lines+markers" so on_click can fire on data points for the
    # two-click "Add Saccade" workflow; markers are near-invisible.
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name="X_raw",
            line=dict(width=1, color="royalblue"),
            marker=dict(size=3, opacity=0.0, color="royalblue"),
        ),
        row=1,
        col=1,
    )
    # Trace 1: auto upward markers
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            name="auto upward",
            marker=dict(color="limegreen", size=7, symbol="circle"),
        ),
        row=1,
        col=1,
    )
    # Trace 2: auto downward markers
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            name="auto downward",
            marker=dict(color="mediumpurple", size=7, symbol="circle"),
        ),
        row=1,
        col=1,
    )
    # Trace 3: manual event markers
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            name="manual",
            marker=dict(color="darkorange", size=8, symbol="diamond"),
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        title=f"Saccade Curation {eye_label}",
        template="plotly_white",
        height=380,
        dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    fig.update_xaxes(title_text="Relative time (s)", row=1, col=1)
    fig.update_yaxes(title_text="X position (px)", row=1, col=1)

    return go.FigureWidget(fig)


def _find_nearest_event_index(
    click_time_rel: float,
    state: CurationState,
    t0: float,
    tolerance_s: float = 2.0,
) -> tuple[str, int] | None:
    """Find the auto or manual event nearest to a clicked relative time.

    Returns ``("auto", original_df_index)`` or ``("manual", list_index)``,
    or ``None`` if nothing is within *tolerance_s*.
    """
    best_source: str | None = None
    best_idx: int = -1
    best_dist: float = tolerance_s
    click_abs = click_time_rel + t0

    kept = state.get_kept_auto_events()
    if len(kept) > 0:
        dists = (kept["start_time"] - click_abs).abs().to_numpy()
        i_min = int(np.argmin(dists))
        if dists[i_min] < best_dist:
            best_dist = dists[i_min]
            best_source = "auto"
            mask = ~state._auto.index.isin(state._deleted_auto)
            best_idx = int(state._auto.index[mask][i_min])

    manual = state.get_manual_events_df()
    if len(manual) > 0:
        dists = (manual["start_time"] - click_abs).abs().to_numpy()
        i_min = int(np.argmin(dists))
        if dists[i_min] < best_dist:
            best_dist = dists[i_min]
            best_source = "manual"
            best_idx = i_min

    if best_source is None:
        return None
    return best_source, best_idx


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_curation_gui(
    df_work: pd.DataFrame,
    auto_events: pd.DataFrame,
    vel_thresh: float,
    cell2_params: dict,
    metadata: dict,
    metadata_path: Path | str,
    save_dir: Path | str,
    eye_label: str = "",
) -> tuple[widgets.VBox, CurationState]:
    """Build and return the curation widget and state for later saving.

    Parameters
    ----------
    df_work : DataFrame
        Working DataFrame with columns ``Seconds``, ``X_raw``,
        ``vel_x_smooth``, ``frame_idx``.
    auto_events : DataFrame
        Auto-detected saccade events (output of Cell 7 filtering).
    vel_thresh : float
        Velocity threshold used for detection.
    cell2_params : dict
        Detection parameter snapshot for metadata persistence.
    metadata : dict
        Mutable metadata dict (loaded from ``saccade_input_metadata.json``).
    metadata_path : Path
        Path to the metadata JSON file.
    save_dir : Path
        Directory for saving curated CSV.
    eye_label : str, optional
        Label for figure title (e.g. "VideoData1, eye=L").
    """
    if not _WIDGETS_OK:
        raise ImportError(
            "ipywidgets is required for the curation GUI. "
            "Install with: pip install ipywidgets"
        )

    from ipyevents import Event as DomEvent

    metadata_path = Path(metadata_path)
    save_dir = Path(save_dir)

    t_arr = df_work["Seconds"].to_numpy(dtype=float)
    x_arr = df_work["X_raw"].to_numpy(dtype=float)
    v_arr = df_work["vel_x_smooth"].to_numpy(dtype=float)
    f_arr = df_work["frame_idx"].to_numpy(dtype=int)
    t0 = float(t_arr[0])

    # Load previous curation if saved (so rerunning Cell 9 preserves edits)
    csv_path = save_dir / "curated_saccade_events.csv"
    deleted_auto: set[int] | None = None
    manual_events: list[dict] | None = None
    if csv_path.exists():
        curated = pd.read_csv(csv_path)
    else:
        curated = None
    if curated is not None and len(curated) > 0 and "source" in curated.columns:
        # Normalize source column (strip whitespace, handle CSV quirks)
        curated["source"] = curated["source"].astype(str).str.strip()
        auto_curated = curated[curated["source"] == "auto"]
        manual_df = curated[curated["source"] == "manual"]
        # Match curated auto events to original by time → kept indices
        orig_times = auto_events["time"].to_numpy(dtype=float)
        kept: set[int] = set()
        for _, row in auto_curated.iterrows():
            i = int(np.argmin(np.abs(orig_times - float(row["time"]))))
            kept.add(i)
        deleted_auto = set(range(len(auto_events))) - kept
        # Build manual_events with EVENT_COLUMNS schema (native Python types)
        if len(manual_df) > 0:
            manual_events = []
            for _, row in manual_df.iterrows():
                ev = {
                    "direction": str(row["direction"]).strip(),
                    "time": float(row["time"]),
                    "velocity": float(row["velocity"]),
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "duration": float(row["duration"]),
                    "start_position": float(row["start_position"]),
                    "end_position": float(row["end_position"]),
                    "amplitude": float(row["amplitude"]),
                    "displacement": float(row["displacement"]),
                    "start_frame_idx": int(float(row["start_frame_idx"])),
                    "peak_frame_idx": int(float(row["peak_frame_idx"])),
                    "end_frame_idx": int(float(row["end_frame_idx"])),
                    "source": "manual",
                }
                manual_events.append(ev)
        else:
            manual_events = []

    state = CurationState(
        auto_events,
        t_arr,
        x_arr,
        v_arr,
        f_arr,
        deleted_auto_indices=deleted_auto,
        manual_events=manual_events,
    )
    nav = {"idx": 0, "gap_idx": 0}

    # -- Build figure (empty — populated by first _navigate_to call) -------
    fw = _build_figure(eye_label)

    # -- Widgets -----------------------------------------------------------
    prev_btn = widgets.Button(
        description="< Prev (Q)",
        button_style="info",
        layout=widgets.Layout(width="120px"),
    )
    next_btn = widgets.Button(
        description="Next (W) >",
        button_style="info",
        layout=widgets.Layout(width="120px"),
    )
    event_label = widgets.HTML(value="")
    undo_btn = widgets.Button(
        description="Undo (Z)",
        icon="undo",
        button_style="warning",
        layout=widgets.Layout(width="120px"),
    )
    done_btn = widgets.Button(
        description="Done curating",
        icon="check",
        button_style="success",
        layout=widgets.Layout(width="140px"),
    )
    event_jump_int = widgets.BoundedIntText(
        value=1,
        min=1,
        max=9999,
        description="Event #:",
        layout=widgets.Layout(width="180px"),
        style={"description_width": "60px"},
    )
    go_event_btn = widgets.Button(
        description="Go",
        button_style="info",
        layout=widgets.Layout(width="50px"),
    )
    gap_jump_int = widgets.BoundedIntText(
        value=1,
        min=1,
        max=9999,
        description="Gap #:",
        layout=widgets.Layout(width="180px"),
        style={"description_width": "60px"},
    )
    go_gap_btn = widgets.Button(
        description="Go",
        button_style="info",
        layout=widgets.Layout(width="50px"),
    )
    shortcuts_html = widgets.HTML(
        value=(
            "<span style='color:gray; font-size:0.9em'>"
            "<b>W</b>/<b>Q</b> events &nbsp;|&nbsp; "
            "<b>E</b>/<b>R</b> prev/next gap &nbsp;|&nbsp; "
            "<b>Click</b> start + end to add &nbsp;|&nbsp; "
            "<b>]</b> delete &nbsp;|&nbsp; "
            "<b>Z</b> undo &nbsp;|&nbsp; "
            "<b>Jump:</b> Event/Gap # + Go"
            "</span>"
        )
    )
    status_html = widgets.HTML(
        value="<i>Click on the widget, then use <b>W</b>/<b>Q</b> to navigate events, "
        "<b>E</b>/<b>R</b> for prev/next gap. "
        "<b>Click</b> twice to add a saccade. "
        "<b>]</b> to delete. "
        "Click <b>Done curating</b> when finished.</i>"
    )
    summary_html = widgets.HTML(
        value=f"<b style='font-size:1.1em'>{state.summary_text()}</b>"
    )

    def _update_summary():
        summary_html.value = f"<b style='font-size:1.1em'>{state.summary_text()}</b>"
        n_ev = len(state.get_final_events())
        event_jump_int.max = max(1, n_ev)
        gap_jump_int.max = max(1, len(_compute_gaps()))

    def _set_status(msg: str):
        status_html.value = f"<i>{msg}</i>"

    # -- Core: navigate to an absolute time ---------------------------------
    MIN_Y_RANGE_PX = 10.0

    def _navigate_to_time(
        center_abs: float,
        label_html: str | None = None,
        force_min_y_range: float | None = None,
        gap_span_abs: tuple[float, float] | None = None,
    ):
        """Render the +/-2s window around *center_abs* (absolute seconds).

        If *label_html* is provided it is written to the event_label widget;
        otherwise the caller is responsible for updating it.
        If *force_min_y_range* is set (e.g. 10), y-axes use at least that range
        for easier visual inspection when data is flat.
        If *gap_span_abs* is (start, end) in absolute seconds, use it as the window
        and draw a gap overlay; otherwise use +/-2s around center.
        """
        events = state.get_final_events()
        if gap_span_abs is not None:
            win_start, win_end = gap_span_abs[0], gap_span_abs[1]
        else:
            win_start = center_abs - HALF_WINDOW_S
            win_end = center_abs + HALF_WINDOW_S

        mask = (t_arr >= win_start) & (t_arr <= win_end)
        win_t_rel = t_arr[mask] - t0
        win_x = x_arr[mask]
        if len(win_t_rel) == 0 and gap_span_abs is not None:
            import warnings

            warnings.warn(
                f"No trace data in gap window t_rel=[{win_start - t0:.1f}, {win_end - t0:.1f}] s. "
                "Check that df_work covers this time range."
            )

        # Events visible in this window
        if len(events) > 0:
            win_events = events[
                (events["start_time"] <= win_end) & (events["end_time"] >= win_start)
            ]
        else:
            win_events = events

        auto_in_win = (
            win_events[
                (win_events["source"] == "auto")
                if "source" in win_events.columns
                else pd.Series(True, index=win_events.index)
            ]
            if len(win_events) > 0
            else win_events
        )
        manual_in_win = (
            win_events[
                (win_events["source"] == "manual")
                if "source" in win_events.columns
                else pd.Series(False, index=win_events.index)
            ]
            if len(win_events) > 0
            else win_events
        )
        up_in_win = (
            auto_in_win[auto_in_win["direction"] == "upward"]
            if len(auto_in_win) > 0
            else auto_in_win
        )
        down_in_win = (
            auto_in_win[auto_in_win["direction"] == "downward"]
            if len(auto_in_win) > 0
            else auto_in_win
        )

        def _marker_xy(evdf):
            if len(evdf) == 0:
                return np.array([]), np.array([])
            st = evdf["start_time"].to_numpy(dtype=float)
            return st - t0, np.interp(st, t_arr, x_arr)

        up_x, up_y = _marker_xy(up_in_win)
        down_x, down_y = _marker_xy(down_in_win)
        man_x, man_y = _marker_xy(manual_in_win)

        shapes = [
            s
            for s in (fw.layout.shapes or [])
            if getattr(s, "name", None) not in ("_sac_span", "_gap_span")
        ]
        if gap_span_abs is not None:
            g_start, g_end = gap_span_abs
            shapes.append(
                go.layout.Shape(
                    type="rect",
                    x0=float(g_start - t0),
                    x1=float(g_end - t0),
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="y domain",
                    fillcolor="rgba(0,128,128,0.08)",
                    line=dict(color="rgba(0,128,128,0.3)", width=1, dash="dot"),
                    name="_gap_span",
                )
            )
        for _, wev in win_events.iterrows():
            is_up = wev["direction"] == "upward"
            shapes.append(
                go.layout.Shape(
                    type="rect",
                    x0=float(wev["start_time"] - t0),
                    x1=float(wev["end_time"] - t0),
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="y domain",
                    fillcolor="rgba(34,139,34,0.10)"
                    if is_up
                    else "rgba(128,0,128,0.10)",
                    line=dict(
                        color="rgba(34,139,34,0.25)"
                        if is_up
                        else "rgba(128,0,128,0.25)",
                        width=1,
                    ),
                    name="_sac_span",
                )
            )

        def _y_range(arr: np.ndarray, min_range: float | None) -> list[float] | None:
            """Return [ymin, ymax] with at least min_range span, or None for autorange."""
            valid = arr[np.isfinite(arr)]
            if len(valid) == 0:
                return None
            lo, hi = float(np.min(valid)), float(np.max(valid))
            if min_range is None or (hi - lo) > min_range:
                return None  # autorange when range > min_range
            mid = (lo + hi) / 2
            half = min_range / 2
            return [mid - half, mid + half]

        pos_range = _y_range(win_x, force_min_y_range)

        with fw.batch_update():
            fw.data[_IDX_POS].x = win_t_rel
            fw.data[_IDX_POS].y = win_x
            fw.data[_IDX_UP_AUTO].x = up_x
            fw.data[_IDX_UP_AUTO].y = up_y
            fw.data[_IDX_DOWN_AUTO].x = down_x
            fw.data[_IDX_DOWN_AUTO].y = down_y
            fw.data[_IDX_MANUAL].x = man_x
            fw.data[_IDX_MANUAL].y = man_y
            fw.layout.shapes = shapes
            fw.layout.xaxis.range = [win_start - t0, win_end - t0]
            if pos_range is not None:
                fw.layout.yaxis.range = pos_range
                fw.layout.yaxis.autorange = False
            else:
                fw.layout.yaxis.autorange = True

        # Force autorange to take effect when range > 10 (Plotly can retain old range)
        if pos_range is None and force_min_y_range is not None:
            fw.update_layout(yaxis=dict(autorange=True))

        if label_html is not None:
            event_label.value = label_html
        _update_summary()

    # -- Navigate to event by index (W/Q) ----------------------------------
    def _navigate_to(idx: int):
        events = state.get_final_events()
        n_events = len(events)
        if n_events == 0:
            event_label.value = "<b>No events</b>"
            return

        idx = int(np.clip(idx, 0, n_events - 1))
        nav["idx"] = idx
        ev = events.iloc[idx]

        src = ev.get("source", "auto")
        label = (
            f"<b>Event {idx + 1}/{n_events}</b> &nbsp;|&nbsp; "
            f"{ev['direction']} &nbsp;|&nbsp; "
            f"{ev['amplitude']:.1f} px &nbsp;|&nbsp; "
            f"{ev['duration'] * 1000:.0f} ms &nbsp;|&nbsp; "
            f"<span style='color:{'darkorange' if src == 'manual' else 'gray'}'>"
            f"{src}</span>"
        )
        _navigate_to_time(float(ev["time"]), label)
        event_jump_int.value = idx + 1
        event_jump_int.max = n_events

    # -- Gap navigation (E/R) ----------------------------------------------
    def _compute_gaps() -> list[tuple[float, float]]:
        """Return [(midpoint_abs, gap_duration), ...] in temporal order.

        Uses true gap boundaries: end_time of one saccade to start_time of next
        (not peak-to-peak). Includes rec_start→first_event.start and
        last_event.end→rec_end. Gaps smaller than threshold are dropped;
        threshold is capped so long visible gaps are never filtered out.
        """
        events = state.get_final_events()
        rec_start = float(t_arr[0])
        rec_end = float(t_arr[-1])

        if len(events) == 0:
            mid = (rec_start + rec_end) / 2.0
            return [(mid, rec_end - rec_start)]

        ev_sorted = events.sort_values("time").reset_index(drop=True)
        starts = ev_sorted["start_time"].to_numpy(dtype=float)
        ends = ev_sorted["end_time"].to_numpy(dtype=float)

        # True gaps: rec_start→first.start, first.end→second.start, ..., last.end→rec_end
        gap_starts = np.concatenate([[rec_start], ends])
        gap_ends = np.concatenate([starts, [rec_end]])
        durations = gap_ends - gap_starts

        # Threshold: filter short gaps, but cap at 12 s so long visible gaps are never missed
        min_dur = 2.0 * HALF_WINDOW_S  # 4 s floor
        pos_durations = durations[durations > 0]
        if len(pos_durations) > 0:
            median_isi = float(np.median(pos_durations))
            threshold = min(max(2.0 * median_isi, min_dur), 12.0)  # cap at 12 s
        else:
            threshold = min_dur

        gaps = []
        for i, dur in enumerate(durations):
            if dur > 0 and dur >= threshold:
                mid = float(gap_starts[i] + dur / 2.0)
                gaps.append((mid, float(dur)))

        gaps.sort(key=lambda g: g[0])  # temporal order along the trace
        return gaps

    gaps = _compute_gaps()
    if len(gaps) > 0:
        min_gap = min(gaps, key=lambda g: g[1])
        max_gap = max(gaps, key=lambda g: g[1])
        min_start, min_end = min_gap[0] - min_gap[1] / 2, min_gap[0] + min_gap[1] / 2
        max_start, max_end = max_gap[0] - max_gap[1] / 2, max_gap[0] + max_gap[1] / 2
        print(
            f"Gap lengths: min={min_gap[1]:.1f} s at t_rel=[{min_start - t0:.1f}, {min_end - t0:.1f}] s, "
            f"max={max_gap[1]:.1f} s at t_rel=[{max_start - t0:.1f}, {max_end - t0:.1f}] s, "
            f"n={len(gaps)}"
        )
        first = gaps[0]
        first_start = first[0] - first[1] / 2.0
        first_end = first[0] + first[1] / 2.0
        rec_end = float(t_arr[-1])
        print(
            f"First gap (idx=0): t_rel=[{first_start - t0:.6f}, {first_end - t0:.6f}] s "
            f"(abs=[{first_start:.6f}, {first_end:.6f}]), dur={first[1]:.6f} s"
        )
        print(
            f"Recording range: t_abs=[{t0:.6f}, {rec_end:.6f}], t_rel=[0, {rec_end - t0:.6f}] s"
        )
        print("Curation window opening takes a few seconds to load")

    def _navigate_to_gap(gap_idx: int):
        gaps = _compute_gaps()
        n_gaps = len(gaps)
        if n_gaps == 0:
            _set_status("No significant gaps found.")
            return

        gap_idx = int(np.clip(gap_idx, 0, n_gaps - 1))
        nav["gap_idx"] = gap_idx
        mid, dur = gaps[gap_idx]

        label = (
            f"<b style='color:teal'>Gap {gap_idx + 1}/{n_gaps}</b> "
            f"&nbsp;|&nbsp; {dur:.1f} s"
        )
        gap_span = (mid - dur / 2.0, mid + dur / 2.0)
        _navigate_to_time(
            mid, label, force_min_y_range=MIN_Y_RANGE_PX, gap_span_abs=gap_span
        )
        _set_status(
            f"Gap {gap_idx + 1}/{n_gaps} ({dur:.1f} s). "
            "Click to add saccade, <b>E</b>/<b>R</b> for prev/next gap, "
            "<b>W</b>/<b>Q</b> to return to events."
        )
        gap_jump_int.value = gap_idx + 1
        gap_jump_int.max = n_gaps

    # Show the first event
    _navigate_to(0)
    gap_jump_int.max = max(1, len(_compute_gaps()))

    # -- Navigation callbacks ----------------------------------------------
    def _go_prev(_=None):
        _navigate_to(nav["idx"] - 1)

    def _go_next(_=None):
        _navigate_to(nav["idx"] + 1)

    prev_btn.on_click(_go_prev)
    next_btn.on_click(_go_next)

    # -- Keyboard shortcuts via ipyevents ----------------------------------
    # Attached to the outer container so any click on the GUI gives focus.
    # Server-side cooldown prevents multi-fire from DOM event bubbling.
    import time as _time

    _nav_cooldown = {"t": 0.0}
    _NAV_COOLDOWN_S = 0.35

    # -- Two-click Add Saccade (left-click on position trace) ----------------
    add_click_state = {"pending_start_rel": None}

    def _on_pos_click(trace, points, selector):
        if points is None or not hasattr(points, "xs") or len(points.xs) == 0:
            return

        click_x_rel = float(points.xs[0])

        if add_click_state["pending_start_rel"] is None:
            add_click_state["pending_start_rel"] = click_x_rel
            _set_status(
                f"Start marked at {click_x_rel:.3f} s (rel). "
                "Now <b>click</b> the saccade endpoint."
            )
            return

        start_rel = add_click_state["pending_start_rel"]
        end_rel = click_x_rel
        add_click_state["pending_start_rel"] = None

        sel_start = float(min(start_rel, end_rel)) + t0
        sel_end = float(max(start_rel, end_rel)) + t0
        if sel_end - sel_start < 0.002:
            _set_status("Start and end too close — try again.")
            return

        event = state.add_manual_saccade(sel_start, sel_end)
        _update_summary()

        final = state.get_final_events()
        new_idx = (
            int((final["time"] - event["time"]).abs().idxmin()) if len(final) > 0 else 0
        )
        _navigate_to(new_idx)
        _set_status(
            f"Added manual {event['direction']} saccade "
            f"({event['amplitude']:.1f} px, {event['duration'] * 1000:.0f} ms)"
        )

    fw.data[_IDX_POS].on_click(_on_pos_click)

    # -- Delete current event (triggered by ] key, see _on_keyup below) ----
    def _delete_current():
        events = state.get_final_events()
        if len(events) == 0:
            _set_status("No events to delete.")
            return
        idx = nav["idx"]
        ev = events.iloc[idx]
        src = ev.get("source", "auto")

        if src == "auto":
            mask = ~state._auto.index.isin(state._deleted_auto)
            orig_indices = state._auto.index[mask].tolist()
            # Find which original auto index corresponds to this merged position
            kept_auto = state.get_kept_auto_events()
            auto_times = kept_auto["time"].to_numpy(dtype=float)
            match = int(np.argmin(np.abs(auto_times - float(ev["time"]))))
            orig_idx = int(state._auto.index[mask][match])
            state.delete_auto_event(orig_idx)
        else:
            manual_df = state.get_manual_events_df()
            manual_times = manual_df["time"].to_numpy(dtype=float)
            match = int(np.argmin(np.abs(manual_times - float(ev["time"]))))
            state.delete_manual_event(match)

        _update_summary()
        new_events = state.get_final_events()
        _navigate_to(min(idx, len(new_events) - 1))
        _set_status(f"Deleted {src} event.")

    # -- Undo --------------------------------------------------------------
    def _on_undo(_=None):
        msg = state.undo()
        if msg is None:
            _set_status("Nothing to undo.")
            return
        _update_summary()
        events = state.get_final_events()
        _navigate_to(min(nav["idx"], len(events) - 1))
        _set_status(msg)

    undo_btn.on_click(_on_undo)

    def _on_done(_):
        _set_status(
            "<b style='color:green'>✅ Curation complete.</b> "
            "Run the next cell for analysis, then save."
        )

    done_btn.on_click(_on_done)

    def _go_to_event(_=None):
        n = len(state.get_final_events())
        if n == 0:
            _set_status("No events.")
            return
        idx = max(0, min(int(event_jump_int.value) - 1, n - 1))
        event_jump_int.value = idx + 1
        add_click_state["pending_start_rel"] = None
        _navigate_to(idx)

    def _go_to_gap(_=None):
        gaps = _compute_gaps()
        n = len(gaps)
        if n == 0:
            _set_status("No gaps.")
            return
        idx = max(0, min(int(gap_jump_int.value) - 1, n - 1))
        gap_jump_int.value = idx + 1
        add_click_state["pending_start_rel"] = None
        _navigate_to_gap(idx)

    go_event_btn.on_click(_go_to_event)
    go_gap_btn.on_click(_go_to_gap)

    # -- Layout assembly ---------------------------------------------------
    jump_row = widgets.HBox(
        [
            event_jump_int,
            go_event_btn,
            widgets.HTML(value="&nbsp;&nbsp;"),
            gap_jump_int,
            go_gap_btn,
        ],
        layout=widgets.Layout(
            justify_content="flex-start", align_items="center", gap="8px"
        ),
    )
    action_row = widgets.HBox(
        [undo_btn, done_btn],
        layout=widgets.Layout(
            justify_content="flex-start", align_items="center", gap="12px"
        ),
    )
    nav_row = widgets.HBox(
        [prev_btn, event_label, next_btn],
        layout=widgets.Layout(
            justify_content="center", align_items="center", gap="8px"
        ),
    )

    container = widgets.VBox(
        [
            nav_row,
            jump_row,
            shortcuts_html,
            summary_html,
            status_html,
            fw,
            action_row,
        ],
        layout=widgets.Layout(width="100%"),
    )

    # Attach keyboard listener to the outer container so clicking anywhere
    # in the GUI gives it focus.  Server-side cooldown in the handler
    # prevents the multi-fire caused by DOM event bubbling.
    key_event = DomEvent(source=container, watched_events=["keyup"])

    def _on_keyup(event):
        now = _time.time()
        if now - _nav_cooldown["t"] < _NAV_COOLDOWN_S:
            return
        _nav_cooldown["t"] = now
        key = event.get("key", "")
        if key in ("w", "W"):
            _go_next()
        elif key in ("q", "Q"):
            _go_prev()
        elif key in ("e", "E"):
            add_click_state["pending_start_rel"] = None
            _navigate_to_gap(nav["gap_idx"] - 1)
        elif key in ("r", "R"):
            add_click_state["pending_start_rel"] = None
            _navigate_to_gap(nav["gap_idx"] + 1)
        elif key == "]":
            add_click_state["pending_start_rel"] = None
            _delete_current()
        elif key in ("z", "Z"):
            _on_undo()

    key_event.on_dom_event(_on_keyup)

    return container, state
