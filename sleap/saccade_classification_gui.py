"""
Interactive saccade classification GUI for Jupyter notebooks.

Provides a Plotly FigureWidget + ipywidgets interface for:
- Reviewing each saccade in a 3-panel windowed view
  (X position / Motor_Velocity / Velocity_0Y, ±5 s around the event)
- Overriding the automatic class assigned by Cell 4
- Navigating with W/Q (next/prev event), J/K (next/prev saccade bout)
- Keyboard class shortcuts: 1=look_around 2=look_ahead 3=recentering
                            4=anomalous   5=no_saccade
- Saving classified events to CSV + metadata JSON

Usage (from notebook cell)::

    from sleap.saccade_classification_gui import build_classification_gui

    classification_widget, classification_state = build_classification_gui(
        saccade_events_df=saccade_events_df,
        saccade_snippets_df=saccade_snippets_df,
        turning_df=turning_df,
        t0=t0,
    )
    display(classification_widget)

    # Later cell — save:
    # classification_state.save(save_dir, metadata, metadata_path, classification_params)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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
# Constants
# ---------------------------------------------------------------------------

ALL_CLASSES = ("look_around", "look_ahead", "recentering", "anomalous", "no_saccade")

_CLASS_COLOURS = {
    "look_around": "steelblue",
    "look_ahead":  "gold",
    "recentering": "limegreen",
    "anomalous":   "crimson",
    "no_saccade":  "lightgray",
}

_KEY_TO_CLASS = {
    "1": "look_around",
    "2": "look_ahead",
    "3": "recentering",
    "4": "anomalous",
    "5": "no_saccade",
}

# Fixed trace indices inside the 3-row FigureWidget
_IDX_SNIP   = 0   # row 1 — X_raw snippet
_IDX_MARKER = 1   # row 1 — saccade peak marker
_IDX_MOTOR  = 2   # row 2 — Motor_Velocity
_IDX_VEL    = 3   # row 3 — Velocity_0Y

HALF_WINDOW_S = 5.0   # ±5 s around each event (matches snippet radius)


# ---------------------------------------------------------------------------
# ClassificationState
# ---------------------------------------------------------------------------

class ClassificationState:
    """Holds automatic classifications and manual overrides.

    Parameters
    ----------
    saccade_events_df : DataFrame
        Output of Cell 4 — must contain ``saccade_class``, ``saccade_bout_id``,
        ``saccade_bout_position``, ``isi_before_s``, ``isi_after_s``, ``TNT_direction``.
        Index must be ``aeon_time``.
    classification_params : dict
        Parameter snapshot (min_saccade_bout_size, max_isi_s) for metadata persistence.
    """

    def __init__(
        self,
        saccade_events_df: pd.DataFrame,
        classification_params: dict,
    ):
        self._df = saccade_events_df.sort_index().copy()
        self._params = classification_params
        # dict: positional integer index → override class string
        self._overrides: dict[int, str] = {}

    # -- public read --------------------------------------------------------

    def get_class(self, pos: int) -> str:
        if pos in self._overrides:
            return self._overrides[pos]
        return str(self._df["saccade_class"].iloc[pos])

    def set_class(self, pos: int, new_class: str) -> None:
        """Record a manual override; remove it if it matches the auto class."""
        if new_class == str(self._df["saccade_class"].iloc[pos]):
            self._overrides.pop(pos, None)
        else:
            self._overrides[pos] = new_class

    def n_events(self) -> int:
        return len(self._df)

    def n_saccade_bouts(self) -> int:
        if "saccade_bout_id" not in self._df.columns:
            return 0
        return (
            int(self._df["saccade_bout_id"].max()) + 1
            if (self._df["saccade_bout_id"] >= 0).any()
            else 0
        )

    def saccade_bout_ids(self) -> np.ndarray:
        if "saccade_bout_id" not in self._df.columns:
            return np.array([], dtype=int)
        return self._df["saccade_bout_id"].to_numpy(dtype=int)

    def summary_text(self) -> str:
        counts: dict[str, int] = {c: 0 for c in ALL_CLASSES}
        for pos in range(len(self._df)):
            counts[self.get_class(pos)] += 1
        parts = [f"{c}: {counts[c]}" for c in ALL_CLASSES if counts[c] > 0]
        n_ov = len(self._overrides)
        return " | ".join(parts) + f"  ({n_ov} manual override{'s' if n_ov != 1 else ''})"

    def get_final_df(self) -> pd.DataFrame:
        """Return a copy of the events DataFrame with all overrides applied."""
        df = self._df.copy()
        for pos, cls in self._overrides.items():
            df.iloc[pos, df.columns.get_loc("saccade_class")] = cls
        return df

    # -- first event of each saccade bout (for J/K navigation) ------------

    def saccade_bout_first_positions(self) -> list[int]:
        """Return the positional indices of the first saccade in each saccade bout."""
        if "saccade_bout_id" not in self._df.columns:
            return []
        saccade_bout_ids = self._df["saccade_bout_id"].to_numpy(dtype=int)
        positions = []
        seen: set[int] = set()
        for pos, bid in enumerate(saccade_bout_ids):
            if bid >= 0 and bid not in seen:
                seen.add(bid)
                positions.append(pos)
        return positions

    # -- persistence --------------------------------------------------------

    def save(
        self,
        save_dir: Path | str,
        metadata: dict,
        metadata_path: Path | str,
        classification_params: dict,
    ) -> str:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        final_df = self.get_final_df()

        # Serialise datetime columns to ISO strings for CSV
        _dt_cols = [c for c in final_df.columns if pd.api.types.is_datetime64_any_dtype(final_df[c])]
        final_to_save = final_df.copy()
        for col in _dt_cols:
            final_to_save[col] = final_to_save[col].apply(
                lambda v: v.isoformat() if pd.notna(v) else ""
            )
        # Also handle index (aeon_time)
        if pd.api.types.is_datetime64_any_dtype(final_to_save.index):
            final_to_save.index = final_to_save.index.map(
                lambda v: v.isoformat() if pd.notna(v) else ""
            )
            final_to_save.index.name = "aeon_time"

        out_path = save_dir / "classified_saccade_events.csv"
        final_to_save.to_csv(out_path)

        # Update metadata
        metadata["classification_parameters"] = classification_params
        metadata["classification_summary"] = {
            "total_events": int(len(final_df)),
            "n_saccade_bouts": self.n_saccade_bouts(),
            "n_manual_overrides": len(self._overrides),
            "class_counts": {
                c: int((final_df["saccade_class"] == c).sum()) for c in ALL_CLASSES
            },
            "classification_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = Path(metadata_path)
        with open(metadata_path, "w") as fh:
            json.dump(metadata, fh, indent=2)

        return (
            f"Saved {len(final_df)} classified events to {out_path.name} "
            f"and updated {metadata_path.name}"
        )


# ---------------------------------------------------------------------------
# FigureWidget builder
# ---------------------------------------------------------------------------

def _build_figure() -> go.FigureWidget:
    """3-row FigureWidget with empty traces (populated by _navigate_to)."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("X position (snippet)", "Motor velocity", "Velocity_0Y"),
    )

    # Row 1: X_raw snippet
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="lines", name="X_raw",
                   line=dict(width=1.2, color="steelblue")),
        row=1, col=1,
    )
    # Row 1: peak marker
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="markers", name="peak",
                   marker=dict(size=10, symbol="circle", color="gold",
                               line=dict(color="black", width=1.5))),
        row=1, col=1,
    )
    # Row 2: Motor_Velocity
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="lines", name="Motor_Velocity",
                   line=dict(width=1, color="firebrick")),
        row=2, col=1,
    )
    # Row 3: Velocity_0Y
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="lines", name="Velocity_0Y",
                   line=dict(width=1, color="darkorange")),
        row=3, col=1,
    )

    fig.update_layout(
        template="plotly_white",
        height=700,
        dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(title_text="Relative time (s)", row=3, col=1)
    fig.update_yaxes(title_text="X position (px)", row=1, col=1)
    fig.update_yaxes(title_text="Motor vel. (deg/s)", row=2, col=1)
    fig.update_yaxes(title_text="Velocity_0Y (deg/s)", row=3, col=1)

    return go.FigureWidget(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_classification_gui(
    saccade_events_df: pd.DataFrame,
    saccade_snippets_df: pd.DataFrame,
    turning_df: pd.DataFrame,
    t0: pd.Timestamp,
    classification_params: dict | None = None,
) -> tuple[widgets.VBox, ClassificationState]:
    """Build and return the classification widget and state.

    Parameters
    ----------
    saccade_events_df : DataFrame
        Cell 4 output — indexed by aeon_time, must contain ``saccade_class``,
        ``saccade_bout_id``, ``saccade_bout_position``, ``isi_before_s``, ``isi_after_s``.
    saccade_snippets_df : DataFrame
        Cell 1 output — indexed by aeon_time, columns ``event_idx``,
        ``time_rel``, ``X_raw``.
    turning_df : DataFrame
        Cell 1 / Cell 3 output — DatetimeIndex, columns ``Motor_Velocity``,
        ``Velocity_0Y``, optionally ``is_platform_rotating``,
        ``is_animal_turning``.
    t0 : pd.Timestamp
        Global reference time (earliest sample) used for relative-time axes.
    classification_params : dict, optional
        Parameter snapshot for metadata persistence (e.g. min_saccade_bout_size,
        max_isi_s). Defaults to empty dict.
    """
    if not _WIDGETS_OK:
        raise ImportError(
            "ipywidgets is required for the classification GUI. "
            "Install with: pip install ipywidgets"
        )

    from ipyevents import Event as DomEvent
    import time as _time

    if classification_params is None:
        classification_params = {}

    state = ClassificationState(saccade_events_df, classification_params)
    n_events = state.n_events()

    # Pre-build per-event snippet arrays for fast rendering
    # snippet_data[pos] = (t_rel_arr, x_arr) relative to t0
    _ev_sorted = state._df  # already sorted
    _snip_by_pos: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pos in range(n_events):
        ev_idx = pos  # event_idx in snippets matches positional order in sorted events
        snip = saccade_snippets_df[saccade_snippets_df["event_idx"] == ev_idx]
        if len(snip) > 0:
            _snip_by_pos[pos] = (
                (snip.index - t0).total_seconds().to_numpy(),
                snip["X_raw"].to_numpy(dtype=float),
            )
        else:
            _snip_by_pos[pos] = (np.array([]), np.array([]))

    # Pre-build turning arrays
    _mot_t_rel = (turning_df.index - t0).total_seconds().to_numpy()
    _mot_vals = turning_df["Motor_Velocity"].to_numpy(dtype=float)
    _vel_vals = turning_df["Velocity_0Y"].to_numpy(dtype=float)
    _rot_state = (
        turning_df["is_platform_rotating"].to_numpy(dtype=bool)
        if "is_platform_rotating" in turning_df.columns
        else None
    )
    _turn_state = (
        turning_df["is_animal_turning"].to_numpy(dtype=bool)
        if "is_animal_turning" in turning_df.columns
        else None
    )

    nav = {"pos": 0}
    _cooldown = {"t": 0.0}
    _COOLDOWN_S = 0.25

    # -- Build FigureWidget ------------------------------------------------
    fw = _build_figure()

    # -- Widgets -----------------------------------------------------------
    prev_btn = widgets.Button(
        description="< Prev (Q)", button_style="info",
        layout=widgets.Layout(width="110px"),
    )
    next_btn = widgets.Button(
        description="Next (W) >", button_style="info",
        layout=widgets.Layout(width="110px"),
    )
    prev_bout_btn = widgets.Button(
        description="< Bout (J)", button_style="warning",
        layout=widgets.Layout(width="110px"),
    )
    next_bout_btn = widgets.Button(
        description="Bout (K) >", button_style="warning",
        layout=widgets.Layout(width="110px"),
    )
    event_label = widgets.HTML(value="")
    info_html = widgets.HTML(value="")
    summary_html = widgets.HTML(value="")
    status_html = widgets.HTML(value="")

    event_jump_int = widgets.BoundedIntText(
        value=1, min=1, max=max(1, n_events),
        description="Event #:",
        layout=widgets.Layout(width="160px"),
        style={"description_width": "60px"},
    )
    go_event_btn = widgets.Button(
        description="Go", button_style="info",
        layout=widgets.Layout(width="50px"),
    )

    # Classification buttons (1–5)
    _cls_buttons = {}
    for _key, _cls in _KEY_TO_CLASS.items():
        _btn = widgets.Button(
            description=f"[{_key}] {_cls}",
            layout=widgets.Layout(width="150px"),
            style={"button_color": _CLASS_COLOURS[_cls]},
        )
        _cls_buttons[_cls] = _btn

    done_btn = widgets.Button(
        description="Done", icon="check", button_style="success",
        layout=widgets.Layout(width="100px"),
    )

    shortcuts_html = widgets.HTML(
        value=(
            "<span style='color:gray; font-size:0.9em'>"
            "<b>W</b>/<b>Q</b> next/prev event &nbsp;|&nbsp; "
            "<b>J</b>/<b>K</b> prev/next bout &nbsp;|&nbsp; "
            "<b>1</b>–<b>5</b> assign class &nbsp;|&nbsp; "
            "<b>Jump:</b> Event # + Go"
            "</span>"
        )
    )

    def _update_summary():
        summary_html.value = f"<b>{state.summary_text()}</b>"

    def _set_status(msg: str):
        status_html.value = f"<i>{msg}</i>"

    # -- Core: render window around event at pos ---------------------------
    def _navigate_to(pos: int):
        nonlocal nav
        pos = int(np.clip(pos, 0, n_events - 1))
        nav["pos"] = pos

        ev = _ev_sorted.iloc[pos]
        ev_t_rel = float((ev.name - t0).total_seconds())  # peak time, relative
        win_lo = ev_t_rel - HALF_WINDOW_S
        win_hi = ev_t_rel + HALF_WINDOW_S

        # Snippet
        snip_t, snip_x = _snip_by_pos.get(pos, (np.array([]), np.array([])))

        # Motor / Vel window
        _mask = (_mot_t_rel >= win_lo) & (_mot_t_rel <= win_hi)
        mot_t_win = _mot_t_rel[_mask]
        mot_v_win = _mot_vals[_mask]
        vel_v_win = _vel_vals[_mask]

        # Determine current class and colour
        cls = state.get_class(pos)
        colour = _CLASS_COLOURS.get(cls, "steelblue")

        # Saccade bout shading shapes
        shapes = []
        bout_id = int(ev["saccade_bout_id"]) if "saccade_bout_id" in ev.index else -1
        if bout_id >= 0:
            bout_mask = _ev_sorted["saccade_bout_id"] == bout_id
            bout_rows = _ev_sorted[bout_mask]
            _aeon_start_col = "aeon_start_time"
            _aeon_end_col = "aeon_end_time"
            if _aeon_start_col in bout_rows.columns and _aeon_end_col in bout_rows.columns:
                _bs = bout_rows[_aeon_start_col].dropna()
                _be = bout_rows[_aeon_end_col].dropna()
                if len(_bs) > 0 and len(_be) > 0:
                    _bout_t0 = float((_bs.min() - t0).total_seconds())
                    _bout_t1 = float((_be.max() - t0).total_seconds())
                    for _row_ref in ("x", "x2", "x3"):
                        _yref = f"y{_row_ref[1:]} domain" if _row_ref != "x" else "y domain"
                        shapes.append(dict(
                            type="rect", xref=_row_ref, yref=_yref,
                            x0=_bout_t0, x1=_bout_t1, y0=0, y1=1,
                            fillcolor="rgba(200,200,200,0.18)",
                            line=dict(color="rgba(100,100,100,0.30)", width=1, dash="dot"),
                        ))

        # Rotation / turning shading (rows 2 and 3)
        def _state_run_shapes(arr, xref, yref, fillcolor):
            if arr is None:
                return []
            _win_mask = (_mot_t_rel >= win_lo) & (_mot_t_rel <= win_hi)
            _t_win = _mot_t_rel[_win_mask]
            _s_win = arr[_win_mask]
            if len(_t_win) < 2:
                return []
            edges = np.diff(_s_win.astype(np.int8), prepend=0)
            starts = _t_win[edges == 1]
            ends = _t_win[edges == -1]
            if len(starts) > len(ends):
                ends = np.append(ends, _t_win[-1])
            return [dict(
                type="rect", xref=xref, yref=yref,
                x0=float(s), x1=float(e), y0=0, y1=1,
                fillcolor=fillcolor, line_width=0,
            ) for s, e in zip(starts, ends)]

        shapes += _state_run_shapes(_rot_state,  "x2", "y2 domain", "rgba(178,34,34,0.15)")
        shapes += _state_run_shapes(_turn_state, "x2", "y2 domain", "rgba(255,140,0,0.12)")
        shapes += _state_run_shapes(_rot_state,  "x3", "y3 domain", "rgba(178,34,34,0.15)")
        shapes += _state_run_shapes(_turn_state, "x3", "y3 domain", "rgba(255,140,0,0.12)")

        # Update figure
        with fw.batch_update():
            fw.data[_IDX_SNIP].x = snip_t
            fw.data[_IDX_SNIP].y = snip_x
            fw.data[_IDX_SNIP].line.color = colour
            fw.data[_IDX_SNIP].name = cls
            fw.data[_IDX_MARKER].x = [ev_t_rel]
            fw.data[_IDX_MARKER].y = (
                [float(np.interp(ev_t_rel, snip_t, snip_x))]
                if len(snip_t) > 0 else [0.0]
            )
            fw.data[_IDX_MARKER].marker.color = colour
            fw.data[_IDX_MOTOR].x = mot_t_win
            fw.data[_IDX_MOTOR].y = mot_v_win
            fw.data[_IDX_VEL].x = mot_t_win
            fw.data[_IDX_VEL].y = vel_v_win
            fw.layout.shapes = shapes
            fw.layout.xaxis.range = [win_lo, win_hi]
            fw.layout.xaxis2.range = [win_lo, win_hi]
            fw.layout.xaxis3.range = [win_lo, win_hi]

        # Info label
        saccade_bout_pos = (
            int(ev["saccade_bout_position"]) if "saccade_bout_position" in ev.index else -1
        )
        isi_b = ev.get("isi_before_s", float("nan"))
        isi_a = ev.get("isi_after_s", float("nan"))
        tnt = ev.get("TNT_direction", "")
        src = ev.get("source", "auto")
        is_manual = pos in state._overrides
        cls_label = (
            f"<b style='color:{colour}'>{cls}</b>"
            + (" <span style='color:gray'>(manual)</span>" if is_manual else "")
        )
        event_label.value = (
            f"<b>Event {pos + 1}/{n_events}</b> &nbsp;|&nbsp; {cls_label} &nbsp;|&nbsp; "
            f"TNT: {tnt} &nbsp;|&nbsp; src: {src}"
        )
        info_html.value = (
            f"<span style='font-size:0.9em; color:gray'>"
            f"saccade_bout_id={bout_id} &nbsp; saccade_bout_pos={saccade_bout_pos} &nbsp;|&nbsp; "
            f"ISI_before={_fmt_isi(isi_b)} &nbsp; ISI_after={_fmt_isi(isi_a)}"
            f"</span>"
        )
        event_jump_int.value = pos + 1
        _update_summary()

    def _fmt_isi(v) -> str:
        try:
            f = float(v)
            return "—" if not np.isfinite(f) else f"{f:.2f} s"
        except (TypeError, ValueError):
            return "—"

    # -- Saccade bout navigation helpers -----------------------------------
    _saccade_bout_first_positions = state.saccade_bout_first_positions()

    def _go_prev_saccade_bout(_=None):
        cur = nav["pos"]
        candidates = [p for p in _saccade_bout_first_positions if p < cur]
        _navigate_to(candidates[-1] if candidates else cur)

    def _go_next_saccade_bout(_=None):
        cur = nav["pos"]
        candidates = [p for p in _saccade_bout_first_positions if p > cur]
        _navigate_to(candidates[0] if candidates else cur)

    # -- Navigation callbacks ---------------------------------------------
    def _go_prev(_=None):
        _navigate_to(nav["pos"] - 1)

    def _go_next(_=None):
        _navigate_to(nav["pos"] + 1)

    def _go_to_event(_=None):
        _navigate_to(max(0, int(event_jump_int.value) - 1))

    prev_btn.on_click(_go_prev)
    next_btn.on_click(_go_next)
    prev_bout_btn.on_click(_go_prev_saccade_bout)
    next_bout_btn.on_click(_go_next_saccade_bout)
    go_event_btn.on_click(_go_to_event)
    done_btn.on_click(lambda _: _set_status(
        "<b style='color:green'>✅ Classification complete. Run Cell 7 to save.</b>"
    ))

    # -- Classification button callbacks ----------------------------------
    def _make_cls_callback(cls):
        def _cb(_=None):
            state.set_class(nav["pos"], cls)
            _navigate_to(nav["pos"])
        return _cb

    for _cls, _btn in _cls_buttons.items():
        _btn.on_click(_make_cls_callback(_cls))

    # -- Keyboard shortcuts via ipyevents ---------------------------------
    def _on_keyup(event):
        now = _time.time()
        if now - _cooldown["t"] < _COOLDOWN_S:
            return
        _cooldown["t"] = now
        key = event.get("key", "")
        if key in ("w", "W"):
            _go_next()
        elif key in ("q", "Q"):
            _go_prev()
        elif key in ("j", "J"):
            _go_prev_saccade_bout()
        elif key in ("k", "K"):
            _go_next_saccade_bout()
        elif key in _KEY_TO_CLASS:
            state.set_class(nav["pos"], _KEY_TO_CLASS[key])
            _navigate_to(nav["pos"])

    # -- Layout assembly ---------------------------------------------------
    nav_row = widgets.HBox(
        [prev_btn, event_label, next_btn],
        layout=widgets.Layout(justify_content="center", align_items="center", gap="8px"),
    )
    bout_nav_row = widgets.HBox(
        [prev_bout_btn, widgets.HTML(value="<span style='color:gray'>J/K: jump to bout</span>"), next_bout_btn],
        layout=widgets.Layout(justify_content="center", align_items="center", gap="8px"),
    )
    jump_row = widgets.HBox(
        [event_jump_int, go_event_btn],
        layout=widgets.Layout(justify_content="flex-start", align_items="center", gap="8px"),
    )
    cls_row = widgets.HBox(
        list(_cls_buttons.values()) + [done_btn],
        layout=widgets.Layout(
            justify_content="flex-start", align_items="center",
            gap="6px", flex_wrap="wrap",
        ),
    )

    container = widgets.VBox(
        [
            nav_row,
            bout_nav_row,
            jump_row,
            shortcuts_html,
            info_html,
            summary_html,
            status_html,
            fw,
            cls_row,
        ],
        layout=widgets.Layout(width="100%"),
    )

    key_event = DomEvent(source=container, watched_events=["keyup"])
    key_event.on_dom_event(_on_keyup)

    _navigate_to(0)
    return container, state
