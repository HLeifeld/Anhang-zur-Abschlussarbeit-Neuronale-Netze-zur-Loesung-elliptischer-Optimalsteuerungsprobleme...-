import argparse
import json
import re
from pathlib import Path
import sys
import os
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, FuncFormatter


CFG_RE = re.compile(r"^W(?P<width>\d+)_L(?P<layer>\d+)$")


# Unified figure sizing and axes placement
# - Same pixel size for all output figures
# - Same axes position: fixed margins via subplots_adjust (no dynamic tight_layout)
# 1920x630 px at 150 dpi (matches the previous good-looking layout)
FIGSIZE = (12.8, 4.2)
SAVE_DPI = 150
SUBPLOT_ADJUST = dict(left=0.08, right=0.985, bottom=0.16, top=0.78)


def select_run_root(default_runs_dir: Path) -> Path:
    """Ask the user to select a single run root directory (e.g., 20260102_181917)."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as e:
        raise RuntimeError('Tkinter is not available. Use --run_root to run non-interactively.') from e

    root = tk.Tk(); root.withdraw()
    run_path = filedialog.askdirectory(
        title='Select RUN directory (contains data/ and figs/)',
        initialdir=str(default_runs_dir),
    )
    if not run_path:
        messagebox.showwarning('No Selection', 'No run directory selected. Exiting.')
        sys.exit(0)
    return Path(run_path)


def find_config_dirs(run_root: Path) -> list[Path]:
    """Configs are under <run_root>/data/W{W}_L{L}."""
    data_root = run_root / 'data'
    if not data_root.exists() or not data_root.is_dir():
        return []
    return [p for p in data_root.iterdir() if p.is_dir() and CFG_RE.match(p.name)]


def load_points(run_dirs: list[Path], metric: str, num_layer: int | None) -> list[tuple[int, int, float]]:
    points: list[tuple[int, int, float]] = []

    for run_dir in run_dirs:
        m = CFG_RE.match(run_dir.name)
        if not m:
            continue
        width = int(m.group('width'))
        layer = int(m.group('layer'))
        if num_layer is not None and layer != num_layer:
            continue

        summary_path = run_dir / 'summary.json'
        if not summary_path.exists():
            continue

        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
        except Exception:
            continue

        value = summary.get('final', {}).get(metric)
        if value is None:
            # Back-compat: maybe stored flat
            value = summary.get(metric)
        if value is None:
            continue

        try:
            points.append((width, layer, float(value)))
        except Exception:
            continue

    points.sort(key=lambda t: (t[1], t[0]))
    return points


def load_error_points(run_dirs: list[Path], num_layer: int | None) -> dict[str, list[tuple[int, int, float]]]:
    """Load error statistics (mean/max) over the L-domain from summary.json.

    Returns dict with keys: u_max, u_mean, y_max, y_mean -> list[(width, value)].
    """
    out: dict[str, list[tuple[int, int, float]]] = {k: [] for k in ('u_max', 'u_mean', 'y_max', 'y_mean')}

    for run_dir in run_dirs:
        m = CFG_RE.match(run_dir.name)
        if not m:
            continue
        width = int(m.group('width'))
        layer = int(m.group('layer'))
        if num_layer is not None and layer != num_layer:
            continue

        summary_path = run_dir / 'summary.json'
        if not summary_path.exists():
            continue

        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
        except Exception:
            continue

        errors = summary.get('errors', {}) if isinstance(summary, dict) else {}
        l_stats = errors.get('L', {}) if isinstance(errors, dict) else {}
        u_stats = l_stats.get('u', {}) if isinstance(l_stats, dict) else {}
        y_stats = l_stats.get('y', {}) if isinstance(l_stats, dict) else {}

        def _add(key: str, val):
            if val is None:
                return
            try:
                out[key].append((width, layer, float(val)))
            except Exception:
                return

        _add('u_max', u_stats.get('max') if isinstance(u_stats, dict) else None)
        _add('u_mean', u_stats.get('mean') if isinstance(u_stats, dict) else None)
        _add('y_max', y_stats.get('max') if isinstance(y_stats, dict) else None)
        _add('y_mean', y_stats.get('mean') if isinstance(y_stats, dict) else None)

    for k in out:
        out[k].sort(key=lambda t: (t[1], t[0]))
    return out


def _span(values: list[int]) -> str:
    if not values:
        return 'n/a'
    lo = min(values)
    hi = max(values)
    if lo == hi:
        return str(lo)
    return f'{lo}-{hi}'


def count_params(width: int, num_layer: int, dim_in: int, dim_out: int) -> int:
    """Parameter count for a fully-connected MLP with num_layer hidden layers of width.

    Architecture assumed:
        input (dim_in)
          -> hidden_1 (width)
          -> ...
          -> hidden_L (width)
          -> output (dim_out)

    Includes biases in every affine layer.

    Total params:
        dim_in*W + W
        + (L-1)*(W*W + W)
        + W*dim_out + dim_out
    """
    W = int(width)
    L = int(num_layer)
    d_in = int(dim_in)
    d_out = int(dim_out)
    if W <= 0 or L <= 0 or d_in <= 0 or d_out <= 0:
        raise ValueError('width, num_layer, dim_in, dim_out must be positive')
    return d_in * W + W + (L - 1) * (W * W + W) + W * d_out + d_out


def _metadata_header(
    summary: dict | None,
    run_root_name: str,
    metric_label: str,
    num_layer: int | None,
    width_span: str,
    layer_span: str,
) -> str:
    if not isinstance(summary, dict):
        base = f"Run: {run_root_name} | Width: {width_span} | Layer: {layer_span}"
        return base

    def _fmt(v):
        if v is None:
            return 'n/a'
        if isinstance(v, float):
            return f'{v:g}'
        return str(v)

    layer_val = _fmt(num_layer) if num_layer is not None else layer_span
    rows = [
        (f"Run-ID: {_fmt(summary.get('run_id', run_root_name))}", f"Width: {width_span}", f"Layer: {layer_val}"),
        (f"MCSizeIn: {_fmt(summary.get('mc_size_in'))}", f"MCsizeB: {_fmt(summary.get('mc_size_b'))}", f"Epoach: {_fmt(summary.get('epoach'))}"),
        (f"LearningRateStart: {_fmt(summary.get('learning_rate_start'))}", f"DecayRate: {_fmt(summary.get('decay_rate'))}", f"EpoachDecay: {_fmt(summary.get('epoach_decay'))}"),
    ]

    col_widths = [
        max(len(r[0]) for r in rows),
        max(len(r[1]) for r in rows),
        max(len(r[2]) for r in rows),
    ]

    lines: list[str] = []
    for a, b, c in rows:
        lines.append(
            "  "
            + a.ljust(col_widths[0])
            + " | "
            + b.ljust(col_widths[1])
            + " | "
            + c.ljust(col_widths[2])
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Plot sweep metric (last iteration) vs width from per-run summary.json files.')
    default_root = Path(__file__).resolve().parents[1]  # 2.1/code
    p.add_argument('--run_root', type=str, default=None,
                   help='Run directory to aggregate (contains data/ and figs/). If omitted, a folder picker opens.')
    p.add_argument('--metric', type=str, default='Loss', choices=['Loss', 'P', 'Pinn', 'Boundary'], help='Which metric to plot.')
    p.add_argument('--num_layer', type=int, default=None, help='Optional: filter by NumLayer.')
    p.add_argument('--dim_in', type=int, default=None, help='Input dimension for parameter count (default: infer from summary.json, else 2).')
    p.add_argument('--dim_out', type=int, default=None, help='Output dimension for parameter count (default: infer from summary.json, else 1).')
    p.add_argument('--yscale', type=str, default='log', choices=['linear', 'log'], help='Y scale for plots (default: log).')
    p.add_argument('--open', action=argparse.BooleanOptionalAction, default=True,
                   help='Open the generated image in the OS viewer (default: true).')
    args = p.parse_args(argv)

    default_runs_dir = default_root / 'Run'
    run_root = Path(args.run_root) if args.run_root else select_run_root(default_runs_dir)

    figs_dir = run_root / 'figs'
    figs_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_config_dirs(run_root)
    points = load_points(run_dirs, args.metric, args.num_layer)
    err_points = load_error_points(run_dirs, args.num_layer)

    if not points:
        raise SystemExit(f'No data points found in {run_root}. Did you run ex1.py and generate summary.json files?')

    widths = [w for w, _, _ in points]
    layers = [l for _, l, _ in points]
    values = [v for _, _, v in points]

    width_span = _span(widths)
    layer_span = _span(layers) if args.num_layer is None else str(args.num_layer)

    metric_label = {
        'Loss': 'Loss',
        'P': 'Loss P',
        'Pinn': 'Loss PINN',
        'Boundary': 'Loss Boundary',
    }.get(args.metric, args.metric)

    # Infer dim_in/dim_out from summary.json when possible
    def _infer_int(s: dict | None, keys: list[str]) -> int | None:
        if not isinstance(s, dict):
            return None
        for k in keys:
            v = s.get(k)
            if isinstance(v, int) and v > 0:
                return v
        return None

    # Styling rules:
    # - u and y: same color per variable
    # - mean/max: same linestyle across u/y
    # - Loss: distinct but sensible
    color_u = 'tab:blue'
    color_y = 'tab:orange'
    ls_mean = '-'
    ls_max = '--'

    # Use the first available summary.json for inference
    first_summary = None
    layer_values: set[int] = set()
    width_values: set[int] = set()
    for d in run_dirs:
        m = CFG_RE.match(d.name)
        if m:
            width_values.add(int(m.group('width')))
            layer_values.add(int(m.group('layer')))

        sp = d / 'summary.json'
        if not sp.exists():
            continue
        try:
            s = json.loads(sp.read_text(encoding='utf-8'))
        except Exception:
            continue
        if first_summary is None:
            first_summary = s

    inferred_dim_in = _infer_int(first_summary, ['dim_in', 'input_dim', 'dim_input', 'd_in'])
    inferred_dim_out = _infer_int(first_summary, ['dim_out', 'output_dim', 'dim_output', 'd_out'])
    dim_in = args.dim_in if args.dim_in is not None else (inferred_dim_in if inferred_dim_in is not None else 2)
    dim_out = args.dim_out if args.dim_out is not None else (inferred_dim_out if inferred_dim_out is not None else 1)

    # Map (width, layer) -> param_count
    param_map: dict[tuple[int, int], int] = {}
    for w, l in {(w, l) for (w, l, _) in points}:
        param_map[(w, l)] = count_params(w, l, dim_in, dim_out)

    # Use a shared x-axis range for all "vs #Parameter" plots.
    # Round up to the next 1000 so the axis ends on a clean tick.
    params_xmax = 0
    if param_map:
        params_xmax = max(int(v) for v in param_map.values())
    params_xmax = int(math.ceil(params_xmax / 1000.0) * 1000) if params_xmax > 0 else 1

    def _open_in_viewer(path: Path) -> None:
        try:
            import platform
            import subprocess
            if platform.system() == 'Windows':
                os.startfile(str(path))
            elif platform.system() == 'Darwin':
                subprocess.run(['open', str(path)], check=False)
            else:
                subprocess.run(['xdg-open', str(path)], check=False)
        except Exception as e:
            print(f'Could not open image automatically: {e}')

    def _plot_series(ax, x_loss: list[float], y_loss: list[float], x_map_fn):
        # Loss
        ax.plot(x_loss, y_loss, color='black', linestyle='-.', marker='o', linewidth=1.9, label=metric_label)

        def _plot_err(key: str, label: str, color: str, linestyle: str):
            series = err_points.get(key) or []
            if not series:
                return
            xy: list[tuple[float, float]] = []
            for w, l, v in series:
                x = x_map_fn(w, l)
                if x is None:
                    continue
                xy.append((float(x), float(v)))
            if not xy:
                return
            xy.sort(key=lambda t: t[0])
            ax.plot(
                [x for x, _ in xy],
                [y for _, y in xy],
                color=color,
                linestyle=linestyle,
                marker='o',
                linewidth=1.6,
                label=label,
            )

        _plot_err('u_mean', 'u mean', color_u, ls_mean)
        _plot_err('u_max', 'u max', color_u, ls_max)
        _plot_err('y_mean', 'y mean', color_y, ls_mean)
        _plot_err('y_max', 'y max', color_y, ls_max)

    def _finalize_and_save(fig, ax, header_text: str, out_path: Path, x_locator: str):
        ax.set_ylabel('Wert (letzte Iteration / Fehler)')
        if args.yscale == 'log':
            ax.set_yscale('log')

        # Fixed y-range as requested
        ax.set_ylim(1e-4, 1.0e0)

        # X-axis must start at 0
        ax.set_xlim(left=0)

        if x_locator == 'params':
            # Force identical x-scale across all parameter plots
            ax.set_xlim(left=0, right=float(params_xmax))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _pos: f"{int(round(x)):,}".replace(',', '.'))
            )
        elif x_locator == 'width':
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.tick_params(axis='x', which='major', length=6)
            ax.tick_params(axis='x', which='minor', length=3)

        ax.grid(True, which='major', alpha=0.35)
        ax.grid(True, which='minor', alpha=0.15)
        ax.legend(loc='upper right')

        fig.text(
            0.5,
            0.985,
            header_text,
            ha='center',
            va='top',
            fontsize=9,
            fontweight='bold',
            family='monospace',
        )

        # Fixed axes placement for consistent comparisons across plots
        fig.subplots_adjust(**SUBPLOT_ADJUST)
        print(f'Saving: {out_path}', flush=True)
        try:
            fig.savefig(out_path, dpi=SAVE_DPI)
        except KeyboardInterrupt:
            print(
                'Aborted while rendering/saving the figure. '\
                'If this is the first Matplotlib run on this machine, font-cache creation can take a while. '\
                'Try again and let it finish, or set MPLCONFIGDIR to a fast local folder (e.g. %TEMP%\\mplconfig).',
                flush=True,
            )
            raise
        plt.close(fig)

    out_paths: list[Path] = []


    # 1) Plot vs parameter count (all layers unless --num_layer was used)
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.set_xlabel('Anzahl trainierbarer Parameter')
    #ax.set_title(f'{metric_label} + Fehler (L-Gebiet) vs #Parameter')

    loss_xy = sorted(((param_map[(w, l)], v) for (w, l, v) in points), key=lambda t: t[0])
    _plot_series(
        ax,
        [float(x) for x, _ in loss_xy],
        [float(y) for _, y in loss_xy],
        x_map_fn=lambda w, l: float(param_map.get((w, l))) if (w, l) in param_map else None,
    )

    inferred_layer = args.num_layer
    if inferred_layer is None and len(layer_values) == 1:
        inferred_layer = next(iter(layer_values))

    header_text = _metadata_header(
        first_summary,
        run_root.name,
        metric_label,
        inferred_layer,
        width_span=_span(sorted(width_values)) if width_values else width_span,
        layer_span=_span(sorted(layer_values)) if layer_values else layer_span,
    )
    out_path_params = figs_dir / f'sweep_{args.metric}_and_errors_vs_params_{run_root.name}.png'
    _finalize_and_save(fig, ax, header_text, out_path_params, x_locator='params')
    out_paths.append(out_path_params)

    # 2) Plot vs width for each layer (2.5 behavior), or only for --num_layer
    if args.num_layer is not None:
        layers_to_plot = [args.num_layer]
    else:
        layers_to_plot = sorted(layer_values)

    # 2a) Plot vs parameter count for each layer (additional plots)
    # If --num_layer is set, the global params plot is already layer-filtered, so avoid duplicates.
    if args.num_layer is None:
        for layer in layers_to_plot:
            layer_param_points = [(param_map[(w, l)], v) for (w, l, v) in points if l == layer and (w, l) in param_map]
            if not layer_param_points:
                continue

            layer_param_points.sort(key=lambda t: t[0])
            xp = [float(x) for x, _ in layer_param_points]
            yv = [float(v) for _, v in layer_param_points]

            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
            ax.set_xlabel('Anzahl Parameter (NN)')
            #ax.set_title(f'{metric_label} + Fehler (L-Gebiet) vs #Parameter (Layer {layer})')

            _plot_series(
                ax,
                xp,
                yv,
                x_map_fn=lambda w, l, _layer=layer: float(param_map.get((w, l))) if l == _layer and (w, l) in param_map else None,
            )

            layer_widths = [w for (w, l, _) in points if l == layer]
            header_text = _metadata_header(
                first_summary,
                run_root.name,
                metric_label,
                num_layer=layer,
                width_span=_span(sorted([int(w) for w in layer_widths])) if layer_widths else 'n/a',
                layer_span=str(layer),
            )
            out_path_lp = figs_dir / f'sweep_{args.metric}_and_errors_vs_params_L{layer}_{run_root.name}.png'
            _finalize_and_save(fig, ax, header_text, out_path_lp, x_locator='params')
            out_paths.append(out_path_lp)

    for layer in layers_to_plot:
        layer_points = [(w, v) for (w, l, v) in points if l == layer]
        if not layer_points:
            continue

        layer_points.sort(key=lambda t: t[0])
        xw = [float(w) for w, _ in layer_points]
        yv = [float(v) for _, v in layer_points]

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        ax.set_xlabel('Breite (Neuronen pro Layer)')
        #ax.set_title(f'{metric_label} + Fehler (L-Gebiet) vs Breite (Layer {layer})')

        _plot_series(
            ax,
            xw,
            yv,
            x_map_fn=lambda w, l, _layer=layer: float(w) if l == _layer else None,
        )

        header_text = _metadata_header(
            first_summary,
            run_root.name,
            metric_label,
            num_layer=layer,
            width_span=_span([int(w) for w, _ in layer_points]),
            layer_span=str(layer),
        )
        out_path_w = figs_dir / f'sweep_{args.metric}_and_errors_vs_width_L{layer}_{run_root.name}.png'
        _finalize_and_save(fig, ax, header_text, out_path_w, x_locator='width')
        out_paths.append(out_path_w)

    if args.open and out_paths:
        _open_in_viewer(out_paths[0])

    for pth in out_paths:
        print(f'✓ Saved: {pth}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
