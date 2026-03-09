import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


def select_run_root(default_runs_dir: Path) -> Path:
    """Select a MATLAB-exported sweep run folder (contains data/ and figs/)."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as e:
        raise RuntimeError('Tkinter is not available. Use --run_root to run non-interactively.') from e

    root = tk.Tk(); root.withdraw()
    run_path = filedialog.askdirectory(
        title='Select MATLAB RUN directory (contains data/ and figs/)',
        initialdir=str(default_runs_dir),
    )
    if not run_path:
        messagebox.showwarning('No Selection', 'No run directory selected. Exiting.')
        sys.exit(0)
    return Path(run_path)


def load_sweep_json(run_root: Path) -> dict:
    json_path = run_root / 'data' / 'sweep_kkt_fem_ndof.json'
    if not json_path.exists():
        raise FileNotFoundError(f'Missing {json_path}. Run the MATLAB sweep first.')
    return json.loads(json_path.read_text(encoding='utf-8'))


def _as_float_list(v) -> list[float]:
    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for x in v:
            try:
                out.append(float(x) if x is not None else float('nan'))
            except Exception:
                out.append(float('nan'))
        return out
    return []


def _as_int_list(v) -> list[int]:
    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for x in v:
            try:
                out.append(int(x) if x is not None else -1)
            except Exception:
                out.append(-1)
        return out
    return []


def _span(values: list[int]) -> str:
    vals = [int(x) for x in values if isinstance(x, int) and x > 0]
    if not vals:
        return 'n/a'
    lo = min(vals)
    hi = max(vals)
    return str(lo) if lo == hi else f'{lo}-{hi}'


def _fmt(v) -> str:
    if v is None:
        return 'n/a'
    try:
        if isinstance(v, float):
            return f'{v:g}'
        return str(v)
    except Exception:
        return 'n/a'


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description='Plot MATLAB-exported KKT-FEM sweep errors (L-domain) vs ndof.'
    )
    # examples/ -> code/ -> 2.6/ -> Projekt/
    projekt_root = Path(__file__).resolve().parents[3]
    p.add_argument(
        '--run_root',
        type=str,
        default=None,
        help='MATLAB run directory to plot (contains data/ and figs/). If omitted, a folder picker opens.',
    )
    p.add_argument('--open', action=argparse.BooleanOptionalAction, default=True,
                   help='Open the generated image in the OS viewer (default: true).')
    args = p.parse_args(argv)

    default_runs_dir = projekt_root / 'Matlab' / 'Runs'
    run_root = Path(args.run_root) if args.run_root else select_run_root(default_runs_dir)

    figs_dir = run_root / 'figs'
    figs_dir.mkdir(parents=True, exist_ok=True)

    data = load_sweep_json(run_root)

    # Prefer free DOFs (= interior nodes), fall back to total nodes for older exports
    ndof_free = _as_int_list(data.get('ndof_free'))
    ndof_total = _as_int_list(data.get('ndof_total'))
    ndof_actual = _as_int_list(data.get('ndof_actual'))

    x_source = ndof_free if ndof_free else (ndof_total if ndof_total else ndof_actual)
    x_label = 'freie DOFs (Innenknoten pro skalarer Funktion)' if ndof_free else 'Knoten gesamt (Mesh.Nodes)'
    u_mean = _as_float_list(data.get('errors', {}).get('L', {}).get('u', {}).get('mean'))
    u_max = _as_float_list(data.get('errors', {}).get('L', {}).get('u', {}).get('max'))
    y_mean = _as_float_list(data.get('errors', {}).get('L', {}).get('y', {}).get('mean'))
    y_max = _as_float_list(data.get('errors', {}).get('L', {}).get('y', {}).get('max'))

    n = min(len(x_source), len(u_mean), len(u_max), len(y_mean), len(y_max))
    if n == 0:
        raise SystemExit('No data found in sweep_kkt_fem_ndof.json')

    x = x_source[:n]
    series = {
        'u mean': u_mean[:n],
        'u max': u_max[:n],
        'y mean': y_mean[:n],
        'y max': y_max[:n],
    }

    # Filter invalid points
    valid_idx = [
        i for i in range(n)
        if x[i] is not None and x[i] > 0
        and all(
            (series[k][i] is not None)
            and math.isfinite(float(series[k][i]))
            and float(series[k][i]) > 0.0
            for k in series
        )
    ]
    x_plot = [x[i] for i in valid_idx]
    x_max = max(x_plot) if x_plot else None

    def _sel(vals: list[float]) -> list[float]:
        return [vals[i] for i in valid_idx]

    fig, ax = plt.subplots(1, 1, figsize=(12.8, 4.2))

    # Colors (not blue/orange)
    color_u = 'tab:green'
    color_y = 'tab:red'

    # Styling: keep it simple (points + solid/dashed)
    marker = 'o'
    ls_mean = '-'
    ls_max = '--'

    ax.semilogy(x_plot, _sel(series['u mean']), color=color_u, linestyle=ls_mean, marker=marker, linewidth=1.6, label='u mean')
    ax.semilogy(x_plot, _sel(series['u max']),  color=color_u, linestyle=ls_max,  marker=marker, linewidth=1.6, label='u max')
    ax.semilogy(x_plot, _sel(series['y mean']), color=color_y, linestyle=ls_mean, marker=marker, linewidth=1.6, label='y mean')
    ax.semilogy(x_plot, _sel(series['y max']),  color=color_y, linestyle=ls_max,  marker=marker, linewidth=1.6, label='y max')

    ax.set_xlabel('Anzahl innerer Knoten')
    ax.set_ylabel('Fehler')

    gridN = data.get('gridN')
    lam = data.get('lambda')
    title_parts = []
    if gridN is not None:
        title_parts.append(f'gridN={gridN}')
    if lam is not None:
        title_parts.append(f'lambda={lam:g}')
    suffix = f" ({', '.join(title_parts)})" if title_parts else ''
    ax.set_title('KKT-FEM: Fehler auf Ω_L' + suffix)

    ax.grid(True, which='major', alpha=0.35)
    ax.grid(True, which='minor', alpha=0.15)
    ax.legend(loc='best')

    # Match y-scale requirements
    ax.set_ylim(1e-5, 2.0)

    # Limit x-axis to actual data range (avoid ticks beyond last data point)
    if x_max is not None:
        ax.set_xlim(left=0, right=x_max)
    else:
        ax.set_xlim(left=0)

    # Fixed x-tick spacing (requested)
    ax.xaxis.set_major_locator(MultipleLocator(4000))

    # German thousands separator, e.g. 5000 -> 5.000
    def _fmt_thousands(xval, _pos=None):
        try:
            if not math.isfinite(float(xval)):
                return ''
            return f"{int(round(xval)):,}".replace(',', '.')
        except Exception:
            return ''

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_thousands))

    # Metadata header (minimal; hyperparameters omitted)
    run_id = data.get('run_id', run_root.name)
    header = (
        f"Run-ID: {_fmt(run_id)} | "
        f"ndof: {_span(x_plot)} | "
        f"gridN: {_fmt(gridN)} | "
        f"lambda: {_fmt(lam)}"
    )
    header_artist = fig.text(
        0.5,
        0.985,
        header,
        ha='center',
        va='top',
        fontsize=10,
        fontweight='bold',
        family='monospace',
    )

    # Keep a small gap between header and plots
    try:
        gap_px = 10
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = header_artist.get_window_extent(renderer=renderer)
        top = (bbox.y0 - gap_px) / fig.bbox.height
        top = float(min(0.95, max(0.75, top)))
    except Exception:
        top = 0.86

    out_path = figs_dir / f'sweep_kkt_fem_errors_vs_ndof_{run_id}.png'
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    if args.open:
        try:
            import platform
            import subprocess
            if platform.system() == 'Windows':
                os.startfile(str(out_path))
            elif platform.system() == 'Darwin':
                subprocess.run(['open', str(out_path)], check=False)
            else:
                subprocess.run(['xdg-open', str(out_path)], check=False)
        except Exception as e:
            print(f'Could not open image automatically: {e}')

    print(f'✓ Saved: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
