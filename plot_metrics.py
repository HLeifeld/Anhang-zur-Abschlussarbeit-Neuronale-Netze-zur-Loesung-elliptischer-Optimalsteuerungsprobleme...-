import sys
import os
from pathlib import Path

# Force CPU BEFORE any JAX imports
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import argparse
import jax
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp


def select_run_dir(project_root: Path) -> Path:
    """Ask the user to select the RUN directory (e.g., 20251111_104154)."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as e:
        raise RuntimeError(
            'Tkinter is not available. Use --run_dir to run non-interactively.'
        ) from e

    root = tk.Tk(); root.withdraw()
    run_path = filedialog.askdirectory(
        title='Select RUN directory (e.g., 20251111_104154)',
        initialdir=str(project_root / 'Run')
    )
    if not run_path:
        messagebox.showwarning('No Selection', 'No run directory selected. Exiting.')
        sys.exit(0)
    return Path(run_path)


def restore_ckpt(path: Path, target_tree: dict | None = None):
    """Robustly restore an Orbax checkpoint (works across orbax versions).

    If a target tree is provided, it will be used to force restore onto CPU numpy arrays
    and avoid device-specific shardings from the original checkpoint (e.g., cuda:0).
    """
    checkpointer = ocp.StandardCheckpointer()
    if target_tree is not None:
        # Try common Orbax signatures with a provided target/item
        for kwargs in (
            {"target": target_tree},
            {"item": target_tree},
        ):
            try:
                return checkpointer.restore(str(path), **kwargs)
            except TypeError:
                continue
            except Exception:
                continue
        # As a last resort, fall through to no-target restore
    # No/failed target: attempt generic restore (may warn and can fail if GPU sharding present)
    try:
        return checkpointer.restore(str(path))
    except TypeError:
        try:
            return checkpointer.restore(str(path), target=None)
        except TypeError:
            return checkpointer.restore(str(path), item=None)


def load_series(run_dir: Path):
    """Load Loss/P/Pinn/Boundary arrays if present in the run directory.
    Returns a dict name -> 1D numpy array, preserving epoch order.
    """
    series = {}
    for name in ['Loss', 'P', 'Pinn', 'Boundary']:
        p = run_dir / name
        if p.exists() and p.is_dir():
            # Provide a minimal target tree to force CPU/NumPy restore and avoid cuda shardings
            target = {name: onp.array([], dtype=onp.float32)}
            data = restore_ckpt(p, target_tree=target)
            if isinstance(data, dict) and name in data:
                arr = data[name]
            elif isinstance(data, dict) and 'value' in data:
                # very rare alternate format
                arr = data['value']
            else:
                # try using the dict as-is if it only has one item
                if isinstance(data, dict) and len(data) == 1:
                    arr = next(iter(data.values()))
                else:
                    arr = None
            if arr is not None:
                try:
                    series[name] = onp.asarray(arr).reshape(-1)
                except Exception:
                    # best effort conversion
                    series[name] = onp.array(arr).reshape(-1)
    return series


def make_plot(
    series: dict,
    figs_root: Path,
    title: str,
    out_path: Path,
    *,
    y_lim: tuple[float, float] | None = None,
):
    """Plot all available series into a single PNG (log scale only)."""
    if not series:
        raise FileNotFoundError('No Loss/P/Pinn/Boundary checkpoints found in the selected directory.')

    # Prepare epochs based on the longest series
    max_len = max(len(v) for v in series.values())
    epochs = onp.arange(max_len)

    # Create figure with one subplot: log scale only
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))

    # Mapping from checkpoint names to display names
    label_map = {
        'Loss': 'Loss All',
        'P': 'Loss P',
        'Pinn': 'Loss Pinn',
        'Boundary': 'Loss Boundary'
    }
    
    # Colors for consistency
    color_map = {
        'Loss All': '#1f77b4',  # blue
        # 'J': '#ff7f0e',     # orange
        'Loss P': '#2ca02c',     # green
        'Loss Pinn': '#d62728',  # red
        'Loss Boundary': '#9467bd'  # purple
    }

    # Ensure consistent layering: PINN in the back, ALL in the front.
    # Higher zorder draws on top.
    zorder_map = {
        'Loss Pinn': 2,
        'Loss Boundary': 3,
        'Loss P': 4,
        'Loss All': 1
    }

    # Log scale (only positive values)
    # Plot order also influences layering (later plots are drawn on top).
    # Desired: Loss Pinn behind; Loss All on top.
    plot_order = ['Pinn', 'Boundary', 'P', 'Loss']
    for name in plot_order:
        if name not in series:
            continue
        values = series[name]
        vals = onp.asarray(values)
        # mask non-positive for log
        mask = vals > 0
        if mask.any():
            display_name = label_map.get(name, name)
            ax.plot(
                onp.arange(len(vals))[mask],
                vals[mask],
                label=display_name,
                color=color_map.get(display_name),
                zorder=zorder_map.get(display_name, 2),
            )
    ax.set_title(title)
    ax.set_xlabel('Iterationen')
    ax.set_ylabel('Wert (log10)')
    ax.set_yscale('log')
    if y_lim is not None:
        ax.set_ylim(list(y_lim))
    else:
        ax.set_ylim([1e-2, 1e2])
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    figs_root.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot training metrics (non-interactive supported).')
    p.add_argument('--run_dir', type=str, default=None, help='Run directory (contains data/ and figs/).')
    p.add_argument('--figs_dir', type=str, default=None, help='Output directory for figures (default: <run_dir>/figs).')
    p.add_argument('--run_name', type=str, default=None, help='Optional run name for title/filename grouping.')
    p.add_argument('--width', type=int, default=None, help='NN width (for title/filename).')
    p.add_argument('--num_layer', type=int, default=None, help='NN depth / number of hidden layers (for title/filename).')
    p.add_argument(
        '--log_min_exp',
        type=int,
        default=-4,
        help='Lower exponent for y-axis limits on log scale (e.g., -1 -> 1e-1).',
    )
    p.add_argument(
        '--log_max_exp',
        type=int,
        default=2,
        help='Upper exponent for y-axis limits on log scale (e.g., 1 -> 1e1).',
    )
    p.add_argument('--no_gui', action='store_true', help='Disable Tkinter dialogs and run headless.')
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    # Verify CPU is being used
    print(f'JAX devices: {jax.devices()}')
    print(f'JAX default backend: {jax.default_backend()}')

    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        if args.no_gui:
            raise RuntimeError('No --run_dir provided and --no_gui is set.')
        run_dir = select_run_dir(project_root)

    figs_root = Path(args.figs_dir) if args.figs_dir else (run_dir / 'figs')
    series = load_series(run_dir)

    # Title/filename must show run id + NNConfig (if provided)
    run_name = args.run_name or run_dir.name
    if args.width is not None and args.num_layer is not None:
        title = f'Run {run_name} | W={args.width}, L={args.num_layer}'
    else:
        title = f'Run {run_name}'

    if args.width is not None and args.num_layer is not None:
        out_path = figs_root / f'metrics_{run_name}_W{args.width}_L{args.num_layer}.png'
    else:
        out_path = figs_root / f'metrics_{run_name}.png'

    if args.log_min_exp > args.log_max_exp:
        raise ValueError(f'Invalid log exponent range: {args.log_min_exp} > {args.log_max_exp}')

    # y-axis range for log scale as 10**exp (e.g., -1 -> 1e-1)
    y_lim = (10.0 ** float(args.log_min_exp), 10.0 ** float(args.log_max_exp))

    out_path = make_plot(series, figs_root, title, out_path, y_lim=y_lim)
    print(f'✓ Saved: {out_path}')
    return out_path


if __name__ == '__main__':
    main()
