import sys
import os
from pathlib import Path

# Force CPU BEFORE any JAX imports (safe for SLURM/headless plotting)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import argparse
import json
import jax
import jax.numpy as np
from jax import Array, random
from jax.typing import ArrayLike as Tensor
import flax.linen as nn
from flax.typing import VariableDict
from typing import Callable, Sequence, Tuple
import numpy as onp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import orbax.checkpoint as ocp

# Type definitions
NN = Callable[[Tensor, VariableDict], Array]

# Constants
pi = np.pi
key = random.PRNGKey(42)

# ===================== Network Architecture =====================

class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    layer_sizes: Sequence[int] = None

    def setup(self, Activation=nn.tanh):
        self.layers = [nn.Dense(features=size) for size in self.layer_sizes[1:]]
        self.act = Activation

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)
        return self.layers[-1](x)


def CreateNN(NN, InputDim: int, OutputDim: int, Depth: int, width, Activation=nn.tanh) -> Tuple[nn.Module, VariableDict]:
    """Create a neural network with specified architecture"""
    _nn = NN(layer_sizes=[InputDim]+[width]*Depth+[OutputDim])
    _x = np.zeros((1, InputDim))
    params = _nn.init(key, _x)
    return _nn, params


def infer_arch_from_params(params: dict) -> tuple[int, int, int]:
    """Infer (InputDim, Width, Depth) from a Flax params tree for the MLP used here.

    Assumes params['params'] contains keys like 'layers_0', 'layers_1', ... each with a 'kernel'.
    - InputDim is kernel.shape[0] of first layer
    - Width is kernel.shape[1] of first layer
    - Depth is (#DenseLayers - 1), where last Dense is output layer
    """
    p = params.get('params', params)
    layer_keys = [k for k in p.keys() if isinstance(k, str) and k.startswith('layers_')]
    if not layer_keys:
        raise ValueError(f"Could not infer architecture: no 'layers_*' keys found. Keys: {list(p.keys())[:10]}")
    layer_keys.sort(key=lambda s: int(s.split('_')[1]))
    first = p[layer_keys[0]]
    kernel = first.get('kernel')
    if kernel is None or getattr(kernel, 'ndim', 0) != 2:
        raise ValueError("Could not infer architecture: first layer has no 2D 'kernel'.")
    input_dim = int(kernel.shape[0])
    width = int(kernel.shape[1])
    num_dense_layers = len(layer_keys)
    depth = max(0, num_dense_layers - 1)
    return input_dim, width, depth


def restore_with_candidate_architectures(checkpointer: ocp.StandardCheckpointer, ckpt_path: Path, Activation) -> tuple[dict, int, int, int]:
    """Restore checkpoint by trying a small set of likely architectures.

    This avoids restoring without a target (which can fail when checkpoint metadata references GPU devices).
    Returns (Paras, DimInput, Width, NumLayer).
    """
    # Common candidates in this repo. Extend if needed.
    candidates: list[tuple[int, int, int]] = [
        (2, 80, 2),
        (2, 80, 4),
        (4, 80, 2),
        (4, 80, 4),
        (2, 100, 2),
        (2, 100, 4),
        (2, 100, 3),
        (2, 100, 4),
        (2,10,4),
        (2,10,3),
        (2,20,3)
    ]

    last_exc: Exception | None = None
    for DimInput, Width, NumLayer in candidates:
        try:
            yNet, yParams = CreateNN(MLP, DimInput, 1, NumLayer, Width, Activation)
            pNet, pParams = CreateNN(MLP, DimInput, 1, NumLayer, Width, Activation)
            target = {'yNet': yParams, 'pNet': pParams}

            try:
                Paras = checkpointer.restore(str(ckpt_path), item=target)
            except TypeError:
                Paras = checkpointer.restore(str(ckpt_path), target=target)

            print(f"Restored checkpoint using DimInput={DimInput}, Width={Width}, NumLayer={NumLayer}")
            return Paras, DimInput, Width, NumLayer
        except Exception as e:
            last_exc = e
            continue

    raise RuntimeError(
        "Could not restore checkpoint with any candidate architecture. "
        "Update candidates in restore_with_candidate_architectures()."
    ) from last_exc

# ===================== File Selection =====================

def select_checkpoint():
    """Open file dialog to select RUN directory (contains 'final' checkpoint)."""
    # Get project root
    project_root = Path(__file__).resolve().parents[1]
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as e:
        raise RuntimeError(
            "Tkinter is not available. Use --run_dir to run non-interactively."
        ) from e

    root = tk.Tk(); root.withdraw()
    run_path = filedialog.askdirectory(
        title='Select RUN directory (e.g., 20251111_100207)',
        initialdir=str(project_root / 'Run')
    )
    if not run_path:
        messagebox.showwarning("No Selection", "No run directory selected. Exiting.")
        sys.exit(0)
    return Path(run_path)


def ask_scale_limits(default_u_min, default_u_max, default_y_min, default_y_max, 
                     default_err_u_min, default_err_u_max, default_err_y_min, default_err_y_max):
    """Ask user for custom scale limits with defaults pre-filled."""
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.title("Scale Settings")
    root.geometry("500x450")
    
    # Bring window to front
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    
    # Variables
    u_min_var = tk.StringVar(root, value=f"{default_u_min:.6f}")
    u_max_var = tk.StringVar(root, value=f"{default_u_max:.6f}")
    y_min_var = tk.StringVar(root, value=f"{default_y_min:.6f}")
    y_max_var = tk.StringVar(root, value=f"{default_y_max:.6f}")
    err_u_min_var = tk.StringVar(root, value=f"{default_err_u_min:.6e}")
    err_u_max_var = tk.StringVar(root, value=f"{default_err_u_max:.6e}")
    err_y_min_var = tk.StringVar(root, value=f"{default_err_y_min:.6e}")
    err_y_max_var = tk.StringVar(root, value=f"{default_err_y_max:.6e}")
    draw_box_var = tk.BooleanVar(root, value=True)
    
    result = {'cancelled': True}
    
    # Layout
    row = 0
    tk.Label(root, text="Control (u) Scale:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', padx=10, pady=(10,5))
    row += 1
    tk.Label(root, text="Min:").grid(row=row, column=0, sticky='e', padx=5, pady=2)
    tk.Entry(root, textvariable=u_min_var, width=20).grid(row=row, column=1, sticky='w', padx=5, pady=2)
    row += 1
    tk.Label(root, text="Max:").grid(row=row, column=0, sticky='e', padx=5, pady=2)
    tk.Entry(root, textvariable=u_max_var, width=20).grid(row=row, column=1, sticky='w', padx=5, pady=2)
    
    row += 1
    tk.Label(root, text="State (y) Scale:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', padx=10, pady=(15,5))
    row += 1
    tk.Label(root, text="Min:").grid(row=row, column=0, sticky='e', padx=5, pady=2)
    tk.Entry(root, textvariable=y_min_var, width=20).grid(row=row, column=1, sticky='w', padx=5, pady=2)
    row += 1
    tk.Label(root, text="Max:").grid(row=row, column=0, sticky='e', padx=5, pady=2)
    tk.Entry(root, textvariable=y_max_var, width=20).grid(row=row, column=1, sticky='w', padx=5, pady=2)
    
    row += 1
    tk.Label(root, text="u Error Scale:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', padx=10, pady=(15,5))
    row += 1
    tk.Label(root, text="Min:").grid(row=row, column=0, sticky='e', padx=5, pady=2)
    tk.Entry(root, textvariable=err_u_min_var, width=20).grid(row=row, column=1, sticky='w', padx=5, pady=2)
    row += 1
    tk.Label(root, text="Max:").grid(row=row, column=0, sticky='e', padx=5, pady=2)
    tk.Entry(root, textvariable=err_u_max_var, width=20).grid(row=row, column=1, sticky='w', padx=5, pady=2)
    
    row += 1
    tk.Label(root, text="y Error Scale:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', padx=10, pady=(15,5))
    row += 1
    tk.Label(root, text="Min:").grid(row=row, column=0, sticky='e', padx=5, pady=2)
    tk.Entry(root, textvariable=err_y_min_var, width=20).grid(row=row, column=1, sticky='w', padx=5, pady=2)
    row += 1
    tk.Label(root, text="Max:").grid(row=row, column=0, sticky='e', padx=5, pady=2)
    tk.Entry(root, textvariable=err_y_max_var, width=20).grid(row=row, column=1, sticky='w', padx=5, pady=2)
    
    row += 1
    tk.Checkbutton(root, text="Draw boundary box (x₁>1/2, x₂>1/2)", variable=draw_box_var).grid(row=row, column=0, columnspan=2, sticky='w', padx=10, pady=(15,5))
    
    def on_ok():
        try:
            print("OK clicked, parsing values...")
            result['u_min'] = float(u_min_var.get())
            result['u_max'] = float(u_max_var.get())
            result['y_min'] = float(y_min_var.get())
            result['y_max'] = float(y_max_var.get())
            result['err_u_min'] = float(err_u_min_var.get())
            result['err_u_max'] = float(err_u_max_var.get())
            result['err_y_min'] = float(err_y_min_var.get())
            result['err_y_max'] = float(err_y_max_var.get())
            result['draw_box'] = draw_box_var.get()
            result['cancelled'] = False
            print("Values parsed successfully, closing dialog...")
            root.quit()  # Stop mainloop
            root.destroy()
        except ValueError as e:
            print(f"Error parsing values: {e}")
            messagebox.showerror("Invalid Input", f"Please enter valid numbers for all fields.\n\nError: {e}")
        except Exception as e:
            print(f"Unexpected error in on_ok: {e}")
            import traceback
            traceback.print_exc()
    
    def on_cancel():
        print("Cancel clicked")
        root.quit()
        root.destroy()
    
    row += 1
    button_frame = tk.Frame(root)
    button_frame.grid(row=row, column=0, columnspan=2, pady=20)
    tk.Button(button_frame, text="OK", command=on_ok, width=10).pack(side='left', padx=5)
    tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side='left', padx=5)
    
    root.mainloop()
    
    if result.get('cancelled', True):
        messagebox.showwarning("Cancelled", "Scale dialog cancelled. Using defaults.")
        return {
            'u_min': default_u_min, 'u_max': default_u_max,
            'y_min': default_y_min, 'y_max': default_y_max,
            'err_u_min': default_err_u_min, 'err_u_max': default_err_u_max,
            'err_y_min': default_err_y_min, 'err_y_max': default_err_y_max,
            'draw_box': True
        }
    
    return result

# ===================== Main Program =====================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Plot final checkpoint (non-interactive supported).')
    p.add_argument('--run_dir', type=str, default=None, help='Run directory containing a final/ checkpoint.')
    p.add_argument('--figs_dir', type=str, default=None, help='Output directory for figures (default: <run_dir>/figs).')
    p.add_argument('--width', type=int, default=None, help='NN width (for restore + title/filename).')
    p.add_argument('--num_layer', type=int, default=None, help='NN depth / number of hidden layers (for restore + title/filename).')
    # Optional hyperparameters for display in the plot header
    p.add_argument('--RunId', dest='run_id', type=str, default=None, help='Run-ID to display in plot header.')
    p.add_argument('--MCSizeIn', dest='mc_size_in', type=int, default=None, help='MCSizeIn to display in plot header.')
    p.add_argument('--MCsizeB', dest='mc_size_b', type=int, default=None, help='MCsizeB to display in plot header.')
    p.add_argument('--Epoach', dest='epoach', type=int, default=None, help='Epoach to display in plot header.')
    p.add_argument('--LearningRateStart', dest='learning_rate_start', type=float, default=None, help='LearningRateStart to display in plot header.')
    p.add_argument('--DecayRate', dest='decay_rate', type=float, default=None, help='DecayRate to display in plot header.')
    p.add_argument('--EpoachDecay', dest='epoach_decay', type=int, default=None, help='EpoachDecay to display in plot header.')
    p.add_argument('--dpi', type=int, default=300, help='Grid resolution for plotting (default: 300).')
    p.add_argument('--no_gui', action='store_true', help='Disable Tkinter dialogs and use CLI/default values.')
    p.add_argument('--open', action='store_true', help='Open the generated image in the OS viewer (default: off).')

    # Fixed-default scales as requested
    p.add_argument('--u_min', type=float, default=0.0)
    p.add_argument('--u_max', type=float, default=40.0)
    p.add_argument('--y_min', type=float, default=0.0)
    p.add_argument('--y_max', type=float, default=1.0)
    p.add_argument('--err_u_min', type=float, default=0.0)
    p.add_argument('--err_u_max', type=float, default=10.0)
    p.add_argument('--err_y_min', type=float, default=0.0)
    p.add_argument('--err_y_max', type=float, default=0.25)

    p.add_argument('--draw_box', action=argparse.BooleanOptionalAction, default=True,
                   help='Draw the boundary box (upper-right excluded quadrant).')
    p.add_argument(
        '--l_domain',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Plot only on the L-shaped domain (x1<=1/2 OR x2<=1/2). If enabled, the boundary box is not drawn.',
    )
    p.add_argument(
        '--err_scale',
        type=str,
        default='linear',
        choices=['linear', 'log'],
        help='Color scale for pointwise error heatmaps (u/y).',
    )
    p.add_argument('--run_name', type=str, default=None, help='Optional run name for title/filename grouping.')
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    # Verify CPU is being used
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    
    # Get project root
    project_root = Path(__file__).resolve().parents[1]
    
    # Config not tied to checkpoint
    lam = 0.01
    args = parse_args(argv)

    Activation = nn.tanh
    dpi = int(args.dpi)

    try:
        # Select run directory and prepare output
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            if args.no_gui:
                raise RuntimeError('No --run_dir provided and --no_gui is set.')
            run_dir = select_checkpoint()
        
        figs_root = Path(args.figs_dir) if args.figs_dir else (run_dir / 'figs')
        figs_root.mkdir(exist_ok=True, parents=True)

        ckpt_root = run_dir

        # Optional: load extra metadata (hyperparameters) from summary.json if not provided via CLI.
        summary_path = run_dir / 'summary.json'
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
            except Exception:
                summary = None
            if isinstance(summary, dict):
                if args.run_id is None:
                    args.run_id = summary.get('run_id')
                if args.mc_size_in is None:
                    args.mc_size_in = summary.get('mc_size_in')
                if args.mc_size_b is None:
                    args.mc_size_b = summary.get('mc_size_b')
                if args.learning_rate_start is None:
                    args.learning_rate_start = summary.get('learning_rate_start')
                if args.decay_rate is None:
                    args.decay_rate = summary.get('decay_rate')
                if args.epoach is None:
                    args.epoach = summary.get('epoach')
                if args.epoach_decay is None:
                    args.epoach_decay = summary.get('epoach_decay')

        # Find 'final' checkpoint, or fallback to largest numbered checkpoint
        final_ckpt = ckpt_root / 'final'
        if not final_ckpt.exists() or not final_ckpt.is_dir():
            print("'final' checkpoint not found. Looking for largest numbered checkpoint...")
            # Find all numeric checkpoints
            def is_numeric_ckpt(p: Path) -> bool:
                if not p.is_dir():
                    return False
                try:
                    int(p.name)
                    return True
                except ValueError:
                    return False
            
            if not ckpt_root.exists() or not ckpt_root.is_dir():
                raise FileNotFoundError(f"Checkpoint root not found: {ckpt_root}")

            numeric_ckpts = [p for p in ckpt_root.iterdir() if is_numeric_ckpt(p)]
            if not numeric_ckpts:
                raise FileNotFoundError(f"No 'final' or numeric checkpoints found in {ckpt_root}.")
            
            # Sort by numeric value and take the largest
            numeric_ckpts.sort(key=lambda p: int(p.name))
            final_ckpt = numeric_ckpts[-1]
            print(f"Using checkpoint: {final_ckpt.name}")
        else:
            print(f"Loading final checkpoint from: {final_ckpt}")

        # Prepare checkpointer
        checkpointer = ocp.StandardCheckpointer()

        # Restore parameters using a CPU-safe target.
        # Prefer the explicit architecture from CLI (ex1.py knows W/L), fall back to candidates.
        if args.width is not None and args.num_layer is not None:
            preferred = (2, int(args.width), int(args.num_layer))
            last_exc: Exception | None = None
            for DimInput, Width, NumLayer in [preferred, (4, int(args.width), int(args.num_layer))]:
                try:
                    yNet, yParams = CreateNN(MLP, DimInput, 1, NumLayer, Width, Activation)
                    pNet, pParams = CreateNN(MLP, DimInput, 1, NumLayer, Width, Activation)
                    target = {'yNet': yParams, 'pNet': pParams}
                    Paras = checkpointer.restore(str(final_ckpt), target=target)
                    print(f"Restored checkpoint using DimInput={DimInput}, Width={Width}, NumLayer={NumLayer}")
                    break
                except Exception as e:
                    last_exc = e
                    Paras = None
            if Paras is None:
                print(f"Explicit restore failed, falling back to candidate search: {last_exc}")
                Paras, DimInput, Width, NumLayer = restore_with_candidate_architectures(checkpointer, final_ckpt, Activation)
        else:
            Paras, DimInput, Width, NumLayer = restore_with_candidate_architectures(checkpointer, final_ckpt, Activation)

        # Build networks for apply (must match restored params)
        yNet, _ = CreateNN(MLP, DimInput, 1, NumLayer, Width, Activation)
        pNet, _ = CreateNN(MLP, DimInput, 1, NumLayer, Width, Activation)

        def ynn(x, para):
            return yNet.apply(para, x)

        def pnn(x, para):
            return pNet.apply(para, x)

        def unn(x, para):
            return -(1.0/lam) * pnn(x, para)

        # Build evaluation grid: always visualize over (x1,x2), remaining dims fixed to 0.5
        X = np.linspace(0, 1, dpi)
        X = np.stack(np.meshgrid(X, X), axis=-1).reshape(-1, 2)
        if DimInput > 2:
            X_fixed = np.zeros((dpi * dpi, DimInput - 2)) + 0.5
            X = np.concatenate([X, X_fixed], axis=1)

        # Define exact solutions (dimension-aware amplitude for u)
        def u_exact(x):
            """u(x) = d·π² · ∏ sin(π x_i)  (d = DimInput)"""
            return (DimInput * pi**2) * np.prod(np.sin(pi * x), axis=1)

        def y_exact(x):
            """y(x) = ∏ sin(π x_i)"""
            return np.prod(np.sin(pi * x), axis=1)

        U_exact_np = onp.array(u_exact(X)).reshape(dpi, dpi)
        Y_exact_np = onp.array(y_exact(X)).reshape(dpi, dpi)

        # Ensure CPU arrays
        import jax.tree_util as tree
        cpu_dev = jax.devices('cpu')[0]
        Paras = tree.tree_map(lambda a: jax.device_put(a, cpu_dev) if isinstance(a, jax.Array) else a, Paras)

        print("Computing approximations...")
        # Compute approximations
        Y_approx = ynn(X, Paras['yNet'])
        U_approx = unn(X, Paras['pNet'])
        Y_approx_np = onp.array(Y_approx).reshape(dpi, dpi)
        U_approx_np = onp.array(U_approx).reshape(dpi, dpi)

        print("Computing errors...")
        # Compute errors
        U_error_np = onp.abs(U_exact_np - U_approx_np)
        Y_error_np = onp.abs(Y_exact_np - Y_approx_np)

        # L-domain mask (always used for error statistics; optionally used for plotting)
        x_lin = onp.linspace(0.0, 1.0, dpi)
        X1, X2 = onp.meshgrid(x_lin, x_lin)
        l_mask = (X1 <= 0.5) | (X2 <= 0.5)

        # Error statistics over L-domain (requested)
        max_u_error = float(onp.max(U_error_np[l_mask]))
        mean_u_error = float(onp.mean(U_error_np[l_mask]))
        max_y_error = float(onp.max(Y_error_np[l_mask]))
        mean_y_error = float(onp.mean(Y_error_np[l_mask]))

        # Persist error statistics into summary.json (if present) so other scripts can reuse them.
        # Only write if the fields are not already stored.
        if summary_path.exists():
            try:
                summary_existing = json.loads(summary_path.read_text(encoding='utf-8'))
            except Exception:
                summary_existing = None

            if isinstance(summary_existing, dict):
                errors = summary_existing.get('errors')
                if not isinstance(errors, dict):
                    errors = {}

                # Store explicitly as L-domain stats (mask: x1<=1/2 OR x2<=1/2)
                l_stats = errors.get('L')
                if not isinstance(l_stats, dict):
                    l_stats = {}

                u_stats = l_stats.get('u')
                if not isinstance(u_stats, dict):
                    u_stats = {}
                y_stats = l_stats.get('y')
                if not isinstance(y_stats, dict):
                    y_stats = {}

                changed = False
                if u_stats.get('max') is None:
                    u_stats['max'] = float(max_u_error)
                    changed = True
                if u_stats.get('mean') is None:
                    u_stats['mean'] = float(mean_u_error)
                    changed = True
                if y_stats.get('max') is None:
                    y_stats['max'] = float(max_y_error)
                    changed = True
                if y_stats.get('mean') is None:
                    y_stats['mean'] = float(mean_y_error)
                    changed = True

                if l_stats.get('mask') is None:
                    l_stats['mask'] = 'x1<=0.5 OR x2<=0.5'
                    changed = True
                if l_stats.get('dpi') is None:
                    l_stats['dpi'] = int(dpi)
                    changed = True

                l_stats['u'] = u_stats
                l_stats['y'] = y_stats
                errors['L'] = l_stats
                summary_existing['errors'] = errors

                if changed:
                    try:
                        summary_path.write_text(json.dumps(summary_existing, indent=2), encoding='utf-8')
                    except Exception:
                        # Do not fail plotting if OneDrive/Windows file locks occur.
                        pass

        # Scale limits
        if args.no_gui:
            u_min, u_max = args.u_min, args.u_max
            y_min, y_max = args.y_min, args.y_max
            err_u_min, err_u_max = args.err_u_min, args.err_u_max
            err_y_min, err_y_max = args.err_y_min, args.err_y_max
            draw_box = bool(args.draw_box)
        else:
            # Compute automatic defaults and ask user (legacy interactive mode)
            u_min_auto = min(U_exact_np.min(), U_approx_np.min())
            u_max_auto = max(U_exact_np.max(), U_approx_np.max())
            y_min_auto = min(Y_exact_np.min(), Y_approx_np.min())
            y_max_auto = max(Y_exact_np.max(), Y_approx_np.max())
            err_u_min_auto = U_error_np.min()
            err_u_max_auto = U_error_np.max()
            err_y_min_auto = Y_error_np.min()
            err_y_max_auto = Y_error_np.max()

            print('Opening scale dialog...')
            scales = ask_scale_limits(
                u_min_auto, u_max_auto, y_min_auto, y_max_auto,
                err_u_min_auto, err_u_max_auto, err_y_min_auto, err_y_max_auto
            )
            u_min = scales['u_min']
            u_max = scales['u_max']
            y_min = scales['y_min']
            y_max = scales['y_max']
            err_u_min = scales['err_u_min']
            err_u_max = scales['err_u_max']
            err_y_min = scales['err_y_min']
            err_y_max = scales['err_y_max']
            draw_box = scales['draw_box']

        # Domain controls
        l_only = bool(args.l_domain)
        effective_draw_box = bool(draw_box) and (not l_only)

        # Apply L-domain masking only for plotting (after scale selection, to avoid NaN affecting auto scales)
        if l_only:
            def _mask(arr2d: onp.ndarray) -> onp.ndarray:
                out = onp.array(arr2d, copy=True)
                out[~l_mask] = onp.nan
                return out

            U_exact_np = _mask(U_exact_np)
            U_approx_np = _mask(U_approx_np)
            U_error_np = _mask(U_error_np)
            Y_exact_np = _mask(Y_exact_np)
            Y_approx_np = _mask(Y_approx_np)
            Y_error_np = _mask(Y_error_np)

        print('Creating plots...')

        use_log_err = str(args.err_scale).lower() == 'log'

        # Move subplot titles slightly upward to avoid overlap.
        # Matplotlib's `pad` is in points; with savefig dpi=150, +3px ~= +1.44pt.
        _save_dpi = 150
        _default_title_pad_pt = 6.0
        _title_pad_pt = _default_title_pad_pt + (3.0 * 72.0 / float(_save_dpi))

        # Create 6-panel figure (2 rows, 3 cols)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Helper function to draw boundary box (x1>0.5, x2>0.5)
        def add_boundary_box(ax):
            """Draw black rectangle around upper-right quadrant (x1>0.5, x2>0.5)"""
            if effective_draw_box:
                from matplotlib.patches import Rectangle
                # Rectangle from (dpi/2, dpi/2) to (dpi, dpi)
                half = dpi / 2
                rect = Rectangle((half, half), half, half, 
                                linewidth=1.5, edgecolor='black', facecolor='none')
                ax.add_patch(rect)
        
        # Row 1: u (Control)
        # u exact
        im0 = axes[0, 0].imshow(U_exact_np, cmap='coolwarm', origin='lower', vmin=u_min, vmax=u_max)
        axes[0, 0].set_title(f'u exakt: ${DimInput}\\pi^2 \\prod\\sin(\\pi x_i)$', fontsize=12, pad=_title_pad_pt)
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # u approximation
        im1 = axes[0, 1].imshow(U_approx_np, cmap='coolwarm', origin='lower', vmin=u_min, vmax=u_max)
        axes[0, 1].set_title('u Approximation (NN)', fontsize=12, pad=_title_pad_pt)
        axes[0, 1].axis('off')
        add_boundary_box(axes[0, 1])
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # u error
        if use_log_err:
            u_pos = U_error_np[onp.isfinite(U_error_np) & (U_error_np > 0)]
            u_eps = float(u_pos.min()) if u_pos.size else 1e-12
            u_vmin = max(float(err_u_min), u_eps)
            u_vmax = max(float(err_u_max), u_vmin * 10.0)
            im2 = axes[0, 2].imshow(
                U_error_np,
                cmap='viridis',
                origin='lower',
                norm=LogNorm(vmin=u_vmin, vmax=u_vmax),
            )
        else:
            im2 = axes[0, 2].imshow(U_error_np, cmap='viridis', origin='lower', vmin=err_u_min, vmax=err_u_max)
        axes[0, 2].set_title(
            f'u punktweiser Fehler (max={max_u_error:.2e}, mean={mean_u_error:.2e})',
            fontsize=12,
            pad=_title_pad_pt,
        )
        axes[0, 2].axis('off')
        add_boundary_box(axes[0, 2])
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Row 2: y (State)
        # y exact
        im3 = axes[1, 0].imshow(Y_exact_np, cmap='coolwarm', origin='lower', vmin=y_min, vmax=y_max)
        axes[1, 0].set_title('y exakt: $\\prod\\sin(\\pi x_i)$', fontsize=12, pad=_title_pad_pt)
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # y approximation
        im4 = axes[1, 1].imshow(Y_approx_np, cmap='coolwarm', origin='lower', vmin=y_min, vmax=y_max)
        axes[1, 1].set_title('y Approximation (NN)', fontsize=12, pad=_title_pad_pt)
        axes[1, 1].axis('off')
        add_boundary_box(axes[1, 1])
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # y error
        if use_log_err:
            y_pos = Y_error_np[onp.isfinite(Y_error_np) & (Y_error_np > 0)]
            y_eps = float(y_pos.min()) if y_pos.size else 1e-12
            y_vmin = max(float(err_y_min), y_eps)
            y_vmax = max(float(err_y_max), y_vmin * 10.0)
            im5 = axes[1, 2].imshow(
                Y_error_np,
                cmap='viridis',
                origin='lower',
                norm=LogNorm(vmin=y_vmin, vmax=y_vmax),
            )
        else:
            im5 = axes[1, 2].imshow(Y_error_np, cmap='viridis', origin='lower', vmin=err_y_min, vmax=err_y_max)
        axes[1, 2].set_title(
            f'y punktweiser Fehler (max={max_y_error:.2e}, mean={mean_y_error:.2e})',
            fontsize=12,
            pad=_title_pad_pt,
        )
        axes[1, 2].axis('off')
        add_boundary_box(axes[1, 2])
        plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # ---- 3-line hyperparameter header (3 columns, aligned separators) ----
        run_name = args.run_name or run_dir.name
        run_id_display = args.run_id or run_name

        def _fmt(v):
            if v is None:
                return 'n/a'
            if isinstance(v, float):
                return f'{v:g}'
            return str(v)

        rows = [
            (f"Run-ID: {_fmt(run_id_display)}", f"Width: {_fmt(Width)}", f"Layer: {_fmt(NumLayer)}"),
            (f"MCSizeIn: {_fmt(args.mc_size_in)}", f"MCsizeB: {_fmt(args.mc_size_b)}", f"Epoach: {_fmt(args.epoach)}"),
            (f"LearningRateStart: {_fmt(args.learning_rate_start)}", f"DecayRate: {_fmt(args.decay_rate)}", f"EpoachDecay: {_fmt(args.epoach_decay)}"),
        ]

        col_widths = [
            max(len(r[0]) for r in rows),
            max(len(r[1]) for r in rows),
            max(len(r[2]) for r in rows),
        ]

        header_lines = []
        for a, b, c in rows:
            header_lines.append(
                "  "
                + a.ljust(col_widths[0])
                + " | "
                + b.ljust(col_widths[1])
                + " | "
                + c.ljust(col_widths[2])
            )
        header_text = "\n".join(header_lines)

        # Put header above plots; keep it monospace so '|' align vertically.
        header_artist = fig.text(
            0.5,
            0.985,
            header_text,
            ha='center',
            va='top',
            fontsize=12,
            fontweight='bold',
            family='monospace',
        )

        # Layout: keep ~10px gap between header and top row of plots.
        # We compute the header bbox in pixel space and convert to figure coordinates.
        try:
            gap_px = 10
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = header_artist.get_window_extent(renderer=renderer)
            # bbox.y0 is the bottom of the header in pixels (from figure bottom)
            top = (bbox.y0 - gap_px) / fig.bbox.height
            # Clamp to a reasonable range
            top = float(min(0.95, max(0.60, top)))
        except Exception:
            top = 0.86

        plt.tight_layout(rect=[0, 0, 1, top])
        
        print("Saving image...")
        # Output filename (must show run name + NNConfig; keep concise)
        box_tag = 'box' if effective_draw_box else 'nobox'
        err_tag = 'logerr' if use_log_err else 'linerr'
        dom_tag = 'Lonly' if l_only else 'full'
        run_name = args.run_name or run_dir.name
        fname = f"final_{run_name}_W{Width}_L{NumLayer}_{box_tag}_{err_tag}_{dom_tag}.png"
        output_path = figs_root / fname
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f"Image saved to: {output_path}")
        # Open in default viewer (optional)
        if args.open:
            try:
                import platform
                import subprocess
                if platform.system() == 'Windows':
                    os.startfile(str(output_path))
                elif platform.system() == 'Darwin':
                    subprocess.run(['open', str(output_path)], check=False)
                else:
                    subprocess.run(['xdg-open', str(output_path)], check=False)
            except Exception as e:
                print(f"Could not open image automatically: {e}")

        # Print to console
        print(f"✓ Saved PNG to: {output_path}")
        print(f"  u error: max={max_u_error:.2e}, mean={mean_u_error:.2e}")
        print(f"  y error: max={max_y_error:.2e}, mean={mean_y_error:.2e}")
        
        return output_path
        
    except FileNotFoundError as e:
        print(f"Error: Checkpoint not found: {str(e)}")
        import traceback; traceback.print_exc()
    except KeyError as e:
        print(f"Error: Invalid checkpoint structure. Missing key {str(e)}")
        print("Make sure the checkpoint contains 'yNet' and 'pNet'.")
        import traceback; traceback.print_exc()
    except Exception as e:
        print(f"Error: An error occurred: {str(e)}")
        import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()