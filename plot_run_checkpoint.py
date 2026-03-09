import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


CFG_RE = re.compile(r"^W(?P<width>\d+)_L(?P<layer>\d+)$")


@dataclass(frozen=True)
class PlotSettings:
    u_min: float
    u_max: float
    y_min: float
    y_max: float
    err_u_min: float
    err_u_max: float
    err_y_min: float
    err_y_max: float
    draw_box: bool
    err_scale: str  # 'linear' | 'log'
    l_domain: bool


def _slug_float(x: float) -> str:
    s = f"{x:g}"
    s = s.replace("-", "m").replace("+", "")
    s = s.replace(".", "p")
    s = s.replace("e", "e")
    return s


def settings_folder_name(s: PlotSettings) -> str:
    # If plotting only on the L-domain, the boundary box is not drawn.
    effective_box = bool(s.draw_box) and (not bool(s.l_domain))
    box_tag = "box" if effective_box else "nobox"
    err_tag = "logerr" if s.err_scale == "log" else "linerr"
    dom_tag = "Lonly" if s.l_domain else "full"
    return (
        f"u{_slug_float(s.u_min)}-{_slug_float(s.u_max)}_"
        f"y{_slug_float(s.y_min)}-{_slug_float(s.y_max)}_"
        f"eu{_slug_float(s.err_u_min)}-{_slug_float(s.err_u_max)}_"
        f"ey{_slug_float(s.err_y_min)}-{_slug_float(s.err_y_max)}_"
        f"{box_tag}_{err_tag}_{dom_tag}"
    )


def select_run_root(default_runs_dir: Path) -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as e:
        raise RuntimeError("Tkinter is not available. Use --run_root.") from e

    root = tk.Tk()
    root.withdraw()
    run_path = filedialog.askdirectory(
        title="Select RUN directory (contains data/)",
        initialdir=str(default_runs_dir),
    )
    if not run_path:
        messagebox.showwarning("No Selection", "No run directory selected. Exiting.")
        sys.exit(0)
    return Path(run_path)


def ask_settings(defaults: PlotSettings) -> PlotSettings:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.title("Run checkpoint plot settings")
    root.geometry("560x660")
    root.minsize(560, 660)
    root.resizable(True, True)

    root.lift()
    root.attributes("-topmost", True)
    root.after_idle(root.attributes, "-topmost", False)

    def sv(v: float) -> tk.StringVar:
        return tk.StringVar(root, value=f"{v:g}")

    u_min_var = sv(defaults.u_min)
    u_max_var = sv(defaults.u_max)
    y_min_var = sv(defaults.y_min)
    y_max_var = sv(defaults.y_max)
    eu_min_var = sv(defaults.err_u_min)
    eu_max_var = sv(defaults.err_u_max)
    ey_min_var = sv(defaults.err_y_min)
    ey_max_var = sv(defaults.err_y_max)
    draw_box_var = tk.BooleanVar(root, value=bool(defaults.draw_box))
    log_err_var = tk.BooleanVar(root, value=(defaults.err_scale == "log"))
    l_domain_var = tk.BooleanVar(root, value=bool(defaults.l_domain))

    result: dict = {"cancelled": True}

    row = 0
    tk.Label(root, text="Value scales", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 6)
    )

    def add_minmax(title: str, min_var: tk.StringVar, max_var: tk.StringVar):
        nonlocal row
        row += 1
        tk.Label(root, text=title, font=("Arial", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(12, 4)
        )
        row += 1
        tk.Label(root, text="Min:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        tk.Entry(root, textvariable=min_var, width=22).grid(
            row=row, column=1, sticky="w", padx=5, pady=2
        )
        row += 1
        tk.Label(root, text="Max:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
        tk.Entry(root, textvariable=max_var, width=22).grid(
            row=row, column=1, sticky="w", padx=5, pady=2
        )

    add_minmax("Control u", u_min_var, u_max_var)
    add_minmax("State y", y_min_var, y_max_var)

    row += 1
    tk.Label(root, text="Error scales", font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(16, 6)
    )

    add_minmax("u error", eu_min_var, eu_max_var)
    add_minmax("y error", ey_min_var, ey_max_var)

    row += 1
    tk.Checkbutton(root, text="Draw boundary box (x₁>1/2, x₂>1/2)", variable=draw_box_var).grid(
        row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(14, 4)
    )

    row += 1
    tk.Checkbutton(root, text="Log scale for pointwise errors (u/y)", variable=log_err_var).grid(
        row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(4, 10)
    )

    row += 1
    tk.Checkbutton(root, text="Plot only L-domain (default)", variable=l_domain_var).grid(
        row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(4, 10)
    )

    def on_ok():
        try:
            result.update(
                {
                    "u_min": float(u_min_var.get()),
                    "u_max": float(u_max_var.get()),
                    "y_min": float(y_min_var.get()),
                    "y_max": float(y_max_var.get()),
                    "err_u_min": float(eu_min_var.get()),
                    "err_u_max": float(eu_max_var.get()),
                    "err_y_min": float(ey_min_var.get()),
                    "err_y_max": float(ey_max_var.get()),
                    "draw_box": bool(draw_box_var.get()),
                    "err_scale": "log" if bool(log_err_var.get()) else "linear",
                    "l_domain": bool(l_domain_var.get()),
                    "cancelled": False,
                }
            )
            root.quit()
            root.destroy()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid numbers.\n\nError: {e}")

    def on_cancel():
        root.quit()
        root.destroy()

    btn = tk.Frame(root)
    btn.grid(row=row + 1, column=0, columnspan=2, pady=16)
    tk.Button(btn, text="OK", command=on_ok, width=10).pack(side="left", padx=6)
    tk.Button(btn, text="Cancel", command=on_cancel, width=10).pack(side="left", padx=6)

    root.mainloop()

    if result.get("cancelled", True):
        return defaults

    return PlotSettings(
        u_min=result["u_min"],
        u_max=result["u_max"],
        y_min=result["y_min"],
        y_max=result["y_max"],
        err_u_min=result["err_u_min"],
        err_u_max=result["err_u_max"],
        err_y_min=result["err_y_min"],
        err_y_max=result["err_y_max"],
        draw_box=result["draw_box"],
        err_scale=result["err_scale"],
        l_domain=result["l_domain"],
    )


def find_config_dirs(run_root: Path) -> list[Path]:
    data_root = run_root / "data"
    if not data_root.exists() or not data_root.is_dir():
        return []
    cfgs = [p for p in data_root.iterdir() if p.is_dir() and CFG_RE.match(p.name)]
    cfgs.sort(key=lambda p: int(CFG_RE.match(p.name).group("width")))
    return cfgs


def parse_cfg_name(cfg_dir: Path) -> tuple[int, int]:
    m = CFG_RE.match(cfg_dir.name)
    if not m:
        raise ValueError(f"Not a config dir: {cfg_dir}")
    return int(m.group("width")), int(m.group("layer"))


def build_plot_checkpoint_cmd(
    plot_checkpoint_py: Path,
    cfg_dir: Path,
    out_dir: Path,
    run_name: str,
    width: int,
    num_layer: int,
    settings: PlotSettings,
) -> list[str]:
    cmd = [
        sys.executable,
        str(plot_checkpoint_py),
        "--run_dir",
        str(cfg_dir),
        "--figs_dir",
        str(out_dir),
        "--run_name",
        str(run_name),
        "--width",
        str(width),
        "--num_layer",
        str(num_layer),
        "--no_gui",
        "--u_min",
        str(settings.u_min),
        "--u_max",
        str(settings.u_max),
        "--y_min",
        str(settings.y_min),
        "--y_max",
        str(settings.y_max),
        "--err_u_min",
        str(settings.err_u_min),
        "--err_u_max",
        str(settings.err_u_max),
        "--err_y_min",
        str(settings.err_y_min),
        "--err_y_max",
        str(settings.err_y_max),
        "--err_scale",
        str(settings.err_scale),
        "--l_domain" if settings.l_domain else "--no-l_domain",
    ]
    if settings.draw_box:
        cmd.append("--draw_box")
    else:
        cmd.append("--no-draw_box")
    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create plot_checkpoint figures for all configs in a run.")
    p.add_argument("--run_root", type=str, default=None, help="Run root (contains data/). If omitted, opens folder picker.")
    p.add_argument("--no_gui", action="store_true", help="Disable dialogs and use defaults (for batch/headless).")
    p.add_argument("--only", type=str, default=None, help="Optional regex filter applied to config folder names (e.g. 'W(10|20)_').")

    # Defaults match the fixed scales used in ex1.py
    p.add_argument("--u_min", type=float, default=0.0)
    p.add_argument("--u_max", type=float, default=20.0)
    p.add_argument("--y_min", type=float, default=0.0)
    p.add_argument("--y_max", type=float, default=1.0)
    p.add_argument("--err_u_min", type=float, default=0.0)
    p.add_argument("--err_u_max", type=float, default=2.0)
    p.add_argument("--err_y_min", type=float, default=0.0)
    p.add_argument("--err_y_max", type=float, default=0.05)
    p.add_argument("--draw_box", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--err_scale", type=str, default="linear", choices=["linear", "log"])
    p.add_argument("--l_domain", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    default_runs_dir = project_root / "Run"

    if args.run_root:
        run_root = Path(args.run_root)
    else:
        if args.no_gui:
            raise RuntimeError("No --run_root provided and --no_gui is set.")
        run_root = select_run_root(default_runs_dir)

    if not (run_root / "data").exists():
        raise FileNotFoundError(f"Selected folder has no data/: {run_root}")

    defaults = PlotSettings(
        u_min=float(args.u_min),
        u_max=float(args.u_max),
        y_min=float(args.y_min),
        y_max=float(args.y_max),
        err_u_min=float(args.err_u_min),
        err_u_max=float(args.err_u_max),
        err_y_min=float(args.err_y_min),
        err_y_max=float(args.err_y_max),
        draw_box=bool(args.draw_box),
        err_scale=str(args.err_scale),
        l_domain=bool(args.l_domain),
    )

    settings = defaults if args.no_gui else ask_settings(defaults)

    out_root = run_root / "Auswertung" / settings_folder_name(settings)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_dirs = find_config_dirs(run_root)
    if args.only:
        rx = re.compile(args.only)
        cfg_dirs = [p for p in cfg_dirs if rx.search(p.name)]

    if not cfg_dirs:
        print(f"No config directories found under: {run_root / 'data'}")
        return 1

    plot_checkpoint_py = Path(__file__).resolve().parent / "plot_checkpoint.py"
    if not plot_checkpoint_py.exists():
        raise FileNotFoundError(f"Missing: {plot_checkpoint_py}")

    print(f"Run: {run_root.name}")
    print(f"Configs: {len(cfg_dirs)}")
    print(f"Output: {out_root}")

    failures: list[str] = []
    for cfg_dir in cfg_dirs:
        width, num_layer = parse_cfg_name(cfg_dir)
        final_dir = cfg_dir / "final"
        if not final_dir.exists():
            print(f"- Skipping {cfg_dir.name}: missing final/")
            continue

        cmd = build_plot_checkpoint_cmd(
            plot_checkpoint_py,
            cfg_dir,
            out_root,
            run_root.name,
            width,
            num_layer,
            settings,
        )

        print(f"- Plotting {cfg_dir.name}...")
        import subprocess

        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            failures.append(cfg_dir.name)
            print(f"  ✗ Failed: {cfg_dir.name} (exit={proc.returncode})")
        else:
            print(f"  ✓ Done: {cfg_dir.name}")

    if failures:
        print("Failures:")
        for name in failures:
            print(f"  - {name}")
        return 2

    print("✓ All done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
