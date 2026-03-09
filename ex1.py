# %%
import sys
import os
from pathlib import Path
import subprocess
import json
import secrets

# Add parent directory to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import header as hdr
from header import * 

import argparse
# %%
parser = argparse.ArgumentParser(description='Hyperparameters for the model')
parser.add_argument('--Width', type=int, default=80, help='NN width')
parser.add_argument('--NumLayer', type=int, default=2, help='NN depth (number of hidden layers)')
parser.add_argument('--Seed', type=int, default=None, help='PRNG seed for sampling/training (default: random)')
parser.add_argument('--RunId', type=str, default=None, help='Optional run id override (default: Timetxt)')
parser.add_argument('--SaveEvery', type=int, default=0, help='Optional periodic checkpoint interval (0=disable)')
parser.add_argument('--DeleteMetricCkpts', type=int, default=1, help='Delete metric checkpoint folders (Loss/P/Pinn/Boundary) after plotting (1=yes, 0=no).')
parser.add_argument('--MCSizeIn', type=int,
                    default=int(6e4), help='Size of MC input') #Zufallspunkte im Inneren
parser.add_argument('--MCsizeB', type=int, default=int(5e3)
                    , help='Size of MC boundary')  #Zufallspunkte auf dem Rand
parser.add_argument('--LearningRateStart', type=float,
                    default=1e-3, help='Initial learning rate')
parser.add_argument('--DecayRate', type=float, default=0.5,
                    help='Decay rate for learning rate')
parser.add_argument('--Epoach', type=int, default=int(9e4), #Iterationen
                    help='Number of epochs')
parser.add_argument('--EpoachDecay', type=int,
                    default=int(6000), help='Epochs before decay')
parser.add_argument('--alpha', type=float, default=100, help='Alpha parameter')
parser.add_argument('--mu', type=float, default=2, help='Mu parameter')

args = parser.parse_args()

# Use a shorter timeout and avoid async dir creation.
if os.name == 'nt':
    try:
        checkpointer = ocp.StandardCheckpointer(
            async_options=ocp.options.AsyncOptions(
                timeout_secs=60,
                create_directories_asynchronously=False,
            )
        )
    except Exception:
        # Fall back to default checkpointer from header
        pass

Width = int(args.Width)
NumLayer = int(args.NumLayer)
seed = int(args.Seed) if args.Seed is not None else secrets.randbelow(2**31 - 1)
hdr.key = random.PRNGKey(seed)
key = hdr.key
run_id = args.RunId or Timetxt
run_id = str(run_id).strip().replace(' ', '_').rstrip('_')
save_every = int(args.SaveEvery)
delete_metric_ckpts = bool(int(args.DeleteMetricCkpts))

MCSizeIn = args.MCSizeIn
MCsizeB = args.MCsizeB
LearningRateStart = args.LearningRateStart
DecayRate = args.DecayRate
Epoach = args.Epoach
EpoachDecay = args.EpoachDecay
alpha = args.alpha
mu = args.mu

lam = 0.01
DimInput = 2
Activation = nn.tanh

# Run root: groups many (Width,Layer) configs
run_root = project_root / 'Run' / f'{run_id}'
figs_dir = run_root / 'figs'
config_dir = run_root / 'data' / f'W{Width}_L{NumLayer}'

config_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
figs_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

def yData(ynn: Function, x: Tensor, para) -> Array:
    return (1.0+4*lam*(pi**4))*np.prod(np.sin(x*pi), axis=1).reshape(-1, 1)


def LossPinn(ynn: Function, lapY: Function, pnn: Function, _x: Tensor, para) -> Array:
    # |A|=3/4 for L-domain: MC mean -> integral
    return (3.0/4.0) * L2Norm(((lapY(_x, para['yNet'])).reshape(-1, 1))-((1.0/lam)*pnn(_x, para['pNet']).reshape(-1, 1)))


def LossP(ynn: Function, pnn: Function, lapP: Function, _x: Tensor, para) -> Array:
    _laplaceP = lapP(_x, para['pNet']).reshape(-1, 1)
    # |A|=3/4 for L-domain: MC mean -> integral
    return (3.0/4.0) * L2Norm(_laplaceP+ynn(_x, para['yNet']).reshape(-1, 1)-yData(ynn, _x, para))

''' Boundary losses for y and p separately (L2 norm over exact Solution) '''
def LossBoundaryP(fnn: Function, para, Xb_all: Array) -> Array:

    return L2Norm(fnn(Xb_all, para)+lam*2*pi**2*np.sin(pi*Xb_all[:, 0:1])*np.sin(pi*Xb_all[:, 1:2]))

def LossBoundaryY(fnn: Function, para, Xb_all: Array) -> Array:

    return L2Norm(fnn(Xb_all, para)-np.sin(pi*Xb_all[:, 0:1])*np.sin(pi*Xb_all[:, 1:2]))

def LossAll(ynn, pnn, lapY, lapP, paras, X_interior: Tensor, Xb_all: Array):
    """Total loss using provided interior samples and boundary samples for y and p."""
    return (
        LossPinn(ynn, lapY, pnn, X_interior, paras)
        + 10 * LossP(ynn, pnn, lapP, X_interior, paras)
        + alpha * (LossBoundaryY(ynn, paras['yNet'], Xb_all) + 10 * LossBoundaryP(pnn, paras['pNet'], Xb_all))
    )


def LossJ(ynn: NN, unn: NN, Paras: Any, X_interior: Tensor) -> Array:
    """Objective evaluated on provided interior samples."""
    # |Omega|=3/4 for L-domain: MC mean -> integral
    return 0.5 * (3.0/4.0) * (
        L2Norm(ynn(X_interior, Paras['yNet']) - yData(ynn, X_interior, Paras))
        + lam * L2Norm(unn(X_interior, Paras['pNet']))
    )


# %%
yNet, yPara = CreateNN(MLP, DimInput, 1, NumLayer, Width, Activation)
pNet, pPara = CreateNN(MLP, DimInput, 1, NumLayer, Width, Activation)
Paras = {'yNet': yPara, 'pNet': pPara}


def ynn(x, para): return yNet.apply(para, x)
def pnn(x, para): return pNet.apply(para, x)
def unn(x, para): return -(1.0/lam)*pnn(x, para)


LapY = CreateLaplaceNN(ynn, DimInput)
LapP = CreateLaplaceNN(pnn, DimInput)
lr_decay_fn = optax.exponential_decay(
    init_value=LearningRateStart,
    transition_steps=EpoachDecay,
    decay_rate=DecayRate
)

optimizer = optax.adam(
    learning_rate=lr_decay_fn)
opt = optimizer.init(Paras)
# %%%

# %%
# Sampling utilities
def sample_interior(rng_key: PRNGKey, n: int, dim: int) -> Tensor:
    """
    Sample points from A = {x ∈ [0,1]^2 : x₁ ≤ 1/2 or x₂ ≤ 1/2}.
    Rejection sampling: keep only points where at least one of x₁, x₂ is ≤ 1/2.
    """
    samples = []
    remaining = n
    current_key = rng_key

    while remaining > 0:
        # Generate 1.4 times more samples than needed to reduce rejection iterations
        current_key, subkey = random.split(current_key)
        candidates = random.uniform(subkey, (int(remaining * 1.4), dim))

        # Keep points where x₁ ≤ 1/2 OR x₂ ≤ 1/2
        mask = (candidates[:, 0] <= 0.5) | (candidates[:, 1] <= 0.5)
        valid = candidates[mask]

        samples.append(valid[:remaining])
        remaining -= len(valid[:remaining])

    return np.concatenate(samples, axis=0)[:n]


def sample_boundary_faces(rng_key: PRNGKey, n_per_face: int, dim: int) -> Array:
    """
    Sample boundary points for 2D L-shaped domain A = {x ∈ [0,1]^2 : x₁ ≤ 1/2 or x₂ ≤ 1/2}.
    
    6 boundary faces:
    1. x₁ = 0, x₂ ∈ [0,1] (left edge)
    2. x₁ ∈ [0,1/2], x₂ = 1 (top edge, left part)
    3. x₁ = 1/2, x₂ ∈ [1/2,1] (inner vertical corner)
    4. x₁ ∈ [1/2,1], x₂ = 1/2 (inner horizontal corner)
    5. x₁ = 1, x₂ ∈ [0,1/2] (right edge, bottom part)
    6. x₁ ∈ [0,1], x₂ = 0 (bottom edge)
    
    Returns: Array of shape (total_boundary_points, 2)
    """
    subkeys = random.split(rng_key, 6)
    faces = []
    n_per_face=n_per_face//4

    # Face 1: x₁ = 0, x₂ ∈ [0,1]
    xb = random.uniform(subkeys[0], (n_per_face, dim))
    xb = xb.at[:, 0].set(0.0)
    faces.append(xb)
    
    # Face 2: x₁ ∈ [0,1/2], x₂ = 1
    xb = random.uniform(subkeys[1], (n_per_face//2, dim))
    xb = xb.at[:, 1].set(1.0)
    xb = xb.at[:, 0].set(0.5 * xb[:, 0])  # scale x₁ to [0, 1/2]
    faces.append(xb)

    ''' Inner Edge Faces half sized samples'''
    # Face 3: x₁ = 1/2, x₂ ∈ [1/2,1]
    xb = random.uniform(subkeys[2], (n_per_face//2, dim))
    xb = xb.at[:, 0].set(0.5)
    xb = xb.at[:, 1].set(0.5 + 0.5 * xb[:, 1])  # scale x₂ to [1/2, 1]
    faces.append(xb)
    
    # Face 4: x₁ ∈ [1/2,1], x₂ = 1/2
    xb = random.uniform(subkeys[3], (n_per_face//2, dim))
    xb = xb.at[:, 1].set(0.5)
    xb = xb.at[:, 0].set(0.5 + 0.5 * xb[:, 0])  # scale x₁ to [1/2, 1]
    faces.append(xb)

    # Face 5: x₁ = 1, x₂ ∈ [0,1/2]
    xb = random.uniform(subkeys[4], (n_per_face//2, dim))
    xb = xb.at[:, 0].set(1.0)
    xb = xb.at[:, 1].set(0.5 * xb[:, 1])  # scale x₂ to [0, 1/2]
    faces.append(xb)
    
    # Face 6: x₁ ∈ [0,1], x₂ = 0
    xb = random.uniform(subkeys[5], (n_per_face, dim))
    xb = xb.at[:, 1].set(0.0)
    faces.append(xb)
    
    # Concatenate all faces into one array
    return np.concatenate(faces, axis=0)


# JIT-compiled losses that take samples explicitly
lossFn = jit(lambda _para, Xin, Xb: LossAll(ynn, pnn, LapY, LapP, _para, Xin, Xb))
gradFn = jit(value_and_grad(lossFn, argnums=0))
BoundaryFn = jit(
    lambda _para, Xb: alpha
    * (
        LossBoundaryY(ynn, _para['yNet'], Xb)
        + 10 * LossBoundaryP(pnn, _para['pNet'], Xb)
    )
)

Pfn = jit(lambda _para, Xin: 10 * LossP(ynn, pnn, LapP, Xin, _para))
PinnFn = jit(lambda _para, Xin: LossPinn(ynn, LapY, pnn, Xin, _para))
Jfn = jit(lambda _para, Xin: LossJ(ynn, unn, _para, Xin))
# %%
LstLoss = [0.0]*Epoach
LstP = [0.0]*Epoach
LstPinn = [0.0]*Epoach
LstBoundary = [0.0]*Epoach
LstJ = [0.0]*Epoach
# %%
ProcessBar = tqdm(range(Epoach), smoothing=0)

for idx in ProcessBar:
    key, k_interior, k_boundary = random.split(key, 3)
    X_interior = sample_interior(k_interior, MCSizeIn, DimInput)
    Xb_all = sample_boundary_faces(k_boundary, MCsizeB, DimInput)
    value, grads = gradFn(Paras, X_interior, Xb_all)
    updates, opt = optimizer.update(grads, opt)
    ProcessBar.set_postfix(Loss=value)
    Paras = optax.apply_updates(Paras, updates)
    LstLoss[idx] = value
    LstP[idx] = Pfn(Paras, X_interior)
    LstPinn[idx] = PinnFn(Paras, X_interior)
    LstBoundary[idx] = BoundaryFn(Paras, Xb_all)
    LstJ[idx] = Jfn(Paras, X_interior)

    if save_every > 0 and idx % save_every == 0:
        checkpointer.save(config_dir / f'{idx}', Paras)
# LapP(x)
# %%
final_dir = config_dir / 'final'
if final_dir.exists():
    import shutil
    shutil.rmtree(final_dir, ignore_errors=True)
checkpointer.save(final_dir, Paras)
checkpointer.wait_until_finished()

# Save a small summary
summary = {
    'run_id': run_id,
    'width': Width,
    'num_layer': NumLayer,
    'epoach': int(Epoach),
    'seed': int(seed),
    'mc_size_in': int(MCSizeIn),
    'mc_size_b': int(MCsizeB),
    'learning_rate_start': float(LearningRateStart),
    'decay_rate': float(DecayRate),
    'epoach_decay': int(EpoachDecay),
    'alpha': float(alpha),
    'mu': float(mu),
    'final': {
        'Loss': float(np.asarray(LstLoss[-1])),
        'P': float(np.asarray(LstP[-1])),
        'Pinn': float(np.asarray(LstPinn[-1])),
        'Boundary': float(np.asarray(LstBoundary[-1])),
        'J': float(np.asarray(LstJ[-1])),
    },
}
(config_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

# %%
# Metric checkpoints
try:
    checkpointer.save(config_dir / 'Loss', {'Loss': np.array(LstLoss)})
    checkpointer.wait_until_finished()
    checkpointer.save(config_dir / 'P', {'P': np.array(LstP)})
    checkpointer.wait_until_finished()
    checkpointer.save(config_dir / 'Pinn', {'Pinn': np.array(LstPinn)})
    checkpointer.wait_until_finished()
    checkpointer.save(config_dir / 'Boundary', {'Boundary': np.array(LstBoundary)})
    checkpointer.wait_until_finished()
    checkpointer.save(config_dir / 'J', {'J': np.array(LstJ)})
    checkpointer.wait_until_finished()
except PermissionError as e:
    print(f'Warning: Orbax metric checkpoint save failed (PermissionError).')
    print(f'  {e}')
    print('  Continuing; summary.json and final checkpoint are already written.')
except Exception as e:
    print(f'Warning: Orbax metric checkpoint save failed: {e}')
    print('  Continuing; summary.json and final checkpoint are already written.')
# %%
print('Creating plots (checkpoint + metrics) headlessly...')

examples_dir = Path(__file__).resolve().parent
plot_checkpoint_py = examples_dir / 'plot_checkpoint.py'
plot_metrics_py = examples_dir / 'plot_metrics.py'

# Fixed plot scales (as requested)
subprocess.run(
    [
        sys.executable,
        str(plot_checkpoint_py),
        '--run_dir', str(config_dir),
        '--figs_dir', str(figs_dir),
        '--run_name', str(run_id),
        '--width', str(Width),
        '--num_layer', str(NumLayer),
        '--RunId', str(run_id),
        '--MCSizeIn', str(MCSizeIn),
        '--MCsizeB', str(MCsizeB),
        '--Epoach', str(Epoach),
        '--LearningRateStart', str(LearningRateStart),
        '--DecayRate', str(DecayRate),
        '--EpoachDecay', str(EpoachDecay),
        '--no_gui',
        '--u_min', '0', '--u_max', '20',
        '--y_min', '0', '--y_max', '1',
        '--err_u_min', '0', '--err_u_max', '0.1',
        '--err_y_min', '0', '--err_y_max', '0.01',
        '--draw_box',
        '--l_domain',
        '--err_scale', 'linear',
    ],
    check=True,
)

subprocess.run(
    [
        sys.executable,
        str(plot_metrics_py),
        '--run_dir', str(config_dir),
        '--figs_dir', str(figs_dir),
        '--run_name', str(run_id),
        '--width', str(Width),
        '--num_layer', str(NumLayer),
        '--log_min_exp', '-6',
        '--log_max_exp', '2',
        '--no_gui',
    ],
    check=True,
)

if delete_metric_ckpts:
    import shutil
    for name in ['Loss', 'P', 'Pinn', 'Boundary', 'J']:
        shutil.rmtree(config_dir / name, ignore_errors=True)

print(f'✓ Run complete: {run_root} | config: {config_dir.name}')
