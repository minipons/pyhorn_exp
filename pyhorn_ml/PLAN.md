# pyhorn-ml: Machine Learning for Horn Speaker Design

## Vision

Use ML to close the loop between horn geometry parameters and acoustic target. Instead of manually iterating (simulate → tweak → repeat), pyhorn-ml searches the parameter space automatically and returns Pareto-optimal designs that best match your targets.

**The iteration loop becomes:**
```
You: "I want flat response down to 80Hz with max sensitivity"
pyhorn-ml: runs 200 simulations in ~3 min, returns 5 designs ranked by target match
```

---

## What Problem Are We Solving?

Designing a horn loudspeaker by hand involves choosing:
- Throat area (driver coupling)
- Mouth area (loading / cutoff)
- Path length (bass extension)
- Expansion profile (exponential, conical, hyperbolic...)
- Folding / segment layout (affects group delay, throat stress)

These 5–20 parameters interact in non-linear ways. Even experienced designers run 20–50 simulations to find a good configuration.

**ML makes this faster by:**
1. Building a **surrogate model** that predicts SPL response from parameters (~100x faster than full simulation)
2. Using **Bayesian optimization** to search efficiently (not random search)
3. Handling **multi-objective trade-offs** automatically (flat response vs. sensitivity vs. bass extension)

---

## Approach: Bayesian Optimization + Multi-Objective Search

### Why Bayesian over random search?

Random search samples the space uniformly. Bayesian optimization builds a probabilistic model (Gaussian Process) of "parameter → score" and actively chooses informative points — typically finds 3–5x better solutions in the same budget of simulations.

### Why multi-objective?

Acoustic targets conflict:
- **Flat response** ↔ **Maximum sensitivity** (higher sensitivity often means more ripple)
- **Deep bass** ↔ **Compact size** (longer path = lower cutoff = bigger cabinet)
- **Low group delay** ↔ **Maximum output** (straight throat vs. folded)

NSGA-II or BO-based Pareto front gives you the trade-off curve, not a single point.

---

## Integration with pyhorn

pyhorn-ml is a **thin layer above pyhorn**:

```
pyhorn-ml                    pyhorn
┌─────────────────────┐     ┌──────────────────┐
│ SurrogateModel      │     │ solver/models.py │
│ MultiObjectiveOpt   │ ──► │ solver/profiles │
│ TargetResponse      │     │ solver/medial_axis │
│ DesignPoint         │     └──────────────────┘
└─────────────────────┘
```

pyhorn-ml calls `pyhorn.solver.models.horn_response()` as the **fitness function** during optimization. The surrogate model is trained on pyhorn simulation data and used to accelerate search.

---

## Core Concepts

### TargetResponse

What you want the horn to do:

```python
@dataclass
class TargetResponse:
    f_min: float           # Lowest frequency to evaluate (Hz)
    f_max: float           # Highest frequency to evaluate (Hz)
    flatness: float        # 0.0–1.0, weight for minimizing ripple
    sensitivity: float     # 0.0–1.0, weight for max SPL in band
    bass_extension: float  # 0.0–1.0, weight for low cutoff
    max_size_m: float      # Physical constraint: max path length (m)
    target_spl: float      # Target mid-band SPL (dB)
```

### DesignSpace

The parameters to search over:

```python
@dataclass
class DesignSpace:
    throat_area_m2: Tuple[float, float]        # (min, max)
    mouth_area_m2: Tuple[float, float]          # (min, max)
    path_length_m: Tuple[float, float]          # (min, max)
    profile_type: List[str]                     # ["conical", "exponential", ...]
    n_segments: Tuple[int, int]                 # (min, max)
    fold_style: List[str]                       # ["straight", "W", "pi", ...]
```

### DesignPoint

A candidate design + its predicted response:

```python
@dataclass
class DesignPoint:
    geometry: HornGeometry
    driver: DriverSpecs
    predicted_spl: np.ndarray        # frequency → SPL (dB)
    predicted_z: np.ndarray           # frequency → impedance (Ohms)
    score: float                      # scalar fitness score
    dominated: bool                   # True if another design beats it on all objectives
```

### SurrogateModel

Trained on pyhorn simulation data. Used to give fast (~1ms) response predictions during optimization, so we can run thousands of virtual evaluations per second.

```python
class SurrogateModel:
    def fit(X: np.ndarray, y_spl: np.ndarray) -> None:
        """Train on N designs × M frequencies → SPL values"""

    def predict(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict SPL + uncertainty for new designs"""
        # Returns (mean_prediction, uncertainty)
```

### MultiObjectiveOptimizer

```python
class BayesianOptimizer:
    def __init__(self, design_space: DesignSpace, n_initial: int = 20):
        """Run n_initial random sims, fit GP, then optimize"""

    def ask(self) -> DesignPoint:
        """Return next design to evaluate (uses acquisition function)"""

    def tell(self, design: DesignPoint, real_spl: np.ndarray) -> None:
        """Update GP with real simulation result"""

    def run(self, budget: int = 200) -> List[DesignPoint]:
        """Run optimization for budget evaluations, return Pareto front"""
```

---

## CLI Interface

```bash
# Basic usage: optimize a BK16-style horn for flat response
pyhorn-ml optimize \\
    --driver pyhorn/fostex.yaml \\
    --target flatness=0.8 sensitivity=0.5 bass_extension=0.6 \\
    --fmin 80 --fmax 5000 \\
    --budget 150 \\
    --output outputs/ml/bk16_optimized

# Multi-objective: show Pareto front
pyhorn-ml optimize \\
    --driver pyhorn/fostex.yaml \\
    --target flatness=1.0 sensitivity=1.0 bass_extension=1.0 max_size_m=1.2 \\
    --mode pareto \\
    --budget 300 \\
    --output outputs/ml/pareto_bk16

# Train a custom surrogate model from your own simulation data
pyhorn-ml train-surrogate \\
    --data-dir outputs/history/bk16_runs/ \\
    --model-type mlp \\
    --save-as models/my_bk16_surrogate.pkl

# Inverse design: "give me a horn that looks like this target curve"
pyhorn-ml inverse \\
    --driver pyhorn/fostex.yaml \\
    --target-curve outputs/ref平坦.response.csv \\
    --budget 100

# Compare: ML-optimized vs manual design vs Hornresp reference
pyhorn-ml compare \\
    --designs outputs/ml/bk16_optimized/design_001.yaml \\
              source/bk16.yaml \\
    --driver pyhorn/fostex.yaml \\
    --plot outputs/ml/compare.png
```

---

## File Structure

```
pyhorn_ml/
├── __init__.py
├── pyproject.toml              # Depends on: pyhorn, scikit-learn, scipy, numpy, jinja2
│
├── cli/
│   ├── __init__.py
│   └── commands.py             # typer CLI: optimize, train-surrogate, inverse, compare
│
├── core/
│   ├── __init__.py
│   ├── target.py               # TargetResponse dataclass + parsing
│   ├── space.py                # DesignSpace + discrete parameter encoding
│   ├── design_point.py         # DesignPoint dataclass
│   └── score.py                # Fitness scoring from predicted SPL
│
├── surrogate/
│   ├── __init__.py
│   ├── base.py                 # SurrogateModel abstract base
│   ├── gp.py                   # GaussianProcessSurrogate (scikit-learn GaussianProcessRegressor)
│   └── mlp.py                  # MLPSurrogate (scikit-learn MLPRegressor)
│
├── optimization/
│   ├── __init__.py
│   ├── base.py                 # Optimizer base class
│   ├── bayesian.py             # BayesianOptimizer (scikit-optimize / GPflow)
│   ├── nsga2.py                # NSGA2Optimizer for true multi-objective Pareto
│   └── acquisition.py          # Acquisition functions: UCB, EI, PI
│
├── pipeline/
│   ├── __init__.py
│   ├── evaluate.py             # Runs pyhorn.horn_response(), returns SPL + Z
│   ├── optimize.py             # Main optimize() loop (ask/tell interface)
│   └── pareto.py               # Pareto front extraction + ranking
│
├── data/
│   ├── __init__.py
│   ├── dataset.py              # DesignDataset: collects + stores simulation results
│   └── io.py                   # Save/load designs, export to YAML
│
├── plot/
│   ├── __init__.py
│   ├── pareto.py               # Pareto front visualization
│   ├── response_overlay.py    # Overlay predicted vs. real SPL curves
│   └── convergence.py           # Optimization convergence plot
│
└── templates/
    └── design_report.md        # Jinja2 report template for optimization results
```

---

## Dependency Plan

```toml
# pyproject.toml for pyhorn_ml
[project]
name = "pyhorn-ml"
version = "0.1.0"
description = "ML-driven horn speaker design optimization built on pyhorn"
requires-python = ">=3.10"
dependencies = [
    "pyhorn>=0.1.0",          # Core simulator (published separately)
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-learn>=1.3",       # MLPRegressor, StandardScaler, model persistence
    "scikit-optimize>=0.9",    # Bayesian optimization (gp_minimize)
    "matplotlib>=3.7",
    "pyyaml>=6.0",
    "typer>=0.9",
    "jinja2>=3.1",
    "joblib>=1.3",             # Model serialization
]
```

Note: `pyhorn` is listed as a dependency, not installed from PyPI yet — it comes from the local editable install during development. Eventually pyhorn-ml requires pyhorn to be installed first.

---

## Algorithm: How the Optimization Loop Works

### Phase 1: Initial Space-Filling Sample
```
for i in range(n_initial_designs):
    design = random_design(design_space)          # Latin Hypercube Sampling
    spl, z = pyhorn.solve(design)                 # real simulation
    dataset.add(design, spl, z)                   # store
    surrogate.fit(dataset)                         # retrain GP
```

### Phase 2: Bayesian Optimization
```
for i in range(budget - n_initial):
    design = acq_max(surrogate, design_space)     # Expected Improvement
    spl, z = pyhorn.solve(design)                 # real simulation
    dataset.add(design, spl, z)
    surrogate.fit(dataset)
    pareto.update(design, score)                  # maintain Pareto front
```

### Phase 3: Results
```
pareto_front = pareto.front()                     # non-dominated designs
for d in pareto_front:
    save_to_yaml(d)
generate_report(pareto_front)
plot_pareto(pareto_front)
```

---

---

## What pyhorn-ml Does NOT Do

### GP vs MLP surrogate?

**Gaussian Process** — smaller data regime (50–200 sims), gives uncertainty estimates, but slow to predict (O(n²) in dataset size). Good for:
- First 100–200 simulations when data is scarce
- When you need uncertainty for acquisition function

**MLP** — requires more data (500+ sims) but predicts in ~1ms regardless of training size. Good for:
- After you have accumulated simulation history
- Inverse design (many forward passes)

**Plan**: Start with GP, switch to MLP when dataset > 500. Auto-switch in `SurrogateModel`.

### Why not reinforcement learning?

RL makes sense when:
- Action space is continuous and smooth
- Episode length is short (like a game move)
- You can run millions of cheap episodes

Horn design has ~20 parameters, each simulation takes ~100ms, and you're happy with 100–300 total. Bayesian optimization is the right tool here — RL would require 10,000+ episodes to be competitive.

### Offline-only training?

Initially yes — pyhorn-ml learns from pyhorn simulation data generated during optimization. Future version could fine-tune a foundation model on Hornresp reference curves (if Guillaume has validation data).

---

## Scoring Function (How We Rank Designs)

For a given SPL curve `S(f)` and target `T`:

```python
def score_design(spl, target: TargetResponse, freq) -> tuple:
    # Flatness: 1 - normalized std deviation in [f_min, f_max]
    band_mask = (freq >= target.f_min) & (freq <= target.f_max)
    band_spl = spl[band_mask]
    flatness_score = 1.0 - np.std(band_spl) / target.target_spl

    # Sensitivity: closeness to target SPL in the horn-loading band
    # Measured in [max(f_min×2.5, 200), f_max] to exclude the driver's
    # free-air resonance tail. Below ~200Hz the SPL is dominated by the
    # driver's Qt/vas roll-off, not the horn geometry — measuring there
    # would reward SPL peaks created by high compression ratios rather
    # than genuine loading efficiency.
    sens_band_lo = max(target.f_min * 2.5, 200.0)
    sens_band_mask = (freq >= sens_band_lo) & (freq <= target.f_max)
    sensitivity_score = 1.0 - abs(np.mean(spl[sens_band_mask]) - target.target_spl) / target.target_spl

    # Bass extension: bonus for response down to f_min with < 12dB rolloff
    bass_mask = (freq >= target.f_min) & (freq <= target.f_min * 2.0)
    bass_score = 1.0 - max(0, np.mean(spl[bass_mask]) - (target.target_spl - 12.0)) / 12.0

    # Hard cutoff constraint: penalise if -3dB cutoff > f_min
    f_c = _cutoff_frequency(spl, freq, target.f_min)  # estimate -3dB point
    cutoff_penalty = _cutoff_penalty(f_c, target.f_min)  # 1.0=pass, 0.0=fail

    total = (target.flatness * flatness_score
           + target.sensitivity * sensitivity_score
           + target.bass_extension * bass_score)
    total /= (target.flatness + target.sensitivity + target.bass_extension)
    total *= cutoff_penalty  # hard constraint zeroes out bad designs

    return flatness_score, sensitivity_score, bass_score, cutoff_penalty, total
```

The cutoff constraint uses a piecewise penalty:
- `f_c ≤ f_min` → 1.0 (passes, no penalty)
- `f_min < f_c < f_min × 1.5` → linear falloff to 0
- `f_c ≥ f_min × 1.5` → 0.0 (fails, score zeroed)

---

## Validation Strategy

Before shipping any ML component:

1. **Surrogate accuracy**: Compare GP predictions against held-out 20% of simulation data. Require R² > 0.85 on validation set.
2. **Optimization sanity**: Run 50-design budget on a simple 2D problem (e.g., just throat area × path length) and verify optimizer finds the known optimum.
3. **Pareto dominance**: Verify that every design on the Pareto front actually dominates designs not on it.
4. **Against Hornresp** (when Onshape integration is stable): Compare pyhorn-ml optimized designs against Guillaume's hand-tuned reference.

---

## Milestones

### v0.1 — MVP (fastest path to useful)
1. `DesignSpace`, `TargetResponse`, `DesignPoint` dataclasses
2. `pyhorn.solver.models.horn_response()` wrapper in `evaluate.py`
3. Latin Hypercube random sampling with sequential evaluation
4. Basic Pareto front extraction
5. CLI: `pyhorn-ml optimize` → saves YAML designs to output dir
6. Plot: SPL overlay of top-5 designs

### v0.2 — Surrogate-accelerated search
7. `GaussianProcessSurrogate` using scikit-learn
8. `BayesianOptimizer` with Expected Improvement acquisition
9. Auto-switch to MLP when dataset > 500
10. Convergence plot: score vs. evaluation number

### v0.3 — Inverse design
11. `InverseDesigner`: given target curve, find parameters via gradient descent on surrogate
12. `pyhorn-ml inverse` CLI command

### v0.4 — Design database
13. `DesignDatabase`: SQLite-backed store of all evaluated designs with metadata
14. `pyhorn-ml search`: find similar designs to a reference curve
15. Cluster analysis: group designs by response shape

---

## Implementation Status (2026-04-28)

### Done
- ✅ Chamber volumes: `lrc` (m) and `vtc` (m³) added to DesignSpace
  - 11 dimensions total (was 12 with one-hot fold encoding): 5 continuous
    + 3 profile one-hot + 1 n_segments + 2 fold acoustic features
  - `vrc = lrc × throat_area` computed in `build_horn_geometry()`
- ✅ **Cutoff penalty: replaced binary 0/1 with soft sigmoid decay.**
  Old: `f_c >= f_min × 1.5 → penalty = 0` (hard binary cut)
  New: sigmoid falloff from 1.0 at `f_c = f_min` to 0.0 at `f_c = f_min × 2.5`,
  preserving designs that are slightly over cutoff but otherwise well-shaped.
- ✅ **Fold style: replaced one-hot with continuous acoustic features.**
  `(bend_count, mean_angle)` → maps to actual acoustic effects (inertance, corner volume).
  Surrogate can now interpolate smoothly across fold geometries.
  - `straight`: (0.0, 0.0)
  - `W`: (2.0, 1.57) — two 90° turns
  - `pi`: (3.0, 2.36) — three ~120° turns
- ✅ **Inverse design: connected surrogate-guided inversion pipeline.**
  - Mode 1 (surrogate available): Nelder-Mead on surrogate loss landscape,
    then real pyhorn validation on best result. Warm-starts from best
    design in existing dataset if available.
  - Mode 2 (no surrogate): Differential Evolution on real simulations
    (slow but functional — can be used without a trained model).
  - Result always validated with real pyhorn simulation and scored via
    the standard scoring pipeline.
- ✅ Sensitivity band: measured in `[max(f_min×2.5, 200), f_max]` — above the driver's
  roll-off tail where the horn loading does real work, not in the Qt/vas resonance region
  that rewards artificial midband peaking from high compression ratios
- ✅ `DesignPoint.to_dict()` includes `cutoff_penalty`, `lrc_m`, `vtc_m3`
- ✅ `DesignSpace.n_dims()` updated to 11 (from 12) to reflect fold encoding change

### Remaining gaps
- 🔲 vrc (rear chamber volume) currently derived from `lrc × throat_area`; explicit vrc param would give more control
- 🔲 MLP surrogate has no native uncertainty estimate — GP remains primary until MLP data is abundant
- 🔲 Validation against Hornresp reference curves (Onshape integration stable enough for this?)

---

## What pyhorn-ml Does NOT Do

- It does **not** replace the acoustic solver — it wraps it
- It does **not** design the driver — it works with whatever driver specs you provide
- It does **not** do structural analysis (cabinet resonance, port chuffing, etc.)
- It does **not** require a GPU — all models run on CPU
- It does **not** need cloud compute — everything runs locally

---

## Prior Work Worth Reading

- J. Bradley, "Optimization of Horn Loudspeakers Using Genetic Algorithms" (AES 2003)
- D. Smith, "Horn Loudspeaker Design Using Multi-Objective Optimization" (AES 2014)
- G. Randall, "Machine Learning for Accelerating loudspeaker Design" (IEEE 2021)
- scikit-optimize docs: https://scikit-optimize.github.io
- GPyTorch for larger GP models if needed: https://gpytorch.ai
