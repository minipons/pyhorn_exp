"""CLI entry point for pyhorn-ml — thinTyper app that delegates to run_* functions."""
import typer

app = typer.Typer(
    name="pyhorn-ml",
    help="ML-driven horn speaker design optimisation.",
    no_args_is_help=True,
)


@app.command()
def optimize(
    driver: str = typer.Option(..., "--driver", "-d", help="Path to driver YAML"),
    target: str = typer.Option(
        ..., "--target", "-t",
        help="Acoustic targets as key=val,key=val "
             "(e.g. flatness=0.8,sensitivity=0.5,bass_extension=0.6)"
    ),
    fmin: float = typer.Option(80.0, "--fmin", help="Minimum frequency Hz"),
    fmax: float = typer.Option(5000.0, "--fmax", help="Maximum frequency Hz"),
    budget: int = typer.Option(150, "--budget", "-n", help="Total simulation budget"),
    n_initial: int = typer.Option(
        20, "--n-initial",
        help="Number of initial random evaluations before Bayesian search"
    ),
    acquisition: str = typer.Option(
        "ei", "--acquisition",
        help="Acquisition function: ei (Expected Improvement) or ucb"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    output: str = typer.Option("outputs/ml", "--output", "-o", help="Output directory"),
) -> None:
    """Run Bayesian multi-objective optimisation to find the best horn designs.

    Searches the geometry parameter space using a Gaussian Process surrogate
    model and Expected Improvement acquisition. Returns a Pareto front of
    non-dominated designs ranked by flatness, sensitivity, and bass extension.

    Examples:

      pyhorn-ml optimize -d pyhorn/fostex.yaml -t flatness=0.8,sensitivity=0.5,bass_extension=0.6

      # More evaluations, custom frequency range
      pyhorn-ml optimize -d pyhorn/fostex.yaml -t flatness=1.0 --fmin 60 --fmax 3000 -n 300
    """
    from pyhorn_ml.cli.optimize import run_optimize
    run_optimize(
        driver=driver,
        target_str=target,
        fmin=fmin,
        fmax=fmax,
        budget=budget,
        output=output,
        n_initial=n_initial,
        acquisition=acquisition,
        seed=seed,
    )


@app.command()
def train_surrogate(
    data_dir: str = typer.Option(
        ..., "--data-dir",
        help="Directory with simulation results (from a previous optimise run)"
    ),
    model_type: str = typer.Option("gp", "--model-type", help="gp or mlp"),
    save_as: str = typer.Option("surrogate.pkl", "--save-as", help="Output model path"),
) -> None:
    """Train a surrogate model from existing optimisation data.

    Loads all designs from a previous optimise run's .dataset/ directory,
    fits a GP or MLP surrogate model, and saves it to a .pkl file.

    The saved model can be used to:
      - Make fast (~1ms) SPL predictions for any geometry in the space
      - Run batch analysis without calling pyhorn repeatedly
      - Seed a new optimisation with pre-trained weights
    """
    from pyhorn_ml.cli.train import run_train
    run_train(data_dir=data_dir, model_type=model_type, save_as=save_as)


@app.command()
def inverse(
    driver: str = typer.Option(..., "--driver", "-d"),
    target_curve: str = typer.Option(
        ..., "--target-curve",
        help="CSV with 'freq,spl' columns describing the desired response"
    ),
    budget: int = typer.Option(100, "--budget", "-n"),
    output: str = typer.Option("outputs/ml/inverse", "--output", "-o"),
    model_path: str = typer.Option(
        None, "--model",
        help="Path to pre-trained surrogate model (from train-surrogate). "
             "If omitted, uses random search."
    ),
) -> None:
    """Inverse design: find geometry parameters that reproduce a target SPL curve.

    Loads a pre-trained surrogate model and uses gradient-free optimisation
    (Nelder-Mead on the surrogate) to find the geometry that minimises
    the mean-squared error between predicted SPL and the target curve.

    Example:
      pyhorn-ml inverse -d pyhorn/fostex.yaml --target-curve target_response.csv -n 200
    """
    from pyhorn_ml.cli.inverse import run_inverse
    run_inverse(
        driver=driver,
        target_curve=target_curve,
        budget=budget,
        output=output,
        model_path=model_path,
    )


@app.command()
def compare(
    designs: str = typer.Option(
        ..., "--designs",
        help="Comma-separated list of YAML design files or directories"
    ),
    driver: str = typer.Option(..., "--driver", "-d"),
    output: str = typer.Option("outputs/ml/compare.png", "--output", "-o"),
    fmin: float = typer.Option(20.0, "--fmin"),
    fmax: float = typer.Option(5000.0, "--fmax"),
) -> None:
    """Compare multiple designs on one SPL plot.

    Loads one or more YAML designs (or a directory of designs), runs
    pyhorn simulations for each, and plots all SPL curves overlaid.

    Example:
      pyhorn-ml compare --designes outputs/ml/top/design_001_*.yaml -d pyhorn/fostex.yaml
    """
    from pyhorn_ml.cli.compare import run_compare
    run_compare(designs=designs, driver=driver, output=output, fmin=fmin, fmax=fmax)


if __name__ == "__main__":
    app()
