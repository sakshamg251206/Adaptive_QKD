"""Command-line interface: ``adaptive-qkd <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence

from adaptive_qkd.config import CLASS_NAMES, DEFAULT_SEED, Paths

logger = logging.getLogger("adaptive_qkd")


def _cmd_simulate(args: argparse.Namespace) -> None:
    from adaptive_qkd.features import extract_features
    from adaptive_qkd.simulator import BB84Simulator, ChannelParams

    params = ChannelParams(noise_prob=args.noise, intercept_prob=args.intercept, pns_prob=args.pns)
    trace = BB84Simulator(num_qubits=args.qubits, seed=args.seed, backend=args.backend).simulate(params)
    features = extract_features(trace)
    print(f"Scenario: {trace.label}  (backend: {trace.backend}, {trace.num_pulses} pulses)")
    print(f"Sifted bits: {trace.sifted_length}  test bits: {trace.test_length}  key bits: {trace.key_length}")
    for name, value in features.items():
        print(f"  {name:<20} {value:.4f}")


def _cmd_generate(args: argparse.Namespace) -> None:
    from adaptive_qkd.dataset import generate_dataset, save_dataset

    paths = Paths.from_env()
    start = time.perf_counter()
    dataset = generate_dataset(sessions_per_class=args.sessions_per_class, seed=args.seed)
    save_dataset(dataset, paths)
    counts = dataset.full_sessions["label"].value_counts().reindex(CLASS_NAMES)
    logger.info("Wrote %d sessions to %s in %.1fs", len(dataset.sequences), paths.data_dir, time.perf_counter() - start)
    logger.info("Class counts: %s", ", ".join(f"{c}={n}" for c, n in counts.items()))


def _cmd_train(args: argparse.Namespace) -> None:
    from adaptive_qkd.training import train_models

    paths = Paths.from_env()
    train_models(paths, include_lstm=not args.no_lstm, lstm_epochs=args.epochs, seed=args.seed)
    logger.info("Models saved to %s, reports to %s", paths.model_dir, paths.results_dir)


def _cmd_benchmark(args: argparse.Namespace) -> None:
    from adaptive_qkd.benchmark import plot_benchmark, render_benchmark_markdown, run_benchmark, summarize
    from adaptive_qkd.protocol import AdaptiveQKDProtocol

    paths = Paths.from_env()
    protocol = AdaptiveQKDProtocol.from_paths(paths)
    if not protocol.uses_model:
        logger.warning("Benchmarking the QBER heuristic; run 'adaptive-qkd train' for the ML policy")
    summary = summarize(run_benchmark(protocol, sessions_per_scenario=args.sessions))
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(paths.results_dir / "benchmark_summary.csv", index=False)
    (paths.results_dir / "benchmark.md").write_text(render_benchmark_markdown(summary, args.sessions))
    plot_benchmark(summary, paths.plots_dir / "benchmark.png")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def _cmd_pipeline(args: argparse.Namespace) -> None:
    _cmd_generate(args)
    _cmd_train(args)
    _cmd_benchmark(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptive-qkd",
        description="Simulate BB84, train eavesdropping classifiers and benchmark the adaptive protocol.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="simulate one session and print its features")
    sim.add_argument("--qubits", type=int, default=1000, help="pulses sent by Alice (default: 1000)")
    sim.add_argument("--noise", type=float, default=0.0, help="depolarizing probability (default: 0)")
    sim.add_argument("--intercept", type=float, default=0.0, help="intercept-resend fraction (default: 0)")
    sim.add_argument("--pns", type=float, default=0.0, help="PNS attack strength (default: 0)")
    sim.add_argument("--backend", choices=("numpy", "qiskit"), default="numpy")
    sim.add_argument("--seed", type=int, default=None)
    sim.set_defaults(func=_cmd_simulate)

    def add_generate_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--sessions-per-class", type=int, default=2000, help="default: 2000")

    def add_train_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--no-lstm", action="store_true", help="train only the Random Forest")
        p.add_argument("--epochs", type=int, default=25, help="LSTM training epochs (default: 25)")

    def add_benchmark_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--sessions", type=int, default=200, help="sessions per scenario (default: 200)")

    gen = sub.add_parser("generate", help="generate the labelled dataset")
    add_generate_args(gen)
    gen.set_defaults(func=_cmd_generate)

    train = sub.add_parser("train", help="train and evaluate the classifiers")
    add_train_args(train)
    train.set_defaults(func=_cmd_train)

    bench = sub.add_parser("benchmark", help="compare defence policies")
    add_benchmark_args(bench)
    bench.set_defaults(func=_cmd_benchmark)

    pipe = sub.add_parser("pipeline", help="generate, train and benchmark in one go")
    add_generate_args(pipe)
    add_train_args(pipe)
    add_benchmark_args(pipe)
    pipe.set_defaults(func=_cmd_pipeline)

    for p in (gen, train, bench, pipe):
        p.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default: {DEFAULT_SEED})")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Third-party libraries (e.g. the Qiskit transpiler) stay at WARNING.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
