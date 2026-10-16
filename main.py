import argparse
import contextlib
import logging
import sys
from pathlib import Path

class TeeStream:
    """Keep console output and append a copy to the run log."""

    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Reproduce Paper_Token.ipynb's English–Bengali tokenization analysis")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-sizes", type=int, nargs="+")
    parser.add_argument("--mode", choices=["benchmark", "reference"], default="benchmark")
    parser.add_argument("--pairs-dir", type=Path, help="Reuse a previous run's archived pairs with a different output directory")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verify-reference", action="store_true", help="Fail if fresh rounded results differ from the notebook")
    args = parser.parse_args()
    args.output = args.output or root / "outputs" / args.mode
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "run.log", mode="a", encoding="utf-8")])
    logging.info("Starting %s run; config=%s; output=%s", args.mode, args.config, args.output)
    try:
        with (args.output / "run.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(TeeStream(sys.stdout, log)):
            if args.mode == "reference":
                if args.pairs_dir or args.verify_reference:
                    parser.error("--pairs-dir and --verify-reference apply to benchmark mode only")
                from tokenization.reference import run_reference
                run_reference(args.config, args.output, args.sample_sizes, args.no_plots)
            else:
                from tokenization.pipeline import run
                run(args.config, args.output, args.sample_sizes, args.pairs_dir, args.no_plots, args.verify_reference)
        logging.info("Finished %s run", args.mode)
    except Exception:
        logging.exception("Benchmark failed; see run.log for details.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
