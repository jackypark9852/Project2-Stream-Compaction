#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot avg_ms vs n for each test_id (series per test). Y-axis is log scale.",
    )
    parser.add_argument("csv_path", help="Path to CSV/TSV file with columns including test_id, n, avg_ms")
    parser.add_argument("-o", "--output", help="Output PNG path (if omitted, show interactively)")
    parser.add_argument("--legend", default="right", choices=["right","best","none"],
                        help="Legend location (default: right)")
    args = parser.parse_args()

    # Read file; auto-detect delimiter (comma or tab)
    try:
        df = pd.read_csv(args.csv_path, sep=None, engine="python")
    except Exception as e:
        print(f"Failed to read file: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = {"test_id", "n", "avg_ms"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing required columns: {missing}", file=sys.stderr)
        sys.exit(2)

    # Coerce numeric types
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["avg_ms"] = pd.to_numeric(df["avg_ms"], errors="coerce")
    df = df.dropna(subset=["n", "avg_ms"])

    # Sort by n within each series for nice lines
    df = df.sort_values(["test_id", "n"])

    # Plot: one series per test_id
    plt.figure(figsize=(10, 6))
    for test_id, sub in df.groupby("test_id"):
        plt.plot(sub["n"], sub["avg_ms"], marker="o", linewidth=1, label=str(test_id))

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n (log scale)")
    plt.ylabel("avg_ms (log scale)")
    plt.title("Benchmark: avg_ms vs n by test_id")

    if args.legend != "none":
        loc = "center left" if args.legend == "right" else "best"
        bbox = (1.02, 0.5) if args.legend == "right" else None
        plt.legend(loc=loc, bbox_to_anchor=bbox, borderaxespad=0.)

    plt.tight_layout()

    if args.output:
        try:
            plt.savefig(args.output, dpi=150)
            print(f"Saved plot to {args.output}")
        except Exception as e:
            print(f"Failed to save figure: {e}", file=sys.stderr)
            sys.exit(3)
    else:
        plt.show()

if __name__ == "__main__":
    main()
