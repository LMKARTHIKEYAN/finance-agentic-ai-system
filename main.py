"""Local entry point for the Autonomous Finance Agentic AI System."""

from __future__ import annotations

import argparse

from src.api.dependencies import get_autonomous_finance_service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="CFO/finance goal")
    args = parser.parse_args()
    response = get_autonomous_finance_service().ask(" ".join(args.question))
    print(response.answer or response.question or response.status)


if __name__ == "__main__":
    main()
