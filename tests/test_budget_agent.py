import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.agents.finance.budget_agent import BudgetAgent


def main():
    file_path = os.path.join(PROJECT_ROOT, "data", "planning", "sample_budget.csv")

    data = pd.read_csv(file_path)

    agent = BudgetAgent()

    result = agent.analyze(
        data=data,
        start_month="2026-04",
        end_month="2026-06",
        group_by="month",
    )

    print(result)


if __name__ == "__main__":
    main()