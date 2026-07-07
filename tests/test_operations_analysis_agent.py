import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.agents.analytics.operations_analysis_agent import OperationsAnalysisAgent


def main():
    file_path = os.path.join(PROJECT_ROOT, "data", "operations", "sample_orders.csv")

    data = pd.read_csv(file_path)

    agent = OperationsAnalysisAgent()

    result = agent.analyze(
        data=data,
        start_date=None,
        end_date=None,
        group_by="week",
    )

    print(result)


if __name__ == "__main__":
    main()