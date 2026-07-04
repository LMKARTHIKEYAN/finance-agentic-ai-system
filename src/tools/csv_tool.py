"""
CSV utility tool for the Finance Agentic AI System.

This module provides reusable methods to read and write CSV files.
It acts as the first data input layer for the FP&A backend.
"""

from pathlib import Path

import pandas as pd


class CSVTool:
    """
    Tool for handling CSV file operations.

    This class is responsible for reading CSV files, validating file paths,
    and writing pandas DataFrames back to CSV files.
    """

    @staticmethod
    def validate_file_path(file_path: str | Path) -> Path:
        """
        Validate whether the given CSV file path exists.

        Args:
            file_path: Path of the CSV file.

        Returns:
            Path: Validated file path as a Path object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the path is not a CSV file.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        if path.suffix.lower() != ".csv":
            raise ValueError(f"File must be a CSV file: {path}")

        return path

    @staticmethod
    def read_csv(file_path: str | Path) -> pd.DataFrame:
        """
        Read a CSV file and return it as a pandas DataFrame.

        Args:
            file_path: Path of the CSV file.

        Returns:
            pd.DataFrame: Data loaded from the CSV file.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV file is empty.
        """
        path = CSVTool.validate_file_path(file_path)

        dataframe = pd.read_csv(path)

        if dataframe.empty:
            raise ValueError(f"CSV file is empty: {path}")

        return dataframe

    @staticmethod
    def write_csv(dataframe: pd.DataFrame, file_path: str | Path) -> None:
        """
        Write a pandas DataFrame to a CSV file.

        Args:
            dataframe: DataFrame to save.
            file_path: Destination CSV file path.

        Raises:
            ValueError: If the DataFrame is empty.
        """
        if dataframe.empty:
            raise ValueError("Cannot write an empty DataFrame to CSV.")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        dataframe.to_csv(path, index=False)