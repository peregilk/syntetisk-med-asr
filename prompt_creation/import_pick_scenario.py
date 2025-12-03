import random
import pandas as pd
from pathlib import Path

class ScenarioLoader:
    """
    Loads scenarios from an Excel file and can return a single random scenario.
    """

    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)

        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")

        # Load Excel once on init
        self.df = pd.read_excel(self.excel_path)

        if "Scenario" not in self.df.columns:
            raise ValueError("Excel file must contain a column named 'Scenario'.")

    def get_random_scenario(self) -> str:
        """
        Returns one random scenario string from the Excel file.
        """
        return random.choice(self.df["Scenario"].dropna().tolist())


# If called directly (python scenario_loader.py), test loading and print a sample scenario
if __name__ == "__main__":
    loader = ScenarioLoader("medical_scenarios_with_categories_v2.xlsx")
    print("Random scenario:", loader.get_random_scenario())
