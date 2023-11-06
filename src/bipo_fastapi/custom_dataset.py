from kedro_datasets.pandas import CSVDataSet
import pandas as pd


# Define Custom CSV Settings
class CustomCSVDataSet(CSVDataSet):
    def __init__(self, *args, **kwargs):
        # Set your desired default load_args and save_args here
        kwargs.setdefault("load_args", {"sep": "\t"})
        kwargs.setdefault("save_args", {"index": True})
        super().__init__(*args, **kwargs)

    def load(self) -> pd.DataFrame:
        # You can modify the load_args here if needed
        self._load_args["sep"] = ","
        self._load_args["index_col"] = "Date"
        return super().load()
