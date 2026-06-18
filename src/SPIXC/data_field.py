""" Class object to work with insitu water surface elevation stored in a csv file"""
import pandas as pd
import numpy as np
from permetrics import RegressionMetric #10.5281/zenodo.3951205
from typing import Literal

MethodInterp = Literal["linear", "nearest"]

class DataField:
    def __init__(self, filename: str):
        """
        :param filename: Filename of the CSV file
        """
        self._filename = filename
        self._data = None
        self._data_compa = None

### LOADER

    @property #SPixx.ds will directly call this function, i.e. load the csv file if needed
    def data(self) -> pd.DataFrame:
        if self._data is None :
            self._load_csv()
        return self._data

    @data.setter
    def data(self, obj: pd.DataFrame) -> None:
        self._data = obj

    @data.setter
    def data_compa(self, obj: pd.DataFrame) -> None:
        self._data_compa = obj

    def _load_csv(self):
        """
        Load csv file
        """
        data = pd.read_csv(self._filename, index_col=0, parse_dates=True)
        try:data.index = pd.to_datetime(data.index)
        except pd._libs.tslibs.parsing.DateParseError: #if the separator is ;
            data = pd.read_csv(self._filename, index_col=0, parse_dates=True,sep=";")
            data.index = pd.to_datetime(data.index)
        self._data = data

##COMPUTE

    def stats(self,data_compa:pd.DataFrame, preprocessing:str='first',method_interp:MethodInterp="linear") -> (dict[float, float], pd.DataFrame):
        """
        Compute statistics about the comparison between self.data and data_compa
        The two dataframe are aligned to cover the same dates, using a nearest-neighbor interpolation.
        :param data_compa: dataframe which contain the SWOT data to compare to
        :param preprocessing: name of the processing which have been applied, used as a index in the output dataframe
        :param method_interp: method which be used to interpolate the field data to the SWOT data, could be linear or nearest
        :return: a dictionary with the RMSE and bias, and the aligned pd dataframe
        """

        data_compa.index = pd.to_datetime(data_compa.index).tz_localize(None)
        self.data.index = pd.to_datetime(self.data.index).tz_localize(None)

        #Align the name of the columns
        if isinstance(data_compa,pd.DataFrame) and "wse" in data_compa.columns: #wse already exist
            data_compa.rename(columns={"wse":"wse_1"},inplace=True)
        else:
            data_compa = data_compa.rename('wse_1')
        self._data_compa = data_compa

        self.data.columns = ["wse_2"]

        data = self.data.sort_index()
        target_index = self._data_compa.index.sort_values()


        if method_interp == "linear":
            # Reindex to target timestamps
            try:
                data_interp = (
                    data
                    .reindex(data.index.union(target_index))
                    .interpolate(method="time")
                    .loc[target_index])
            except ValueError:
                data = data.groupby(level=0).mean()
                data_interp = (
                    data
                    .reindex(data.index.union(target_index))
                    .interpolate(method="time")
                    .loc[target_index])
            aligned = pd.DataFrame({"wse_1":data_compa,"wse_2":data_interp.wse_2})

        if method_interp == "nearest":
            aligned = pd.merge_asof(
                self._data_compa.sort_index(),
                self.data.sort_index(),
                left_index=True,
                right_index=True,
                direction="nearest",  # take closest in time
            )
        # Compute RMSE
        bias = np.mean(aligned["wse_1"]) - np.mean(aligned["wse_2"])
        evaluator = RegressionMetric(aligned["wse_2"].to_numpy(), aligned["wse_1"].to_numpy())
        results = evaluator.get_metrics_by_list_names(["RMSE", "MAE", "MAPE", "R2", "KGE","R"])

        absolute_errors = np.abs(aligned["wse_1"] - aligned["wse_2"])
        percentile_68 = np.percentile(absolute_errors, 68)

        stats_tab = pd.DataFrame({"preprocessing":[preprocessing],"1 sigma":percentile_68, "Bias": [bias],"RMSE": [results["RMSE"]], "KGE": [results["KGE"]],"MAE":[results["MAE"]], "MAPE":[results["MAPE"]], "R2": [results["R2"]],"R":[results["R"]],"nb_points":aligned.shape[0]})
        return stats_tab, aligned