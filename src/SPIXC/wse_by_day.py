import pandas as pd


class WSEbyDay:
    def __init__(self, filename: str):
        """
        :param filename: Filename of the CSV file
        """
        self._data = None

    ### LOADER
    @property  # SPixx.ds will directly call this function, i.e. load the csv file if needed
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, obj: pd.DataFrame) -> None:
        self._data = obj

    def get_stats(self):

        return self.data.mean()



