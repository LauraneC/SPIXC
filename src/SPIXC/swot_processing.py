import pandas as pd
import geopandas as gpd
import operator
from pyproj import CRS
import numpy as np
import math
from scipy.stats import gaussian_kde
from typing import Literal

MethodWSE = Literal["ATBD", "gaussianKDE"]
MethodSTD = Literal["random","total","weighted_variance"]

def weighted_avg_and_std(values:np.array, weights:np.array)-> (np.array, np.array):
    """
    Compute weighted average and standard deviation.
    :param values: data used to compute weighted average and standard deviation.
    :param weights: weights used to compute weighted average and standard deviation.
    :return: average, standard deviation
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Weighted average, ignoring NaN
    sum_w = np.nansum(weights)
    if sum_w == 0:
        return np.nan, np.nan

    avg = np.nansum(values * weights) / sum_w
    var = np.nansum(weights * (values - avg) ** 2) / sum_w
    return avg, math.sqrt(var)

def get_pdf_peak_value(data_array:pd.DataFrame, remove_outliers:bool=True):
    """Get the elevation of the peak density using a function from
    Han, X., Zhang, G., Crétaux, J.-F., Wang, J., Schwatke, C., Peng, M., Wang, X., Shum, C. K., Woolway, R. I., Ke, Y., Wang, Y., Zhou, T., & Xu, F. (2025). Surface Water and Ocean Topography (SWOT) L2_HR_PIXC data processing for lakes. In Water Resource Research (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.15735885
    :param data_array: Array of data
    :param remove_outliers: if yes remove outliers
    """

    data = data_array["wse"].to_numpy().copy()
    data = data[~np.isnan(data)]

    if len(data) < 2:
        return None

    # remove outlier (IQR)
    if remove_outliers and len(data) > 1:
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data >= lower_bound) & (data <= upper_bound)]

    if len(data) < 2:
        return None
    # Gaussian KDE
    x_min = np.min(data)
    x_max = np.max(data)
    x_grid = np.arange(x_min, x_max + 0.1, 0.1)

    kde = gaussian_kde(data)
    pdf = kde(x_grid)

    # find max
    peak_value = x_grid[np.argmax(pdf)]
    return peak_value


class SPixc:
    def __init__(self, filename: str):
        """
        :param filename: Filename of the CSV file
        """
        self._filename = filename
        self._data = None
        self._polygon_name = None
        self._gdf = None
        self._wse_by_day = None

    ### LOADER

    @property  # SPixx.ds will directly call this function, i.e. load the csv file if needed
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._load_csv()
        return self._data

    @data.setter
    def data(self, obj: pd.DataFrame) -> None:
        self._data = obj


    def _load_csv(self):
        data = pd.read_csv(self._filename)
        data['time'] = pd.to_datetime(data['time'])
        data.set_index('time', inplace=True)
        data.sort_index(inplace=True)
        self._data = data

    def _load_polygon(self, polygon_name:str)->gpd.GeoDataFrame:
        """
        Load a polygon, used to define the lake outline.
        :param polygon_name: filename of the polygone
        :return: polygone loaded as a geopandas object
        """
        if self._gdf is None:
            self._polygon_name = polygon_name
            self._gdf = gpd.read_file(polygon_name)
        return self._gdf

    ### COMPUTATION

    def compute_wse(self):
        """
        Compute wse as defined in ATBD for each acquisition and pixel.
        Remove the EGM2008 modeled geoid,
        """
        data = self.data
        data['wse'] = data['height'] - data['geoid'] - data['solid_earth_tide'] - \
                      data['load_tide_fes'] - data['pole_tide']
        data['height_std'] = 1 / (data['phase_noise_std'] * data['dheight_dphase']) ** 2

    def compute_weighted_mean_wse_by_day(self, method_wse:MethodWSE="ATBD",method_uncertainty="weighted_variance") -> pd.DataFrame:
        """
        Compute wse per day for the entire dataframe, using weighted wse as defined in ATBD
        """
        data = self.data
        if 'wse' not in data or 'height_std' not in data:
            self.compute_wse()

        def weighted_mean_wse(group):
            weighted_sum = (group.wse * group.height_std).sum()
            weight_sum = group.height_std.sum()
            return weighted_sum / weight_sum

        def uncertainty_total(group):
            nb_pixel = group.height_std.count()
            weight_sum = group.height_std.sum()
            height_mean = np.sum(group.height * group.height_std)/weight_sum
            weighted_std = ((group.height -height_mean) * group.height_std).sum()/weight_sum
            first_part = (np.sum(group.eff_num_medium_looks/group.eff_num_rare_looks)/nb_pixel)/nb_pixel
            uncertainty = np.sqrt(first_part)*np.sqrt(weighted_std)
            return uncertainty

        def uncertainty_random(group):
            wp_normalized = group.phase_noise_std * group.dheight_dphase
            uncertainty = np.sqrt(np.sum(group.phase_noise_std)) * np.abs((np.sum(group.dheight_dphase**2*wp_normalized)/np.sum(wp_normalized))/(np.sum(group.dheight_dphase*wp_normalized)/np.sum(wp_normalized)))
            return uncertainty

        def uncertainty_weighted_variance(group):
            weight_sum = group.height_std.sum()
            height_mean = np.sum(group.height * group.height_std) / weight_sum
            std_weighted = np.sum(group.height_std*(group.height -height_mean)**2)/weight_sum
            Neff = group.height_std.sum()**2/np.sum(group.height_std**2)
            uncertainty = np.sqrt(std_weighted/Neff)
            return pd.Series({
                "uncertainty": uncertainty,
                "n_points_eff": Neff
            })

        if not data.empty:
            if method_wse == "ATBD": wse_by_day = data.groupby(data.index.date).apply(weighted_mean_wse)
            elif method_wse == "gaussianKDE": wse_by_day= data.groupby(data.index.date).apply(get_pdf_peak_value)
            else: raise ValueError(f"Please set a value for method_wse among these options {MethodWSE}")
            if method_uncertainty == "random": results = data.groupby(data.index.date).apply(uncertainty_random)
            elif method_uncertainty == "total": results = data.groupby(data.index.date).apply(
                uncertainty_total)
            elif method_uncertainty == "weighted_variance": results = data.groupby(data.index.date).apply(
                uncertainty_weighted_variance)
            else: raise ValueError(f"Please set a value for method_uncertainty among these options {MethodSTD}")

            self._wse_by_day = pd.DataFrame({"wse_by_day": wse_by_day, "uncertainty": results["uncertainty"],"n_points":results["n_points_eff"]})
            return self._wse_by_day
        return None

    ### PRE-PROCESSING

    def filter_by_variable(self, conditions) -> None:
        """
        Filters xarray dataset based on operator and threshold on specific variables.
        function modified from PIXCDust

        :param conditions: Conditions to filter variables.\
                    Example: {\
                    "sig0":{'operator': "ge", 'threshold': 20},\
                    "classification":{'operator': "ge", 'threshold': 3},\
                    }
        Note that Perform “rich comparisons” between a and b. Specifically, lt(a, b) is equivalent to a < b, le(a, b) is equivalent to a <= b, eq(a, b) is equivalent to a == b, ne(a, b) is equivalent to a != b, gt(a, b) is equivalent to a > b and ge(a, b) is equivalent to a >= b.
        Raises:
            IOError: If the variable provided in conditions is not in the dataset.
            ValueError: If 'operator' or 'threshold' keys are not in conditions.
            AttributeError: If operator is not the function name of the operator module.
        """
        _k_operator = 'operator'
        _k_to = 'threshold'

        # Loop through each condition and apply the filter
        for var, condition in conditions.items():
            if var not in self.data.columns:
                raise IOError(
                    f"Variable '{var}' not found in dataset variables (available: {list(self.data.variables)})"
                )

            # Ensure the condition dictionary has the correct keys
            if _k_operator not in condition or _k_to not in condition:
                raise ValueError(f"Condition for variable '{var}' must include '{_k_operator}' and '{_k_to}'")

            # Get the operator function dynamically from the operator module
            try:
                operator_func = getattr(operator, condition[_k_operator])
            except AttributeError:
                raise AttributeError(
                    f"Operator '{condition[_k_operator]}' is not a valid operator in the operator module")

            threshold = condition[_k_to]

            # Apply the filter using .where() on the dataset
            self.data = self.data[(operator_func(self.data[var], threshold))]

    def filter_by_space_stats(self, type: str = "normal", treshold: int = 2) -> pd.DataFrame:
        """
        Filter wse per pixel and day using spatial statics
        :param type: type of statics, normal
        :return:
        """
        data = self.data
        if 'wse' not in data:
            self.compute_wse()
        group_stats = data.groupby("time")[
            "wse"].transform  # to assign the computed criterion on the original dataframe

        if type == "normal":
            data["wse_center"] = group_stats("mean")
            data["wse_scale"] = group_stats("std")

        elif type == "normal_weightedstd":
            #filter based on the weighted average and weighted std
            # Apply on each group and return as two named columns
            group_stats = (
                data.groupby("time")
                .apply(lambda g: pd.Series(
                    weighted_avg_and_std(g["wse"], g["height_std"]),
                    index=["wse_center", "wse_scale"]
                ))
            )

            data["wse_center"] = group_stats.wse_center
            data["wse_scale"] = group_stats.wse_scale

        elif type == "normal_weighted":
            #filter based on the weighted average and not weighted std

            # Apply on each group and return as two named columns
            group_stats_weighted = (
                data.groupby("time")
                .apply(lambda g: pd.Series(
                    weighted_avg_and_std(g["wse"], g["height_std"]),
                    index=["wse_center", "wse_scale"]
                ))
            )
            data["wse_center"] = group_stats_weighted.wse_center
            data["wse_scale"] = group_stats("std")

        elif type == "robust":
            data["wse_center"] = group_stats("median")
            # MAD = median(|x - median|)
            data["wse_scale"] = data.groupby("time")["wse"].transform(
                lambda x: np.median(np.abs(x - np.median(x)))
            ) * 1.4826  # scaling factor to make MAD ~ std

        else:
            raise ValueError("Please enter robust, normal or normal_weighted")

        # Keep only values within center ± threshold*scale
        self.data = data[(data["wse"] >= data["wse_center"] - treshold * data["wse_scale"]) &
                         (data["wse"] <= data["wse_center"] + treshold * data["wse_scale"])].drop(
            columns=["wse_center", "wse_scale"])

    def filter_by_temporal_stats(self, type: str = "normal", treshold: int = 3) -> pd.DataFrame:
        """
        Filter wse per pixel and day using temporal statistics
        :param type:
        :return:
        """
        data = self.data
        if 'wse' not in data:
            self.compute_wse()
        group_stats = data.groupby(["latitude", "longitude"])[
            "wse"].transform  # to assign the computed criterion on the original dataframe

        if type == "normal":
            data["wse_center"] = group_stats("mean")
            data["wse_scale"] = group_stats("std")

        elif type == "robust":
            data["wse_center"] = group_stats("median")
            # MAD = median(|x - median|)
            data["wse_scale"] = data.groupby(["latitude", "longitude"])["wse"].transform(
                lambda x: np.median(np.abs(x - np.median(x)))
            ) * 1.4826  # scaling factor to make MAD ~ std

        else:
            raise ValueError("Please enter robust or normal")

        # Keep only values within center ± threshold*scale
        self.data = data[(data["wse"] >= data["wse_center"] - treshold * data["wse_scale"]) &
                         (data["wse"] <= data["wse_center"] + treshold * data["wse_scale"])].drop(
            columns=["wse_center", "wse_scale"])

    def filter_by_wsedaystats(self,type="normal",treshold = 2):
        """

        :param type:
        :param treshold:
        :return:
        """

        data = self.data

        if 'wse' not in data:
            self.compute_wse()

        if self._wse_by_day is None:
            self.compute_weighted_mean_wse_by_day()

        if type == "normal":
            center, scale = self._wse_by_day.wse_by_day.mean(), self._wse_by_day.wse_by_day.std()
        elif type == "robust":
            center, scale = np.median(self._wse_by_day.wse_by_day), np.median(np.abs(self._wse_by_day.wse_by_day - np.median(self._wse_by_day.wse_by_day)))* 1.4826

        self.data = data[(data["wse"] >= center - treshold * scale) &
                          (data["wse"] <= center + treshold * scale)]


    def mask_according_to_polygon(self, polygon_name: str):
        # 1. Convert df_combined (Pandas DataFrame) to a GeoDataFrame
        data = self.data
        gdf = self._load_polygon(polygon_name)

        if gdf.crs != CRS("4326"):
            gdf = gdf.to_crs("epsg:4326")

        gdf_combined = gpd.GeoDataFrame(
            data,
            geometry=gpd.points_from_xy(data.longitude, data.latitude),
            crs=gdf.crs  # Make sure it matches your polygon CRS
        )

        # 2. Use spatial join or contains to keep only points inside the polygon
        # gdf_filtered = gpd.sjoin(gdf_combined, gdf, how="inner", predicate="within")
        gdf_filtered = gpd.clip(gdf_combined, gdf)

        # 3. Drop geometry if you want a regular DataFrame again
        self._data = pd.DataFrame(gdf_filtered.drop(columns='geometry'))

    def mask_by_year_month_polygons(self, polygon_map: dict[tuple[int, int], gpd.GeoDataFrame]):
        """
        Mask according to a dictionary of geodataframe.
        Each of the geodataframe corresponds to the contour of the lake at a given date.
        The SWOT data are masked according to the polygon which is the closest in time.
        :param polygon_map:
        :return:
        """

        from pandas import Timestamp

        # Convert polygon_map to time-indexed structure
        poly_dates = []
        poly_gdfs = []

        for (y, m,d), gdf in polygon_map.items():
            poly_dates.append(Timestamp(y, m, d))  # mid-month anchor
            poly_gdfs.append(gdf)

        poly_index = pd.DatetimeIndex(poly_dates)
        poly_series = pd.Series(poly_gdfs, index=poly_index).sort_index()

        df = self.data.copy()
        df["date"] = df.index

        # Find nearest polygon date
        nearest_idx = poly_series.index.get_indexer(df["date"], method="nearest")
        df["poly_date"] = poly_series.index[nearest_idx]

        masked_parts = []

        for poly_date, df_grp in df.groupby("poly_date"):
            polygon_gdf = poly_series.loc[poly_date]

            if polygon_gdf.crs != "EPSG:4326":
                polygon_gdf = polygon_gdf.to_crs("EPSG:4326")

            gdf_pts = gpd.GeoDataFrame(
                df_grp,
                geometry=gpd.points_from_xy(df_grp.longitude, df_grp.latitude),
                crs="EPSG:4326",
            )

            clipped = gpd.clip(gdf_pts, polygon_gdf)
            masked_parts.append(clipped.drop(columns=["geometry", "date", "poly_date"]))

        self._data = pd.concat(masked_parts).sort_index()
