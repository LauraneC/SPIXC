""" Class object to work with SWOT PIXC water surface elevation stored in a csv or parquet file using polars and numba"""


import polars as pl
import geopandas as gpd
import numpy as np
import numba as nb
from geopandas import GeoDataFrame
from numpy import dtype, ndarray
from polars import DataFrame
from scipy.stats import gaussian_kde
from typing import Literal, Any
from pathlib import Path
from shapely import vectorized
import pandas as pd

MethodWSE = Literal["ATBD", "gaussianKDE"]
MethodSTD = Literal["random", "total", "weighted_variance"]
MethodFilter = Literal["robust", "normal"]


# ============================================================================
# Numba-accelerated functions
# ============================================================================

@nb.njit(cache=True)
def weighted_mean_numba(values: np.ndarray, weights: np.ndarray,
                        group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """Compute weighted mean per group using Numba."""
    sum_w = np.zeros(n_groups)
    sum_vw = np.zeros(n_groups)

    for i in range(values.shape[0]):
        g = group_idx[i]
        w = weights[i]
        v = values[i]

        if not np.isnan(v) and not np.isnan(w):
            sum_w[g] += w
            sum_vw[g] += v * w

    result = np.empty(n_groups)
    for g in range(n_groups):
        if sum_w[g] > 0:
            result[g] = sum_vw[g] / sum_w[g]
        else:
            result[g] = np.nan

    return result


@nb.njit(cache=True)
def uncertainty_weighted_variance_numba(
        height: np.ndarray, weights: np.ndarray,
        group_idx: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute uncertainty using weighted variance per group."""
    sum_w = np.zeros(n_groups)
    sum_hw = np.zeros(n_groups)
    sum_w2 = np.zeros(n_groups)

    # First pass: compute sums
    for i in range(height.shape[0]):
        g = group_idx[i]
        w = weights[i]
        h = height[i]

        if not np.isnan(h) and not np.isnan(w):
            sum_w[g] += w
            sum_hw[g] += h * w
            sum_w2[g] += w * w

    mean = np.empty(n_groups)
    for g in range(n_groups):
        if sum_w[g] > 0:
            mean[g] = sum_hw[g] / sum_w[g]
        else:
            mean[g] = np.nan

    # Second pass: compute variance
    var = np.zeros(n_groups)
    for i in range(height.shape[0]):
        g = group_idx[i]
        w = weights[i]
        h = height[i]

        if not np.isnan(h) and not np.isnan(w):
            diff = h - mean[g]
            var[g] += w * diff * diff

    result = np.empty(n_groups)
    Neff = np.empty(n_groups)

    for g in range(n_groups):
        if sum_w[g] > 0:
            var[g] /= sum_w[g]
            Neff[g] = sum_w[g] ** 2 / sum_w2[g]
            result[g] = np.sqrt(var[g] / Neff[g])
        else:
            result[g] = np.nan
            Neff[g] = np.nan

    return result, Neff


@nb.njit(cache=True)
def uncertainty_random_numba(
        phase_noise_std: np.ndarray, dheight_dphase: np.ndarray,
        group_idx: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute random uncertainty per group."""
    sum_pns = np.zeros(n_groups)
    sum_wp = np.zeros(n_groups)
    sum_dhp2_wp = np.zeros(n_groups)
    sum_dhp_wp = np.zeros(n_groups)
    count = np.zeros(n_groups, dtype=np.int64)

    for i in range(phase_noise_std.shape[0]):
        g = group_idx[i]
        pns = phase_noise_std[i]
        dhp = dheight_dphase[i]

        if not np.isnan(pns) and not np.isnan(dhp):
            wp = pns * dhp
            sum_pns[g] += pns
            sum_wp[g] += wp
            sum_dhp2_wp[g] += dhp ** 2 * wp
            sum_dhp_wp[g] += dhp * wp
            count[g] += 1

    result = np.empty(n_groups)
    n_points = np.empty(n_groups)

    for g in range(n_groups):
        if sum_wp[g] > 0:
            result[g] = np.sqrt(sum_pns[g]) * np.abs(
                (sum_dhp2_wp[g] / sum_wp[g]) / (sum_dhp_wp[g] / sum_wp[g])
            )
            n_points[g] = count[g]
        else:
            result[g] = np.nan
            n_points[g] = np.nan

    return result, n_points


@nb.njit(cache=True)
def uncertainty_total_numba(
        height: np.ndarray, weights: np.ndarray,
        eff_num_medium_looks: np.ndarray, eff_num_rare_looks: np.ndarray,
        group_idx: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute total uncertainty per group."""
    sum_w = np.zeros(n_groups)
    sum_hw = np.zeros(n_groups)
    sum_ratio = np.zeros(n_groups)
    count = np.zeros(n_groups, dtype=np.int64)

    # First pass
    for i in range(height.shape[0]):
        g = group_idx[i]
        w = weights[i]
        h = height[i]
        em = eff_num_medium_looks[i]
        er = eff_num_rare_looks[i]

        if not np.isnan(h) and not np.isnan(w) and not np.isnan(em) and not np.isnan(er) and er != 0:
            sum_w[g] += w
            sum_hw[g] += h * w
            sum_ratio[g] += em / er
            count[g] += 1

    mean = np.empty(n_groups)
    for g in range(n_groups):
        if sum_w[g] > 0:
            mean[g] = sum_hw[g] / sum_w[g]
        else:
            mean[g] = np.nan

    # Second pass: compute weighted std
    weighted_std = np.zeros(n_groups)
    for i in range(height.shape[0]):
        g = group_idx[i]
        w = weights[i]
        h = height[i]

        if not np.isnan(h) and not np.isnan(w) and sum_w[g] > 0:
            weighted_std[g] += (h - mean[g]) * w

    result = np.empty(n_groups)
    n_points = np.empty(n_groups)

    for g in range(n_groups):
        if count[g] > 0 and sum_w[g] > 0:
            weighted_std[g] /= sum_w[g]
            first_part = (sum_ratio[g] / count[g]) / count[g]
            result[g] = np.sqrt(first_part) * np.sqrt(np.abs(weighted_std[g]))
            n_points[g] = count[g]
        else:
            result[g] = np.nan
            n_points[g] = np.nan

    return result, n_points


def get_pdf_peak_value(wse_array: np.ndarray, remove_outliers: bool = True) -> ndarray[Any, dtype[Any]] | None | Any:
    """Get the elevation of the peak density using Gaussian KDE."""
    data = wse_array[~np.isnan(wse_array)]

    if len(data) < 2:
        return None

    if remove_outliers:
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data >= lower_bound) & (data <= upper_bound)]

    if len(data) < 2:
        return None

    x_min = np.min(data)
    x_max = np.max(data)
    x_grid = np.arange(x_min, x_max + 0.1, 0.1)

    kde = gaussian_kde(data)
    pdf = kde(x_grid)

    return x_grid[np.argmax(pdf)]


# ============================================================================
# Main SPixc class using Polars
# ============================================================================

class SPixc:
    """
    SWOT Pixel Cloud processor using Polars for improved performance.
    """

    def __init__(self, filename: str | Path):
        """
        Initialize SPixc processor.

        :param filename: Path to the Parquet file
        """
        self._filename = Path(filename)
        self._data: pl.LazyFrame | None = None
        self._data_collected: pl.DataFrame | None = None
        self._polygon_name: str | None = None
        self._gdf: gpd.GeoDataFrame | None = None
        self._wse_by_day: pl.DataFrame | None = None
        self._area: pl.DataFrame | None = None

    # ========================================================================
    # Data Loading
    # ========================================================================

    @property
    def data(self) -> pl.LazyFrame | None:
        """Lazy-load the parquet file."""
        if self._data is None:
            if Path(self._filename).suffix == ".parquet":
                self._load_parquet()
            elif Path(self._filename).suffix == ".csv":
                self._load_csv()
            else: raise NotImplementedError("Please provide csv or parquet files")
        return self._data

    @data.setter
    def data(self, obj: pl.LazyFrame | pl.DataFrame) -> None:
        if isinstance(obj, pl.DataFrame):
            self._data = obj.lazy()
            self._data_collected = obj
        else:
            self._data = obj
            self._data_collected = None

    def _load_parquet(self) -> None:
        """Load parquet file as a LazyFrame."""
        #
        # self._data = (
        #     pl.scan_parquet(self._filename)
        #     .with_columns(
        #         pl.col("time").str.to_datetime(format=None, strict=False).alias("time")
        #     )
        #     .sort("time")
        #     )


        self._data = (
            pl.scan_parquet(self._filename, try_parse_hive_dates=True))
        self._data_collected = None

    def _load_csv(self) -> None:
        """Load csv file as a LazyFrame."""
        self._data = (pl.scan_csv(
        self._filename,
        try_parse_dates=True,))

        self._data_collected = None


    def collect(self) -> DataFrame | None:
        """Collect the LazyFrame into a DataFrame (materializes the data)."""
        if self._data_collected is None:
            self._data_collected = self.data.collect()
        return self._data_collected

    def _load_polygon(self, polygon_name: str) -> GeoDataFrame | None:
        """Load a polygon file."""
        if self._gdf is None or self._polygon_name != polygon_name:
            self._polygon_name = polygon_name
            self._gdf = gpd.read_file(polygon_name)
        return self._gdf

    # ========================================================================
    # WSE Computation
    # ========================================================================

    def compute_wse(self) -> None:
        """
        Compute WSE as defined in ATBD for each acquisition and pixel.
        Removes geoid and tidal corrections.
        """
        self._data = self.data.with_columns([
            (
                    pl.col("height")
                    - pl.col("geoid")
                    - pl.col("solid_earth_tide")
                    - pl.col("load_tide_fes")
                    - pl.col("pole_tide")
            ).alias("wse"),
            (1.0 / (pl.col("phase_noise_std") * pl.col("dheight_dphase")) ** 2).alias("height_std")
        ])
        self._data_collected = None

    def _ensure_wse_computed(self) -> None:
        """Ensure WSE columns exist."""
        schema = self.data.collect_schema()
        if "wse" not in schema or "height_std" not in schema:
            self.compute_wse()

    def compute_weighted_mean_wse_by_day(
            self,
            method_wse: MethodWSE = "ATBD",
            method_uncertainty: MethodSTD = "weighted_variance", minimal_nb_of_points=2
    ) -> pl.DataFrame | None:
        """
        Compute WSE per day using Numba-accelerated functions.

        :param method_wse: Method for WSE computation ("ATBD" or "gaussianKDE")
        :param method_uncertainty: Method for uncertainty ("random", "total", "weighted_variance")
        :return: DataFrame with daily WSE, uncertainty, and point counts

        gaussianKDE implement the method proposed in
        """
        self._ensure_wse_computed()
        df = self.collect()

        #convert dates to (day, month, year), and keep only the dates where there is a sufficient number of points (defined by minimal_nb_of_points)
        df = (
            df.with_columns(pl.col("time").dt.date().alias("date"))
            .with_columns(pl.len().over("date").alias("n_points_raw"))
            .filter(pl.col("n_points_raw") >= minimal_nb_of_points)
        )

        if df.is_empty():
            return None

        dates = df["date"].to_numpy()
        unique_dates, group_idx = np.unique(dates, return_inverse=True)
        n_groups = len(unique_dates)

        wse = df["wse"].to_numpy().astype(np.float64)
        height = df["height"].to_numpy().astype(np.float64)
        weights = df["height_std"].to_numpy().astype(np.float64)

        # Compute WSE
        if method_wse == "ATBD":
            wse_by_day = weighted_mean_numba(wse, weights, group_idx, n_groups)
        elif method_wse == "gaussianKDE":
            # For KDE, compute per group
            wse_by_day = np.empty(n_groups)
            for g in range(n_groups):
                mask = group_idx == g
                peak = get_pdf_peak_value(wse[mask])
                wse_by_day[g] = peak if peak is not None else np.nan
        else:
            raise ValueError(f"method_wse must be one of {MethodWSE.__args__}")

        # Compute uncertainty
        if method_uncertainty == "weighted_variance":
            uncertainty, n_eff = uncertainty_weighted_variance_numba(
                height, weights, group_idx, n_groups
            )
        elif method_uncertainty == "random":
            phase_noise_std = df["phase_noise_std"].to_numpy().astype(np.float64)
            dheight_dphase = df["dheight_dphase"].to_numpy().astype(np.float64)
            uncertainty, n_eff = uncertainty_random_numba(
                phase_noise_std, dheight_dphase, group_idx, n_groups
            )
        elif method_uncertainty == "total":
            eff_num_medium = df["eff_num_medium_looks"].to_numpy().astype(np.float64)
            eff_num_rare = df["eff_num_rare_looks"].to_numpy().astype(np.float64)
            uncertainty, n_eff = uncertainty_total_numba(
                height, weights, eff_num_medium, eff_num_rare, group_idx, n_groups
            )
        else:
            raise ValueError(f"method_uncertainty must be one of {MethodSTD.__args__}")

        self._wse_by_day = pd.DataFrame({
            "date": unique_dates,
            "wse_by_day": wse_by_day,
            "uncertainty": uncertainty,
            "n_points": n_eff
        })
        self._wse_by_day.index = pd.DatetimeIndex(self._wse_by_day["date"])

        return self._wse_by_day

    # ========================================================================
    # Area Computation
    # ========================================================================

    def computation_area(self) -> pl.DataFrame:
        """
        Compute total lake area per day using ATBD formula.

        Area = sum(pixel_area * coverage_fraction) where:
        - Interior water pixels (class > 3): coverage = 1
        - Edge water pixels (class == 3 or 6): coverage = water_frac

        :return: DataFrame with date and area in km²
        """
        df = self.collect()

        # Classification:
        # 3 = water on land edge, 6 = low coherence water edge
        # > 3 = interior water (4, 5, 7)

        self._area = (
            df.filter(pl.col("classification") > 2)
            .with_columns([
                pl.col("time").dt.date().alias("date"),
                pl.when(pl.col("classification").is_in([3, 6]))
                .then(pl.col("water_frac"))
                .otherwise(1.0)
                .alias("coverage")
            ])
            .group_by("date")
            .agg(
                (pl.col("pixel_area") * pl.col("coverage")).sum().alias("area_m2")
            )
            .with_columns(
                (pl.col("area_m2") / 1e6).alias("area_km2")
            )
            .sort("date")
        )

        return self._area

    # ========================================================================
    # Filtering Methods
    # ========================================================================

    def filter_by_variable(self, conditions: dict) -> None:
        """
        Filter data based on variable conditions.
        Inspired by PIXCDust library: https://github.com/SWOT-community/PixCDust.git

        :param conditions: Dict mapping variable names to condition specs.
            Example: {
                "sig0": {"operator": "ge", "threshold": 20},
                "classification": {"operator": "ge", "threshold": 3}
            }

        Operators: lt (<), le (<=), eq (==), ne (!=), gt (>), ge (>=)
        """
        OPERATORS = {
            "lt": pl.Expr.__lt__,
            "le": pl.Expr.__le__,
            "eq": pl.Expr.__eq__,
            "ne": pl.Expr.__ne__,
            "gt": pl.Expr.__gt__,
            "ge": pl.Expr.__ge__,
        }
        schema = self.data.collect_schema()
        filters = []

        for var, condition in conditions.items():
            if var not in schema:
                raise IOError(
                    f"Variable '{var}' not found in dataset. "
                    f"Available: {list(schema.keys())}"
                )

            # --- Absolute value range filter ---
            if "abs_between" in condition:
                lower, upper = condition["abs_between"]

                if lower > upper:
                    raise ValueError(
                        f"For '{var}', lower bound must be <= upper bound."
                    )

                expr = pl.col(var).abs()
                filters.append((expr >= lower) & (expr <= upper))
                continue

            # --- Standard operator filter ---
            if "operator" not in condition or "threshold" not in condition:
                raise ValueError(
                    f"Condition for '{var}' must include either "
                    f"'operator' + 'threshold' or 'abs_between'"
                )

            op_name = condition["operator"]
            if op_name not in OPERATORS:
                raise ValueError(
                    f"Operator '{op_name}' not valid. "
                    f"Use one of: {list(OPERATORS.keys())}"
                )

            threshold = condition["threshold"]
            op_func = OPERATORS[op_name]
            filters.append(op_func(pl.col(var), threshold))

        if filters:
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter = combined_filter & f

            self._data = self.data.filter(combined_filter)
            self._data_collected = None



    def filter_by_space_stats(
            self,
            method_filter: MethodFilter = "normal",
            threshold: float = 2.0
    ) -> None:
        """
        Filter WSE per pixel and day using spatial statistics.

        :param method_filter: "normal" (mean/std) or "robust" (median/MAD)
        :param threshold: Number of standard deviations for filtering
        """
        self._ensure_wse_computed()

        if method_filter == "normal":
            self._data = (
                self.data
                .with_columns([
                    pl.col("wse").mean().over("time").alias("wse_center"),
                    pl.col("wse").std().over("time").alias("wse_scale")
                ])
                .filter(
                    (pl.col("wse") >= pl.col("wse_center") - threshold * pl.col("wse_scale")) &
                    (pl.col("wse") <= pl.col("wse_center") + threshold * pl.col("wse_scale"))
                )
                .drop(["wse_center", "wse_scale"])
            )
        elif method_filter == "robust":
            # For MAD, we need to collect since Polars doesn't have native MAD
            df = self.collect()

            # Compute median and MAD per time group
            stats = (
                df.group_by("time")
                .agg([
                    pl.col("wse").median().alias("wse_median"),
                    pl.col("wse").alias("wse_values")
                ])
                .with_columns(
                    pl.col("wse_values").map_elements(
                        lambda x: np.median(np.abs(x - np.median(x))) * 1.4826,
                        return_dtype=pl.Float64
                    ).alias("wse_mad")
                )
                .drop("wse_values")
            )

            df = df.join(stats, on="time", how="left")
            df = df.filter(
                (pl.col("wse") >= pl.col("wse_median") - threshold * pl.col("wse_mad")) &
                (pl.col("wse") <= pl.col("wse_median") + threshold * pl.col("wse_mad"))
            ).drop(["wse_median", "wse_mad"])

            self._data = df.lazy()
        else:
            raise ValueError(f"method_filter must be one of {MethodFilter.__args__}")

        self._data_collected = None

    def filter_by_temporal_stats(
            self,
            method_filter: MethodFilter = "normal",
            threshold: float = 3.0
    ) -> None:
        """
        Filter WSE per pixel using temporal statistics (across all dates for each location).

        :param method_filter: "normal" (mean/std) or "robust" (median/MAD)
        :param threshold: Number of standard deviations for filtering
        """
        self._ensure_wse_computed()

        if method_filter == "normal":
            self._data = (
                self.data
                .with_columns([
                    pl.col("wse").mean().over(["latitude", "longitude"]).alias("wse_center"),
                    pl.col("wse").std().over(["latitude", "longitude"]).alias("wse_scale")
                ])
                .filter(
                    (pl.col("wse") >= pl.col("wse_center") - threshold * pl.col("wse_scale")) &
                    (pl.col("wse") <= pl.col("wse_center") + threshold * pl.col("wse_scale"))
                )
                .drop(["wse_center", "wse_scale"])
            )
        elif method_filter == "robust":
            df = self.collect()

            stats = (
                df.group_by(["latitude", "longitude"])
                .agg([
                    pl.col("wse").median().alias("wse_median"),
                    pl.col("wse").alias("wse_values")
                ])
                .with_columns(
                    pl.col("wse_values").map_elements(
                        lambda x: np.median(np.abs(x - np.median(x))) * 1.4826,
                        return_dtype=pl.Float64
                    ).alias("wse_mad")
                )
                .drop("wse_values")
            )

            df = df.join(stats, on=["latitude", "longitude"], how="left")
            df = df.filter(
                (pl.col("wse") >= pl.col("wse_median") - threshold * pl.col("wse_mad")) &
                (pl.col("wse") <= pl.col("wse_median") + threshold * pl.col("wse_mad"))
            ).drop(["wse_median", "wse_mad"])

            self._data = df.lazy()
        else:
            raise ValueError(f"method_filter must be one of {MethodFilter.__args__}")

        self._data_collected = None

    def filter_by_wsedaystats(
            self,
            method_filter: MethodFilter = "normal",
            threshold: float = 2.0
    ) -> None:
        """
        Filter data using temporal statistics of the daily WSE time series.

        :param method_filter: "normal" (mean/std) or "robust" (median/MAD)
        :param threshold: Number of standard deviations for filtering
        """
        self._ensure_wse_computed()

        if self._wse_by_day is None:
            self.compute_weighted_mean_wse_by_day()

        wse_values = self._wse_by_day["wse_by_day"].to_numpy()
        wse_values = wse_values[~np.isnan(wse_values)]

        if method_filter == "normal":
            center = np.mean(wse_values)
            scale = np.std(wse_values)
        elif method_filter == "robust":
            center = np.median(wse_values)
            scale = np.median(np.abs(wse_values - center)) * 1.4826
        else:
            raise ValueError(f"method_filter must be one of {MethodFilter.__args__}")

        lower = center - threshold * scale
        upper = center + threshold * scale

        self._data = self.data.filter(
            (pl.col("wse") >= lower) & (pl.col("wse") <= upper)
        )
        self._data_collected = None

    # ========================================================================
    # Spatial Masking
    # ========================================================================

    def mask_according_to_polygon(self, polygon_path):

        gdf = gpd.read_file(polygon_path)

        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        polygon = gdf.geometry.unary_union
        minx, miny, maxx, maxy = polygon.bounds

        # step 1: lazy bbox filter
        df = self._data.filter(
            (pl.col("longitude") >= minx) &
            (pl.col("longitude") <= maxx) &
            (pl.col("latitude") >= miny) &
            (pl.col("latitude") <= maxy)
        )

        # step 2: collect ONLY coords
        coords = df.select(["longitude", "latitude"]).collect()

        mask = vectorized.contains(
            polygon,
            coords["longitude"].to_numpy(),
            coords["latitude"].to_numpy()
        )


        # step 3: apply mask via index (NO pandas)
        df = df.with_row_index("row_nr")

        idx = np.where(mask)[0]

        self._data = df.filter(pl.col("row_nr").is_in(idx)).drop("row_nr")

    def mask_by_year_month_polygons(
            self,
            polygon_map: dict[tuple[int, int, int], gpd.GeoDataFrame]
    ) -> None:
        """
        Mask data according to time-varying lake boundaries.

        Each polygon corresponds to a lake contour at a specific date.
        Data points are masked using the temporally nearest polygon.

        :param polygon_map: Dict mapping (year, month, day) tuples to GeoDataFrames
        """
        import pandas as pd
        from pandas import Timestamp

        df = self.collect()

        # Build time-indexed polygon series
        poly_dates = []
        poly_gdfs = []

        for (y, m, d), gdf in polygon_map.items():
            poly_dates.append(Timestamp(y, m, d))
            poly_gdfs.append(gdf)

        poly_index = pd.DatetimeIndex(poly_dates)
        poly_series = pd.Series(poly_gdfs, index=poly_index).sort_index()

        # Convert to pandas for datetime operations
        df_pd = df.to_pandas()
        df_pd["date"] = df_pd["time"]

        # Find nearest polygon date for each row
        nearest_idx = poly_series.index.get_indexer(df_pd["date"], method="nearest")
        df_pd["poly_date"] = poly_series.index[nearest_idx]

        masked_parts = []

        for poly_date, df_grp in df_pd.groupby("poly_date"):
            polygon_gdf = poly_series.loc[poly_date]

            if polygon_gdf.crs != "EPSG:4326":
                polygon_gdf = polygon_gdf.to_crs("EPSG:4326")

            gdf_pts = gpd.GeoDataFrame(
                df_grp,
                geometry=gpd.points_from_xy(df_grp.longitude, df_grp.latitude),
                crs="EPSG:4326",
            )

            clipped = gpd.clip(gdf_pts, polygon_gdf)
            masked_parts.append(
                clipped.drop(columns=["geometry", "date", "poly_date"])
            )

        result = pd.concat(masked_parts).sort_values("time")
        self._data = pl.from_pandas(result).lazy()
        self._data_collected = None

    # ========================================================================
    # Utility Methods
    # ========================================================================
    def get_orbit_number(self) -> pd.DataFrame:
        """
        Return a pd.Dataframe with an orbit number per date
        """
        # Assuming datapixc is a Polars DataFrame
        orbit_number = (
            self.data
            .group_by(pl.col("time").dt.date())  # Group by date part of "time"
            .agg(pl.col("pass_number").first())  # Get first pass_number for each date
        )
        print("Start collecting orbit number")
        orbit_number = orbit_number.collect()
        orbit_number = orbit_number.to_pandas()
        return orbit_number

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def to_parquet(self, filename: str | Path) -> None:
        """Save current data to a parquet file."""
        self.collect().write_parquet(filename)

    def __repr__(self) -> str:
        schema = self.data.collect_schema()
        return f"SPixc(file={self._filename}, columns={list(schema.keys())})"
