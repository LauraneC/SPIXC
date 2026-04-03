import polars as pl
import numpy as np
import numba as nb
import pandas as pd
from typing import Literal

MethodWSE = Literal["ATBD", "gaussianKDE"]
MethodSTD = Literal["weighted_variance"]


# =========================
# NUMBA KERNELS
# =========================

@nb.njit
def weighted_mean_numba(values, weights, group_idx, n_groups):
    sum_w = np.zeros(n_groups)
    sum_vw = np.zeros(n_groups)

    for i in range(values.shape[0]):
        g = group_idx[i]
        w = weights[i]
        v = values[i]

        if not np.isnan(v) and not np.isnan(w):
            sum_w[g] += w
            sum_vw[g] += v * w

    out = np.empty(n_groups)
    for g in range(n_groups):
        out[g] = sum_vw[g] / sum_w[g] if sum_w[g] > 0 else np.nan

    return out


@nb.njit
def uncertainty_weighted_variance_numba(height, weights, group_idx, n_groups):
    sum_w = np.zeros(n_groups)
    sum_hw = np.zeros(n_groups)
    sum_w2 = np.zeros(n_groups)

    for i in range(height.shape[0]):
        g = group_idx[i]
        w = weights[i]
        h = height[i]

        if not np.isnan(h) and not np.isnan(w):
            sum_w[g] += w
            sum_hw[g] += h * w
            sum_w2[g] += w * w

    mean = sum_hw / sum_w

    var = np.zeros(n_groups)
    for i in range(height.shape[0]):
        g = group_idx[i]
        w = weights[i]
        h = height[i]

        if not np.isnan(h) and not np.isnan(w):
            diff = h - mean[g]
            var[g] += w * diff * diff

    out = np.empty(n_groups)
    Neff = np.empty(n_groups)

    for g in range(n_groups):
        if sum_w[g] > 0:
            var[g] /= sum_w[g]
            Neff[g] = sum_w[g]**2 / sum_w2[g]
            out[g] = np.sqrt(var[g] / Neff[g])
        else:
            out[g] = np.nan
            Neff[g] = np.nan

    return out, Neff

@nb.njit
def uncertainty_random_numba(phase_noise, dheight, group_idx, n_groups):

    sum_phase = np.zeros(n_groups)
    sum_wp = np.zeros(n_groups)
    sum_dh_wp = np.zeros(n_groups)
    sum_dh2_wp = np.zeros(n_groups)
    count = np.zeros(n_groups)

    for i in range(len(phase_noise)):
        g = group_idx[i]
        p = phase_noise[i]
        dh = dheight[i]

        if not np.isnan(p) and not np.isnan(dh):
            wp = p * dh

            sum_phase[g] += p
            sum_wp[g] += wp
            sum_dh_wp[g] += dh * wp
            sum_dh2_wp[g] += dh * dh * wp
            count[g] += 1

    out = np.empty(n_groups)
    Neff = np.empty(n_groups)

    for g in range(n_groups):
        if count[g] > 0 and sum_wp[g] != 0:
            ratio = (sum_dh2_wp[g] / sum_wp[g]) / (sum_dh_wp[g] / sum_wp[g])
            out[g] = np.sqrt(sum_phase[g]) * np.abs(ratio)
            Neff[g] = count[g]
        else:
            out[g] = np.nan
            Neff[g] = np.nan

    return out, Neff

@nb.njit
def uncertainty_total_numba(height, weights, eff_med, eff_rare, group_idx, n_groups):

    sum_w = np.zeros(n_groups)
    sum_hw = np.zeros(n_groups)
    sum_ratio = np.zeros(n_groups)
    count = np.zeros(n_groups)

    for i in range(len(height)):
        g = group_idx[i]

        h = height[i]
        w = weights[i]
        em = eff_med[i]
        er = eff_rare[i]

        if not np.isnan(h) and not np.isnan(w) and not np.isnan(em) and not np.isnan(er):
            sum_w[g] += w
            sum_hw[g] += h * w
            sum_ratio[g] += em / er
            count[g] += 1

    mean = sum_hw / sum_w

    weighted_std = np.zeros(n_groups)

    for i in range(len(height)):
        g = group_idx[i]
        h = height[i]
        w = weights[i]

        if not np.isnan(h) and not np.isnan(w):
            diff = h - mean[g]
            weighted_std[g] += diff * w

    out = np.empty(n_groups)

    for g in range(n_groups):
        if count[g] > 0:
            first_part = (sum_ratio[g] / count[g]) / count[g]
            out[g] = np.sqrt(first_part) * np.sqrt(np.abs(weighted_std[g]))
        else:
            out[g] = np.nan

    return out, count

# =========================
# MAIN CLASS
# =========================

class SPixcFast:

    def __init__(self, filename: str):
        self.filename = filename
        self._lazy = (pl.scan_csv(
        filename,
        try_parse_dates=True,  # 👈 important
    )
    .with_columns(
        pl.col("time")
        .str.strptime(pl.Datetime, strict=False)  # 👈 tolerant parsing
    )
)
        self._filtered = self._lazy
        self._numpy_cache = None
        self._wse_by_day = None

    # =========================
    # FILTERING (POLARS LAZY)
    # =========================

    def filter_by_variable(self, conditions):
        df = self._filtered

        for var, cond in conditions.items():
            op = cond["operator"]
            thr = cond["threshold"]

            if op == "ge":
                df = df.filter(pl.col(var) >= thr)
            elif op == "gt":
                df = df.filter(pl.col(var) > thr)
            elif op == "le":
                df = df.filter(pl.col(var) <= thr)
            elif op == "lt":
                df = df.filter(pl.col(var) < thr)
            elif op == "eq":
                df = df.filter(pl.col(var) == thr)
            else:
                raise ValueError(f"Unsupported operator {op}")

        self._filtered = df

    # =========================
    # LOAD ONLY REQUIRED DATA
    # =========================

    def _load_numpy(self):

        if self._numpy_cache is not None:
            return self._numpy_cache

        df = (
            self._filtered
            .select([
                "time", "height", "geoid",
                "solid_earth_tide", "load_tide_fes", "pole_tide",
                "phase_noise_std", "dheight_dphase",
                "eff_num_medium_looks", "eff_num_rare_looks",
                "latitude", "longitude"
            ])
            .with_columns(pl.col("time").str.strptime(pl.Datetime))
            .collect()
        )

        # numpy arrays
        time = df["time"].to_numpy()
        height = df["height"].to_numpy()
        geoid = df["geoid"].to_numpy()
        solid = df["solid_earth_tide"].to_numpy()
        load = df["load_tide_fes"].to_numpy()
        pole = df["pole_tide"].to_numpy()

        phase = df["phase_noise_std"].to_numpy()
        dheight = df["dheight_dphase"].to_numpy()

        eff_med = df["eff_num_medium_looks"].to_numpy()
        eff_rare = df["eff_num_rare_looks"].to_numpy()

        lat = df["latitude"].to_numpy()
        lon = df["longitude"].to_numpy()

        wse = height - geoid - solid - load - pole
        weights = 1.0 / (phase * dheight) ** 2
        dates = time.astype("datetime64[D]")

        self._numpy_cache = (
            dates, wse, height, weights,
            phase, dheight,
            eff_med, eff_rare,
            lat, lon
        )

        return self._numpy_cache

    def _group_index(self, keys):
        unique, idx = np.unique(keys, return_inverse=True)
        return unique, idx

    # =========================
    # FAST COMPUTATION
    # =========================

    def compute_wse_by_day(self, method_uncertainty="weighted_variance"):

        (dates, wse, height, weights,
         phase, dheight, eff_med, eff_rare, _, _) = self._load_numpy()

        unique_dates, group_idx =  self._group_index(dates)
        n_groups = len(unique_dates)

        wse_mean = weighted_mean_numba(wse, weights, group_idx, n_groups)

        if method_uncertainty == "weighted_variance":
            uncertainty, Neff = uncertainty_weighted_variance_numba(
                height, weights, group_idx, n_groups
            )

        elif method_uncertainty == "random":
            uncertainty, Neff = uncertainty_random_numba(
                phase, dheight, group_idx, n_groups
            )

        elif method_uncertainty == "total":
            uncertainty, Neff = uncertainty_total_numba(
                height, weights, eff_med, eff_rare, group_idx, n_groups
            )

        else:
            raise ValueError(method_uncertainty)

        return pd.DataFrame({
            "wse_by_day": wse_mean,
            "uncertainty": uncertainty,
            "n_points": Neff
        }, index=unique_dates)

    def mask_according_to_polygon(self, polygon_path):

        import geopandas as gpd

        df = self._filtered.collect().to_pandas()

        gdf_poly = gpd.read_file(polygon_path)

        if gdf_poly.crs != "EPSG:4326":
            gdf_poly = gdf_poly.to_crs("EPSG:4326")

        gdf_pts = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )

        clipped = gpd.clip(gdf_pts, gdf_poly)

        # back to polars
        self._filtered = pl.from_pandas(clipped.drop(columns="geometry")).lazy()

        # reset cache
        self._numpy_cache = None


    def mask_according_to_polygon_fast(self, polygon_path):

        import geopandas as gpd
        from shapely import vectorized

        gdf = gpd.read_file(polygon_path)

        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        polygon = gdf.geometry.unary_union
        minx, miny, maxx, maxy = polygon.bounds

        # 👉 step 1: bounding box filter (lazy, fast)
        df = self._filtered.filter(
            (pl.col("longitude") >= minx) &
            (pl.col("longitude") <= maxx) &
            (pl.col("latitude") >= miny) &
            (pl.col("latitude") <= maxy)
        )

        # 👉 step 2: collect ONLY coordinates
        coords = df.select(["longitude", "latitude"]).collect()

        lon = coords["longitude"].to_numpy()
        lat = coords["latitude"].to_numpy()

        mask = vectorized.contains(polygon, lon, lat)

        # 👉 step 3: apply mask via index (NO pandas)
        df = df.with_row_count("row_nr")

        idx = np.where(mask)[0]

        self._filtered = df.filter(pl.col("row_nr").is_in(idx)).drop("row_nr")

        self._numpy_cache = None

    def filter_by_space_stats(self, method="normal", threshold=2):

        dates, wse, *_ = self._load_numpy()
        unique, idx = self._group_index(dates)

        keep = np.zeros_like(wse, dtype=bool)

        for g in range(len(unique)):
            m = idx == g
            data = wse[m]

            if len(data) < 2:
                continue

            if method == "normal":
                center = np.nanmean(data)
                scale = np.nanstd(data)
            else:
                med = np.nanmedian(data)
                center = med
                scale = np.nanmedian(np.abs(data - med)) * 1.4826

            keep[m] = (
                    (data >= center - threshold * scale) &
                    (data <= center + threshold * scale)
            )

        self._apply_numpy_mask(keep)

    def filter_by_temporal_stats(self, method="normal", threshold=3):

        (dates, wse, _, _, _, _, _, _, lat, lon) = self._load_numpy()

        coords = np.stack((lat, lon), axis=1)
        unique, idx = np.unique(coords, axis=0, return_inverse=True)

        keep = np.zeros_like(wse, dtype=bool)

        for g in range(len(unique)):
            m = idx == g
            data = wse[m]

            if len(data) < 2:
                continue

            if method == "normal":
                center = np.nanmean(data)
                scale = np.nanstd(data)
            else:
                med = np.nanmedian(data)
                center = med
                scale = np.nanmedian(np.abs(data - med)) * 1.4826

            keep[m] = (
                    (data >= center - threshold * scale) &
                    (data <= center + threshold * scale)
            )

        self._apply_numpy_mask(keep)

    def filter_by_wsedaystats(self, method="normal", threshold=2):

        if self._wse_by_day is None:
            self._wse_by_day = self.compute_wse_by_day()

        series = self._wse_by_day["wse_by_day"].values

        # compute stats (NumPy is fine here)
        if method == "normal":
            center = np.nanmean(series)
            scale = np.nanstd(series)
        else:
            med = np.nanmedian(series)
            center = med
            scale = np.nanmedian(np.abs(series - med)) * 1.4826

        keep_days = self._wse_by_day.index[
            (series >= center - threshold * scale) &
            (series <= center + threshold * scale)
            ]

        # 👉 convert to numpy datetime64[D]
        keep_days = np.array(keep_days, dtype="datetime64[D]")

        # 👉 stay in Polars
        self._filtered = (
            self._filtered
            .with_columns(pl.col("time").dt.date().alias("date"))
            .filter(pl.col("date").is_in(keep_days))
            .drop("date")
        )
        self._numpy_cache = None

    def _apply_numpy_mask(self, mask):

        # create row index
        df = self._filtered.with_row_count("row_nr")

        # collect only indices (lightweight)
        idx = np.where(mask)[0]

        self._filtered = df.filter(pl.col("row_nr").is_in(idx)).drop("row_nr")

        self._numpy_cache = None