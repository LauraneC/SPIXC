import os
import re
import geopandas as gpd
import glob
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

def load_polygon_map_from_folder(folder_path: str, base_name: str = "Lac_Joux") -> dict[
    tuple[int, int], gpd.GeoDataFrame]:
    """
    Load shapefiles named like 'base_name_YYYYMM.shp' from a folder and return
    a dictionary {(year, month): GeoDataFrame}.
    :param folder_path : Path to the folder containing the shapefiles.
    :param base_name : Base name of the shapefiles before the date component.

    :return  Dictionary mapping (year, month) -> GeoDataFrame in EPSG:4326.
    """
    polygon_map = {}
    pattern = re.compile(
        rf"{re.escape(base_name)}_(\d{{4}})(\d{{2}})(\d{{2}})?\.shp$"
    )

    shp_files = glob.glob(os.path.join(folder_path, "*.shp"))
    if len(shp_files)==0: shp_files = glob.glob(os.path.join(folder_path, "*.gpkg"))
    for pathname in shp_files:
        fname = os.path.basename(pathname)
        match = pattern.match(fname)
        if not match:
            print(f"Skipping {fname}")
            continue

        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3)) if match.group(3) else None



        full_path = os.path.join(folder_path, fname)

        try:
            gdf = gpd.read_file(full_path)
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            polygon_map[(year, month,day)] = gdf
        except Exception as e:
            print(f"Failed to read {fname}: {e}")

    return polygon_map

def convert_npy(dyn_path:str):
    S1_dyn = np.load(dyn_path)
    S1_dyn_converted = pd.DataFrame({
        'time_str': pd.date_range("2016-01-01", "2024-12-15", freq="MS") + pd.Timedelta(days=14),
        'area': S1_dyn.flatten() / 10 ** 6
    })
    S1_dyn_converted.set_index('time_str', inplace=True)
    return S1_dyn_converted

#%%
def fit_linear_regression(X,y):
    """
    Fit a linear regression model
    :param X:
    :param y:
    :return: x prediction, y prediction, linreg
    """
    # Fit linear regression
    linreg = LinearRegression()
    linreg.fit(X, y)

    # Predictions
    x_pred = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    y_pred = linreg.predict(x_pred)
    return x_pred, y_pred, linreg

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from scipy import stats

def fit_polynomial_regression(X, y, degree=2, alpha=0.05):
    """
    Fit a polynomial regression model and return an inlier mask for points within the 95% confidence interval.

    :param X: Input features (1D array-like or 2D array-like)
    :param y: Target values (1D array-like)
    :param degree: Degree of the polynomial (default: 2)
    :param alpha: Significance level for confidence interval (default: 0.05)
    :return: x_pred, y_pred, model, inlier_mask
    """
    # Reshape X if it's 1D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Fit linear regression
    linreg = LinearRegression()
    linreg.fit(X_poly, y)

    # Predictions for the original X
    y_pred = linreg.predict(X_poly)

    # Calculate residuals and standard error
    residuals = y - y_pred
    mse = np.mean(residuals**2)
    std_err = np.sqrt(mse)

    # Calculate confidence interval
    n = len(X)
    t_critical = stats.t.ppf(1 - alpha/2, df=n - degree - 1)
    margin_of_error = t_critical * std_err

    # Inlier mask: points within the confidence interval
    inlier_mask = np.abs(residuals) <= margin_of_error

    # Predictions for plotting
    x_pred = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    X_pred_poly = poly.transform(x_pred)
    y_pred_plot = linreg.predict(X_pred_poly)

    return x_pred, y_pred_plot, linreg, inlier_mask

def fit_RANSAC(X,y):
    """
    RANSAC regressor
    :param X:
    :param y:
    :return: x prediction, y prediction, linreg, inliermask
    """

    # Base estimator (ordinary least squares)
    ols = LinearRegression()

    # RANSAC wrapper
    ransac = RANSACRegressor(
        estimator=ols,
        min_samples=0.5,        # or an int, e.g. 50
        residual_threshold=None,  # let RANSAC estimate it
        random_state=42
    )

    # Fit
    ransac.fit(X, y)

    # Extract inliers
    inlier_mask = ransac.inlier_mask_
    X_inliers = X[inlier_mask]
    y_inliers = y[inlier_mask]

    # Final OLS fit on inliers only (optional but explicit)
    linreg = LinearRegression()
    linreg.fit(X_inliers, y_inliers)

    # Predictions (if needed)
    x_pred = np.linspace(X_inliers.min(), X_inliers.max(), 100).reshape(-1, 1)
    y_pred = linreg.predict(x_pred)
    return x_pred, y_pred, linreg, inlier_mask

