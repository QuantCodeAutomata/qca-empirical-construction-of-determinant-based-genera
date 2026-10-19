"""Visualization and diagnostic functions for the generalized realized variance study.

All functions:
  - Accept an optional ``save_path`` parameter; figures are saved when provided.
  - Call ``plt.tight_layout()`` before saving.
  - Close all figures after saving to prevent memory leaks.
  - Use the non-interactive 'Agg' backend, compatible with headless execution.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Return distribution plots
# ---------------------------------------------------------------------------

def plot_return_distributions(
    returns_dict: dict,
    save_path: str = None,
) -> None:
    """Plot histogram and KDE of returns for each asset in each portfolio group.

    Produces a grid of panels with rows corresponding to portfolio groups and
    columns corresponding to individual assets (columns of each DataFrame).

    Parameters
    ----------
    returns_dict : dict
        Mapping of group name -> pd.DataFrame (or pd.Series) of returns.
        DataFrame columns are treated as individual assets.
    save_path : str, optional
        File path at which to save the figure. Supports any matplotlib-
        supported format (e.g. '.png', '.pdf').
    """
    if not returns_dict:
        return

    groups = list(returns_dict.keys())
    n_groups = len(groups)

    # Coerce every entry to a DataFrame and determine max number of assets
    frames = {}
    for g in groups:
        obj = returns_dict[g]
        if isinstance(obj, pd.Series):
            name = obj.name if obj.name is not None else 'asset'
            frames[g] = obj.to_frame(name=str(name))
        else:
            frames[g] = pd.DataFrame(obj)

    n_cols = max(len(df.columns) for df in frames.values())

    fig, axes = plt.subplots(
        n_groups, n_cols,
        figsize=(5 * n_cols, 4 * n_groups),
        squeeze=False,
    )

    for row_idx, group in enumerate(groups):
        df = frames[group]
        asset_cols = list(df.columns)

        for col_idx, col in enumerate(asset_cols):
            ax = axes[row_idx, col_idx]
            series = df[col].dropna()
            sns.histplot(series, kde=True, ax=ax, color='steelblue', alpha=0.7)
            ax.set_title(f"{group} — {col}", fontsize=10)
            ax.set_xlabel('Return')
            ax.set_ylabel('Count')

        # Hide any unused panels in this row
        for col_idx in range(len(asset_cols), n_cols):
            axes[row_idx, col_idx].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


# ---------------------------------------------------------------------------
# Correlation heatmaps
# ---------------------------------------------------------------------------

def plot_correlation_heatmaps(
    corr_dict: dict,
    save_path: str = None,
) -> None:
    """Plot annotated correlation heatmaps for each portfolio group.

    Parameters
    ----------
    corr_dict : dict
        Mapping of group name -> correlation matrix.  Each value may be a
        pd.DataFrame or a np.ndarray; if an ndarray, integer column/row labels
        are used automatically.
    save_path : str, optional
        File path at which to save the figure.
    """
    if not corr_dict:
        return

    groups = list(corr_dict.keys())
    n_groups = len(groups)
    ncols = min(n_groups, 3)
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
    )

    for idx, group in enumerate(groups):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        corr = corr_dict[group]
        if not isinstance(corr, pd.DataFrame):
            corr = pd.DataFrame(corr)
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1.0,
            vmax=1.0,
            square=True,
            ax=ax,
        )
        ax.set_title(f"Correlation — {group}", fontsize=10)

    # Hide unused panels
    total_panels = nrows * ncols
    for idx in range(n_groups, total_panels):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


# ---------------------------------------------------------------------------
# Cumulative return plots
# ---------------------------------------------------------------------------

def plot_cumulative_returns(
    returns_dict: dict,
    save_path: str = None,
) -> None:
    """Plot cumulative returns for each portfolio group.

    Cumulative returns are computed as the running sum of the provided series
    (assumes log-returns or additive returns). Each asset within a group is
    shown as a distinct line on the same axes.

    Parameters
    ----------
    returns_dict : dict
        Mapping of group name -> pd.DataFrame (or pd.Series) of returns.
    save_path : str, optional
        File path at which to save the figure.
    """
    if not returns_dict:
        return

    groups = list(returns_dict.keys())
    n_groups = len(groups)
    ncols = min(n_groups, 3)
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows),
        squeeze=False,
    )

    for idx, group in enumerate(groups):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        data = returns_dict[group]
        if isinstance(data, pd.Series):
            name = data.name if data.name is not None else 'asset'
            data = data.to_frame(name=str(name))
        else:
            data = pd.DataFrame(data)

        for asset_col in data.columns:
            series = data[asset_col].dropna()
            cumulative = series.cumsum()
            ax.plot(cumulative.index, cumulative.values, label=str(asset_col))

        ax.set_title(f"Cumulative Returns — {group}", fontsize=10)
        ax.set_xlabel('Date / Observation')
        ax.set_ylabel('Cumulative Return')
        ax.legend(fontsize=8)

    # Hide unused panels
    total_panels = nrows * ncols
    for idx in range(n_groups, total_panels):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


# ---------------------------------------------------------------------------
# Determinant-based generalized variance time series
# ---------------------------------------------------------------------------

def plot_det_series(
    det_dict: dict,
    save_path: str = None,
) -> None:
    """Plot time series of determinant-based generalized realized variance.

    Automatically switches to a log-scale y-axis when the ratio of the maximum
    to the minimum positive value exceeds 100 across all series, indicating
    values spanning multiple orders of magnitude.

    Parameters
    ----------
    det_dict : dict
        Mapping of group name -> pd.Series of determinant-based variance values.
    save_path : str, optional
        File path at which to save the figure.
    """
    if not det_dict:
        return

    groups = list(det_dict.keys())
    n_groups = len(groups)
    ncols = min(n_groups, 3)
    nrows = int(np.ceil(n_groups / ncols))

    # Determine whether log scale is appropriate across all series
    all_positive_vals = np.concatenate([
        np.asarray(s.values if isinstance(s, pd.Series) else s, dtype=float)
        for s in det_dict.values()
    ])
    positive_mask = all_positive_vals > 0.0
    use_log = (
        positive_mask.any()
        and (all_positive_vals[positive_mask].max() / all_positive_vals[positive_mask].min()) > 100.0
    )

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 4 * nrows),
        squeeze=False,
    )

    for idx, group in enumerate(groups):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        series = det_dict[group]
        if not isinstance(series, pd.Series):
            series = pd.Series(np.asarray(series, dtype=float))

        ax.plot(series.index, series.values, linewidth=1.0, color='royalblue')
        ax.set_title(f"Det-based Generalized Variance — {group}", fontsize=10)
        ax.set_xlabel('Date / Observation')
        ax.set_ylabel('Generalized Realized Variance')
        if use_log:
            ax.set_yscale('log')

    # Hide unused panels
    total_panels = nrows * ncols
    for idx in range(n_groups, total_panels):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


# ---------------------------------------------------------------------------
# Fitted vs realized comparison
# ---------------------------------------------------------------------------

def plot_fitted_vs_realized(
    det_series: pd.Series,
    fitted_heston: np.ndarray,
    fitted_bns: np.ndarray,
    group_name: str,
    save_path: str = None,
) -> None:
    """Plot realized generalized variance alongside Heston and BNS model fits.

    Parameters
    ----------
    det_series : pd.Series
        Realized determinant-based generalized variance.  The index is used as
        the x-axis (e.g. dates or time horizons T).
    fitted_heston : np.ndarray
        Fitted values from the Heston model, same length as det_series.
    fitted_bns : np.ndarray
        Fitted values from the BNS model, same length as det_series.
    group_name : str
        Portfolio group label used in the figure title.
    save_path : str, optional
        File path at which to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = det_series.index
    ax.plot(x, det_series.values, color='black', linewidth=1.5, label='Realized', zorder=3)
    ax.plot(x, fitted_heston, color='steelblue', linewidth=1.5,
            linestyle='--', label='Heston fit', zorder=2)
    ax.plot(x, fitted_bns, color='tomato', linewidth=1.5,
            linestyle='-.', label='BNS fit', zorder=2)

    ax.set_title(f"Realized vs Fitted Generalized Variance — {group_name}", fontsize=12)
    ax.set_xlabel('Date / Maturity T')
    ax.set_ylabel('Generalized Realized Variance')
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: str = None,
) -> None:
    """Bar chart comparing RMSE, APE, and AAE for Heston vs BNS across portfolios.

    Expects ``comparison_df`` to contain at minimum:
      - A column identifying the portfolio/group (``'portfolio'`` or the first column).
      - A column identifying the model (``'model'``), if both models appear together.
      - Numeric columns for error metrics: any subset of ``'rmse'``, ``'ape'``, ``'aae'``.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Error metrics table.  Rows represent (portfolio, model) combinations;
        columns include metric names and identifying columns.
    save_path : str, optional
        File path at which to save the figure.
    """
    if comparison_df.empty:
        return

    metric_cols = [c for c in ('rmse', 'ape', 'aae') if c in comparison_df.columns]
    if not metric_cols:
        raise ValueError(
            "comparison_df must contain at least one of: 'rmse', 'ape', 'aae'."
        )

    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False)

    # Identify the group/portfolio column (first non-metric, non-model column)
    reserved = set(metric_cols) | {'model'}
    group_col = next(
        (c for c in comparison_df.columns if c not in reserved),
        comparison_df.columns[0],
    )
    has_model_col = 'model' in comparison_df.columns

    for col_idx, metric in enumerate(metric_cols):
        ax = axes[0, col_idx]
        if has_model_col:
            sns.barplot(
                data=comparison_df,
                x=group_col,
                y=metric,
                hue='model',
                ax=ax,
                palette='Set2',
            )
            ax.legend(title='Model', fontsize=8)
        else:
            sns.barplot(
                data=comparison_df,
                x=group_col,
                y=metric,
                ax=ax,
                color='steelblue',
            )

        ax.set_title(metric.upper(), fontsize=11)
        ax.set_xlabel('Portfolio', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')
