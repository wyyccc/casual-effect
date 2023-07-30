import pandas as pd
import numpy as np
import tensorflow as tf

def qini_score(
    df,
    outcome_col="y",
    treatment_col="t",
    treatment_effect_col="tau",
    normalize=True,
    *args,
    **kwarg,
):
    """Calculate the Qini score: the area between the Qini curves of a model and random.

    For details, see Radcliffe (2007), `Using Control Group to Target on Predicted Lift:
    Building and Assessing Uplift Models`

     Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not

    Returns:
        (float): the Qini score
    """

    qini = get_qini(df, outcome_col, treatment_col, treatment_effect_col, normalize)
    return (qini.sum(axis=0) - qini[RANDOM_COL].sum()) / qini.shape[0]

RANDOM_COL = "Random"

def get_qini(
    df,
    outcome_col="y",
    treatment_col="t",
    treatment_effect_col="tau",
    normalize=False,
    random_seed=42,
):
    """Get Qini of model estimates in population.

    If the true treatment effect is provided (e.g. in synthetic data), it's calculated
    as the cumulative gain of the true treatment effect in each population.
    Otherwise, it's calculated as the cumulative difference between the mean outcomes
    of the treatment and control groups in each population.

    For details, see Radcliffe (2007), `Using Control Group to Target on Predicted Lift:
    Building and Assessing Uplift Models`

    For the former, `treatment_effect_col` should be provided. For the latter, both
    `outcome_col` and `treatment_col` should be provided.

    Args:
        df (pandas.DataFrame): a data frame with model estimates and actual data as columns
        outcome_col (str, optional): the column name for the actual outcome
        treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
        treatment_effect_col (str, optional): the column name for the true treatment effect
        normalize (bool, optional): whether to normalize the y-axis to 1 or not
        random_seed (int, optional): random seed for numpy.random.rand()

    Returns:
        (pandas.DataFrame): cumulative gains of model estimates in population
    """
    assert (
        (outcome_col in df.columns)
        and (treatment_col in df.columns)
        or treatment_effect_col in df.columns
    )

    df = df.copy()
    np.random.seed(random_seed)
    random_cols = []
    for i in range(10):
        random_col = "__random_{}__".format(i)
        df[random_col] = np.random.rand(df.shape[0])
        random_cols.append(random_col)

    model_names = [
        x
        for x in df.columns
        if x not in [outcome_col, treatment_col, treatment_effect_col]
    ]

    qini = []
    for i, col in enumerate(model_names):
        sorted_df = df.sort_values(col, ascending=False).reset_index(drop=True)
        sorted_df.index = sorted_df.index + 1
        sorted_df["cumsum_tr"] = sorted_df[treatment_col].cumsum()

        if treatment_effect_col in sorted_df.columns:
            # When treatment_effect_col is given, use it to calculate the average treatment effects
            # of cumulative population.
            l = (
                sorted_df[treatment_effect_col].cumsum()
                / sorted_df.index
                * sorted_df["cumsum_tr"]
            )
        else:
            # When treatment_effect_col is not given, use outcome_col and treatment_col
            # to calculate the average treatment_effects of cumulative population.
            sorted_df["cumsum_ct"] = sorted_df.index.values - sorted_df["cumsum_tr"]
            sorted_df["cumsum_y_tr"] = (
                sorted_df[outcome_col] * sorted_df[treatment_col]
            ).cumsum()
            sorted_df["cumsum_y_ct"] = (
                sorted_df[outcome_col] * (1 - sorted_df[treatment_col])
            ).cumsum()

            l = (
                sorted_df["cumsum_y_tr"]
                - sorted_df["cumsum_y_ct"]
                * sorted_df["cumsum_tr"]
                / sorted_df["cumsum_ct"]
            )

        qini.append(l)

    qini = pd.concat(qini, join="inner", axis=1)
    qini.loc[0] = np.zeros((qini.shape[1],))
    qini = qini.sort_index().interpolate()

    qini.columns = model_names
    qini[RANDOM_COL] = qini[random_cols].mean(axis=1)
    qini.drop(random_cols, axis=1, inplace=True)

    if normalize:
        qini = qini.div(np.abs(qini.iloc[-1, :]), axis=1)

    return qini


