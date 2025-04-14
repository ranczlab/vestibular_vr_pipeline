import pandas as pd
import statsmodels.formula.api as smf

def fit_and_compare_models(data, dependent_variable, independent_vars):
    """
    Fit and compare models using specified independent variables and a dependent variable.

    Parameters:
        data (pd.DataFrame): Data containing dependent and independent variables.
        dependent_variable (str): Name of the dependent variable to model.
        independent_vars (list): List of independent variables to consider in models.

    Returns:
        dict: A dictionary of fitted models, their AIC values, and a summary DataFrame.
    """
    # Check if dependent variable exists
    if dependent_variable not in data.columns:
        raise ValueError(f"Dependent variable '{dependent_variable}' not found in the DataFrame.")

    # Rename the dependent variable if necessary
    safe_dependent_variable = dependent_variable.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")

    # Rename the column in the DataFrame temporarily for modeling
    data = data.rename(columns={dependent_variable: safe_dependent_variable})

    # Check if all independent variables exist
    for var in independent_vars:
        if var not in data.columns:
            raise ValueError(f"Independent variable '{var}' not found in the DataFrame.")

    # Build the formula
    independent_formula = " + ".join(independent_vars)
    formula = f"{safe_dependent_variable} ~ {independent_formula}"

    # Fit the model
    model = smf.ols(formula=formula, data=data).fit()

    # Return results
    return {
        "model_summary": model.summary(),
        "aic": model.aic,
        "bic": model.bic
    }

