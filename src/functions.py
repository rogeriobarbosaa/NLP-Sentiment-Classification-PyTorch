def token_outlier(column):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)

    iqr = q3 - q1

    upper_limit = column - 1.5*iqr
    lower_limit = column + 1.5*iqr

    outliers = column < lower_limit | column > upper_limit

    return outliers