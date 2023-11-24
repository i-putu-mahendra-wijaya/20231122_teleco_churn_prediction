from typing import List, Dict

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

from src.classes.Outlier import Outlier
from src.classes.TrData import TrData
from src.classes.NpData import NpData


def detect_outliers_iqr(
        data_ds: pd.Series
) -> List[Outlier]:
    """
    This function is to detect outliers in a column of pandas dataframe

    :param data_ds: is the pd.Series taken from a single column from the dataframe
    :return: list of NamedTuple of (idx, outlier_val)
    """

    outliers: List[Outlier] = []

    q1: float = data_ds.quantile(0.25)
    q3: float = data_ds.quantile(0.75)
    iqr: float = q3 - q1

    lower_tail: float = q1 - (1.5 * iqr)
    upper_tail: float = q3 + (1.5 * iqr)

    for each_idx, each_val in data_ds.items():
        if (each_val < lower_tail) or (each_val > upper_tail):
            outliers.append(Outlier(idx=each_idx, outlier_val=each_val))

    return outliers


def augment_raw_data(
        raw_data: pd.DataFrame,
        column_name: str
) -> pd.DataFrame:
    """
    Augment `raw_dat` dataframe using `column_name` distribution.

    The output of this function is to make the distribution of `column_name` to have about equal proportion for each categories

    The function assumes that `column_name` column is categorical

    :param raw_data:
    :param column_name:
    :return: augmented dataframe
    """

    target_column: pd.Series = raw_data[column_name].copy()

    # get the proportion of each category
    category_counts: pd.Series = target_column.value_counts()

    # get the number of categories
    num_categories: int = len(category_counts)

    # calculate proportion for each category, to make sure equal distribution
    target_proportion: float = 1 / num_categories

    # calculate the number of samples to add for each categories
    samples_to_add: Dict[str, int] = {}
    for categ, count in category_counts.items():
        samples_to_add[categ] = int((target_proportion * raw_data.shape[0]) - count)

    # identify the majority and minority classes
    minority_class: str = max(samples_to_add, key=lambda x: samples_to_add[x])
    majority_class: str = min(samples_to_add, key=lambda x: samples_to_add[x])

    # count num rows to add per instance found
    num_rows_to_add: int = int(category_counts[minority_class] / samples_to_add[minority_class])

    # augment the data
    augmented_data: List = []
    for each_idx, each_row in raw_data.iterrows():
        if each_row[column_name] == minority_class:
            augmented_data.append(each_row.to_list())
            for _ in range(num_rows_to_add):
                augmented_data.append(each_row.to_list())
        else:
            augmented_data.append(each_row.to_list())

    # create new dataframe from the augmented data
    augmented_df: pd.DataFrame = pd.DataFrame(augmented_data, columns=raw_data.columns)

    return augmented_df


def convert_churn_categ_to_int(
        df_categ: pd.DataFrame
) -> pd.DataFrame:
    """
    Function to convert `Yes` and `No` in `Churn` column in df_categ to `1` and `0` respectively

    :param df_categ:
    :return: converted dataframe
    """

    df_categ["Churn"]= df_categ["Churn"].replace(
        {
            "Yes": 1,
            "No": 0
        }
    )

    df_categ["Churn"] = df_categ["Churn"].astype("int64")

    return df_categ


def cleanse_transform_data(
        rw_data: pd.DataFrame
) -> TrData:
    """
     Function to wrap steps to cleanse & transform data for teleco_churn prediction task

     The task of cleansing and transforming include:

     1. Convert `TotalCharges` column to numerical --> this will gives `null` which need to be handled later
     2. Augment data to make `Churn` target column more balanced
     3. Convert `SeniorCitizen` column to categorical
     4. Furthermore, once we have augmented the data, we usually convert the categorica columns into one-hot encoded columns

    :param rw_data: the raw_data dataframe
    :return:  cleaned and transformed dataframe
    """

    # 1.  Convert `TotalCharges` column to numerical
    rw_data["TotalCharges"] = pd.to_numeric(
        rw_data["TotalCharges"],
        errors="coerce"
    )

    # 2. Augment data so that `Curn` column have almost equal distribution
    _raw_data: pd.DataFrame = augment_raw_data(
        raw_data=rw_data,
        column_name="Churn"
    )

    rw_data: pd.DataFrame = _raw_data.copy()

    # 3. Convert `SeniorCitizen` to categorical
    rw_data["SeniorCitizen"] = rw_data["SeniorCitizen"].astype("object")

    # -- drop `customerID` column and `Churn` column before we convert categ columns to one-hot
    customer_id: pd.DataFrame = rw_data["customerID"].to_frame()
    churn_label: pd.DataFrame = rw_data["Churn"].to_frame()
    churn_label = convert_churn_categ_to_int(df_categ=churn_label)
    rw_data.drop(labels=["customerID", "Churn"], axis=1, inplace=True)

    # 4. Convert categorical columns to one-hot encoded dummy variables
    cat_columns: List[str] = rw_data.select_dtypes(include=["object"]).columns
    num_columns: List[str] = rw_data.select_dtypes(include=["int64", "float64"]).columns

    df_ohe: pd.DataFrame = pd.get_dummies(rw_data, columns=cat_columns, dtype="int64")

    return TrData(
        customer_id=customer_id,
        churn_label=churn_label,
        transformed_df=df_ohe
    )


def impute_scale_and_convert_to_numpy(
        ohe_df: pd.DataFrame,
        columns_with_nulls: List[str],
        impute_val: List,
        scaler_folder: str,
        churn_df: pd.DataFrame
) -> NpData:
    """

    :param ohe_df: dataframe to impute and scale
    :param columns_with_nulls:  the column which has null values
    :param impute_val:  list of values to replace the nulls
    :param scaler_folder:  folder where we put the scaler
    :param churn_df: dataframe of the churn target
    :return: A tuple of scaled feature numpy array and churn target numpy array
    """

    # Impute null values
    for each_cols, im_val in zip(columns_with_nulls, impute_val):
        ohe_df[[each_cols]] = ohe_df[[each_cols]].fillna(value=im_val, inplace=False)

    # scale columns to between 0 and 1
    df_range: pd.DataFrame = ohe_df.agg(func=["min", "max"])
    columns_to_scale: List[str] = df_range.columns[df_range.loc["max"] > 1].to_list()

    for each_cols in columns_to_scale:
        ds: pd.Series = ohe_df[[each_cols]].copy()

        file_path: str = f"{scaler_folder}/mm_scaler_{each_cols}.pkl"
        mm_scaler: MinMaxScaler = joblib.load(filename=file_path)
        ds_scaled = mm_scaler.transform(ds)
        ohe_df[[each_cols]] = ds_scaled

    ohe_np: np.ndarray = ohe_df.to_numpy()
    churn_np: np.ndarray = churn_df.to_numpy()

    return NpData(
        ohe_np=ohe_np,
        churn_np=churn_np
    )