import numpy as np
from typing import Tuple
import torch
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
import holidays
import pytz
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
import config
localtime = pytz.timezone('CET')

def load_train_data(city_name: str = '') -> pd.DataFrame:
    train = pd.read_excel('../demand_temp.xlsx')[['time', 'place', 'temperature', 'demand']].sort_values(by=['place', 'time']).set_index(keys=['time'])
    train = train[train['place'] == city_name] if city_name else train
    return train


def filter_global_train_data(train_data: pd.DataFrame) -> pd.DataFrame:
    train_data_count = train_data.groupby([train_data.index.year, train_data.index.month, train_data.index.date, 'place'])['demand'].count().reset_index(
    level=0, drop=True).reset_index().rename(columns={'level_1': 'date', 'demand': 'count'})[['date', 'place', 'count']]
    summary_book = train_data_count[train_data_count['count'] < 24]
    
    print(summary_book)


    train_data_reset = train_data.reset_index()
    filtered_df = []
    for _, row in summary_book.iterrows():
        filtered_df.extend(train_data_reset[(train_data_reset['place'] == row.place) & (
            train_data_reset['time'].dt.date == row.date)].index)

    train_data_new = train_data_reset.loc[~train_data_reset.index.isin(
        filtered_df)]
    train_data_new = train_data_new.set_index(keys=['time'])
    return train_data_new

    
def filter_train_data(train_data: pd.DataFrame) -> pd.DataFrame:
    train_data_count = train_data.groupby([train_data.index.year, train_data.index.month, train_data.index.date])['demand'].count().reset_index(
    level=0, drop=True).reset_index().rename(columns={'level_1': 'date', 'demand': 'count'})[['date', 'count']]
    summary_book = train_data_count[train_data_count['count'] < 24]
    
    train_data_reset = train_data.reset_index()
    filtered_df = []
    for _, row in summary_book.iterrows():
        filtered_df.extend(train_data_reset[(train_data_reset['time'].dt.date == row.date)].index)

    train_data_new = train_data_reset.loc[~train_data_reset.index.isin(filtered_df)]
    train_data_new = train_data_new.set_index(keys=['time'])
    return train_data_new

# scale(Standard scaler) train and test data
def scale(data: pd.DataFrame) -> pd.DataFrame:
    # scaler
    x_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data[['temperature']])
    y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data[['demand']])

    # transform data
    x_transformed = pd.DataFrame(x_scaler.transform(
        data[['temperature']]), columns=['temperature'], index=data.index)
    y_transformed = pd.DataFrame(y_scaler.transform(
        data[['demand']]), columns=['demand'], index=data.index)

    return x_scaler, y_scaler, pd.concat([data[data.columns.difference(['temperature', 'demand'])], x_transformed, y_transformed], axis=1)

def one_hot_encoding(data_frame: pd.DataFrame, categorical_columns: list[str]):
    data_frame = data_frame.copy(deep=True)

    for column in categorical_columns:
        slice_data_frame = data_frame[[column]]
        dummy = pd.get_dummies(slice_data_frame, columns=[
                               column], prefix='', prefix_sep='').iloc[:, :-1]
        data_frame.drop([column], axis=1, inplace=True)
        data_frame = pd.concat([data_frame, dummy], axis=1)
    return data_frame


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

def add_feature(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy(deep=True)
    train["month_sin"] = sin_transformer(12).fit_transform(train.index.month)
    train["month_cos"] = cos_transformer(12).fit_transform(train.index.month)
    train["day_sin"] = sin_transformer(365).fit_transform(train.index.dayofyear)
    train["day_cos"] = cos_transformer(365).fit_transform(train.index.dayofyear)
    train["hour_sin"] = sin_transformer(23).fit_transform(train.index.hour)
    train["hour_cos"] = cos_transformer(23).fit_transform(train.index.hour)
    train["business_hour"] = np.where(((train.index.hour >= 8) & (train.index.hour <= 16)), 1, 0)
    train["season"] = np.select([((train.index.month == 12) | (train.index.month <= 2)), ((train.index.month >= 3) & (train.index.month <= 5)), ((
        train.index.month >= 6) & (train.index.month <= 8)), ((train.index.month >= 9) & (train.index.month <= 11))], ['winter', 'spring', 'summer', 'autumn'])
    train["weekend"] = train.apply(lambda row: int(bool(row.name.day_of_week > 4)), axis=1)
    train["daylight"] = train.apply(lambda row: int(bool(localtime.localize(row.name.to_pydatetime()).dst())), axis=1)
    norge_holidays = [item[0].strftime('%Y-%m-%d') for item in holidays.Norway(years=train.index.year.unique()).items()]
    train['holiday'] = train.apply(lambda row: int(row.name.strftime('%Y-%m-%d') in norge_holidays), axis=1)
    return train

# frame a sequence as a supervised learning problem
def add_lag(data, lag=1):
    data = data.copy(deep=True)
    index = data.columns.get_loc('demand')
    for i in range(1, lag + 1):
        data.insert(index, f'demand-{i}',
                    data['demand'].shift(i, fill_value=0.000))
        index = data.columns.get_loc(f'demand-{i}')
    # data = data.iloc[lag:, :]
    return data

def add_baseline_feature(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy(deep=True)
    data.insert(data.columns.get_loc('demand-1'), 'base_line_demand',
                data['demand'].shift(24 * 7))  # Last week same hour
    data['base_line_demand'] = data['base_line_demand'].fillna(data['demand'])
    return data

# split a multivariate sequence into samples
def split_sequences(df, n_steps):
    X, y = [], []
    for i in range(len(df)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(df) - 1:
            break
        # gather input and output parts of the pattern
        X.append(df.iloc[i:end_ix, :-1]), y.append(df.iloc[end_ix-1, -1])
    return np.array(X), np.array(y)

# split a multivariate sequence into samples
def split_sequences_global(df, n_steps, places: list[str]):
    X, y = [], []
    for city in places:
        df_new = df[df[city] == 1]
        for i in range(len(df_new)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(df_new) - 1:
                break
            # gather input and output parts of the pattern
            X.append(df_new.iloc[i:end_ix, :-1]
                     ), y.append(df_new.iloc[end_ix-1, -1])
    return np.array(X), np.array(y)


def get_batch_data(x, y, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(x), batch_size):
        yield (x[i:i + batch_size], y[i:i + batch_size])

def within_range(target: any, target_range: Tuple[float, float]) -> bool:
    return target_range[0] <= target if len(target_range) <= 1 else target_range[0] <= target and target <= target_range[1]

# Send the training data
def add_random_noise(X: list[any], light_blue_range: Tuple[float, float], dark_red_range: Tuple[float, float], model: any) -> list[any]:
    # Loop through the data
    for data in X:
        # For each and every data, find the heat-map score
        input_val = torch.FloatTensor([data]).to(device=config.DEVICE)
        # input_val = [1, timesteps, n_features] or [1, timesteps, n_features-1] according to if we include or not the target feature

        res = model(input_val, dropout=False)
        # Replace the data with a random noise

        # Heat map scores
        # If heatmap is light blue or dark red, could means they were relevant
        scores = res[2].cpu().detach().numpy()
        scores = scores[0, :, :]
        scores_ = np.transpose(scores)[0]

        # Loop through the window data and add the random noise
        for index_ in range(data.shape[0]):
            # add the noise
            if not (within_range(scores_[index_], light_blue_range) or within_range(scores_[index_], dark_red_range)):
                noise = np.random.normal(2, 0.1, [data.shape[1], 1])
                noise = noise.reshape(1, -1)[0]
                data[index_] = data[index_] + noise
    return X

# From the PDP we can conculde that, at the begianing of the week and year consumption decrease, also, on daylignt increase, during weekdays and normal working season consumption decrease
# So lets adjust X and see the performance
def adjust_train_data(train_data: pd.DataFrame):
    train_data_new = train_data.copy()
    train_data_new.loc[((train_data_new['summer'] == 1) & (train_data_new['daylight'] == 1) & (train_data_new['day_sin'].between(0.25, 0.50)) & (train_data_new['hour_sin'].between(0.00, 0.25))), 'demand'] = -1.00
    return train_data_new

def show_density_plot(df: pd.DataFrame) -> None:
    ax = df['demand'].plot.density(color='green')
    mean_val =  np.mean(df['demand'])
    # Annotate points
    ax.annotate('mean', xy=(mean_val, 0.008))
    # vertical dotted line originating at mean value
    plt.axvline(mean_val, linestyle='dashed', linewidth=2)