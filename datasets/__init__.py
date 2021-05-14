from sklearn.datasets import make_swiss_roll
import pandas as pd
from dataclasses import dataclass


@dataclass
class Dataset:
    df: pd.DataFrame
    display_index: bool = False
    color_index: bool = False


def get_datasets():

    cars = pd.read_csv('datasets/cars.csv', header=None)
    cars[0] = cars[0].str.strip("'")
    cars.set_index(0, inplace=True)

    swiss_roll = make_swiss_roll()
    swiss_roll = pd.DataFrame(swiss_roll[0], index=swiss_roll[1])

    return {
        'Swiss Roll': Dataset(swiss_roll, color_index=True),
        'Cars': Dataset(cars, display_index=True)
    }
