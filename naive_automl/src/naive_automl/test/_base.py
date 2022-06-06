import os
import pathlib
import pandas as pd


def load_lending_club():
    lending_club = pd.read_csv(pathlib.Path(os.path.dirname(__file__)) / 'fixture' / 'Lending_Club_reduced.csv')
    lending_club_y = lending_club['is_bad'].to_numpy()
    lending_club_x = lending_club.drop(['is_bad'], axis=1)

    return lending_club_x, lending_club_y

# def assert_raise_message(exception, message, function, *args, **kwargs):
#     try:
#         function(*args, **kwargs)
#     except exceptions as e:
#         error_message = str(e)
#         if message != error_message:
#             raise AssertionError(f"Error message does equal to the expected string: {message}. Observed error message: {error_message}")
