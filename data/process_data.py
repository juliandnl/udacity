import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Load data from csv"""
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath, sep =",")
    df = messages.merge(categories, how = "left", on = "id")
    return df, categories


def clean_data(df, categories):
    """ clean data"""
    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(";", expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = [re.split("-",word)[0] for word in row.tolist()]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Get the values from the last character of the string
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].apply(int)

    # Drop col
    df.drop("categories", axis = 1, inplace = True)
    df = df.merge(categories, how = "inner", left_index = True, right_index = True)

    # Drop duplicates
    df = df.drop_duplicates("id")
    return df


def save_data(df, database_filename):
    """ save data to sql db"""
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('emergency', engine, index=False)


def main():
    """ execute the processing of the data"""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
