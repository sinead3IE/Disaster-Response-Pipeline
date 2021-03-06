#importing libraries
import os
import sys
import pandas as pd
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Loading csvs and merging into one dataframe
def load_data(messages_filepath, categories_filepath):
    '''
        input:
            messages_filepath = The file path of messages dataset.
            categories_filepath = The file path  of categories dataset.
        output:
            df = The merged dataset
        '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df
    pass

## Cleaning the data
def clean_data(df):
     '''
    input:
        df = The merged dataset.
    output:
        df = Cleaned dataset.
    '''
#split the values in the categories column on the ; character so that each value becomes a separate column
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)

#Select the first row of the categories dataframe
#use the first row of categories dataframe to create column names for the categories data.
    row = categories.loc[0]

# use this first row of categories dataframe to create column names for the categories data
    category_colnames = row.apply(lambda x: x[:-2])

# rename the columns of categories
    categories.columns = category_colnames

# Convert category values to just numbers 0 or 1.
    for column in categories:
# set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
# convert column from string to numeric
        categories[column] = categories[column].astype(int)
    categories.replace(2, 1, inplace=True)

#drop the categories column from the df dataframe since it is no longer needed
    df.drop('categories', axis=1, inplace = True)

#concatenate df and categories data frames.
    df = pd.concat([df, categories], axis=1)

#drop the duplicates
    df.drop_duplicates(subset = 'id', inplace = True)
    return df
    pass

## Saving the dataset to SQLite Database
def save_data(df, database_filename):
    '''
        input:
            df = Cleaned dataset.
        output:
            ETL_Table in ETL_DB.db in SQLite.
        '''
    engine = create_engine('sqlite:///ETL_DB.db')
    df.to_sql('ETL_Table', engine, index=False)
    pass

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
