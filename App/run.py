import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
import pickle
from sqlalchemy import create_engine

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../Data/ETL_DB.db')
df = pd.read_sql_table('ETL_Table', engine)

# load model
model = pickle.load(open('udacity_model.sav', 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

 # extract data needed for visuals
# Udacity Visual
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


 # OwnVisual_1: Plotting of Categories Distribution in "News" Genre
    news_cat = df[df.genre == 'news']
    news_cat_counts = (news_cat.mean() * news_cat.shape[0]).sort_values(ascending=False)
    news_cat_names = list(news_cat_counts.index)

# OwnVisual_2: Plotting of Top 20 Responses
    df_drop = df.iloc[:, 4:]
    sum_topn_response = df_drop.sum(axis=0).sort_values(ascending=False)[:20]
    sum_topn_response_names = list(sum_topn_response.index)

# create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        # Own Visualisation 1: News genre
        {
            'data': [
                Bar(
                    x=news_cat_names,
                    y=news_cat_counts
                )
            ],

            'layout': {
                'title': 'Categories Distribution in "News" Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }

        },
        # # Own Visualisation 1: Top 20 responses
        {
            'data': [
                Bar(
                    x=sum_topn_response_names,
                    y=sum_topn_response
                )
            ],

            'layout': {
                'title': 'Top 20 Responses',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Responses"
                }
            }
        }
    ]

# encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

# render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()