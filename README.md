# Disaster-Response-Pipeline
Udacity Project with the goal of creating an ETL &amp; Machine Learning pipeline and creating visualisations in a Flask Web App

## Folder Description
There are 3 folders contained in this repository: 
- Data
  * This is where the ETL scripts is stored and can be accessed via 'process_data.py'

- Models
  * The Machine Learning model script is stored here, aswell as the .sav file. 
  * You can run this script by accessing 'train_classifier.py'
  * The udacity_model.sav file is in a zipped folder as the file was too large to upload

- App
 * The script 'run.py' will show you 3 visualisations in the Fask web app, the visualisations are: 
    ~ 'Distribution of Message Genres' (this was provided by Udacity)
    ~ 'Categories Distribution in 'News' genre'
    ~ Top 20 responses
    
    
## Instructions
1. To set up the database and model, run the following commands in the project's root directory.

* To run ETL pipeline that cleans data and stores in database:
  - python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/ETL_DB.db
* To run ML pipeline that trains classifier and saves:
  - python models/train_classifier.py data/ETL_DB.db models/udacity_model.sav

2. Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3001/
