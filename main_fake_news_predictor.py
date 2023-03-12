########################################################### CREDITS ##############################################################################################

'''
                                                  WELCOME TO THE fake_news_predictor

                            THIS PROGRAM DETECTS FAKE NEWS BY TAKING AUTHOR AND TITLE OF THE NEWS ARTICLE

                                                    MADE BY : ARYAN RATHORE 
                                            COMPUTER SCIENCE ENGINEER AT VIT BHOPAL

                                                        CONTACT INFO
                                                aryanrathore13572002@gmail.com
                                               aryan.rathore2021@vitbhopal.ac.in
                                          github :- https://github.com/aryanrathore1012
                                    LINKEDIN - https://www.linkedin.com/in/aryan-rathore-b15459215/
'''

########################################################### IMPORTS ##############################################################################################

import numpy as np
import pandas as pd
import datetime as dt
from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox as tmsg
from tkinter import ttk
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

########################################################### FUNCTIONS ###################################################################0#########################

'''                                                             IMPORTANT NOTE                                                              
                                  BEFORE YOU RUN THE PROGRAM MAKE SURE YOU READ AND FOLLOW THE LINES BELOW  
                                                       OTHERWISE THE PROGRAM WONT RUN                                                      '''

# 1

'''          THERE ARE IN TOTAL OF 4 FUNCTIONS THAT CALL EACH OTHER, "EVERY FUNCTION HAS A DOCSTRING LIKE THIS SPECIFIES WHAT THAT FUNCTION DOES
                                                        AND HOW THE THE FUNCTION WORKS"                                                      '''
# 2

'''      MAKE SURE YOU READ THE "dataset_analysis_and_model_selection.ipynb" BEFORE USING THIS AS IT HAS THE INFO ON THE DATA
                                            AND WHY USED RANDOM FOREST CLASSIFIER MODEL FOR MY PREDICTIONS                                         '''

# 3

'''   I HAVE TO SPECIFY A FILE PATH TO read the data from (TEST.CSV, TRAIN.csv and all the images and icons) FILES IF 
    YOU ARE USING OR COPY PASTING MY CODE MAKE YOU CHANGE THE FILE PATHS I HAVE SPECIFIED WHICH FUNCTIONS NEED A 'FILE PATH CHANGE' 
                                                    SO MAKE SURE YOU CHANGE THEM FIRST '''

# 4

'''
                  ||||||| JUST CHANGE THE FILE PATHS FROM LINE 75 TO 78 AND YOU CAN RUN THE PROGRAM FOR YOURSELF |||||||
'''

class fake_news_predictor:

    def __init__(self): # file path change in this function
        
        '''
            this function initializes all the data needed for the program like:

            0. initializes all the path variables
            1. reads the train_csv and trains the random forest model
            2. reads the test_csv and checks the F1 score of the model
            3. feeds the data of the test_csv to the model 
            4. initializes all the object variables

        '''

        # -------------------------------------------------------------------------------------------------------------
        # 0. initializes all the path variables

        self.train_data_path = "F://aryans_code_notes//machine_learning//fake_news_predictor//train.csv"
        self.test_data_path = "F://aryans_code_notes//machine_learning//fake_news_predictor//test.csv"
        self.tile_image_path = "F://aryans_code_notes//machine_learning//fake_news_predictor//tile_image.ico"
        self.icon_path = "F://aryans_code_notes//machine_learning//fake_news_predictor//icon.ico"

        # -------------------------------------------------------------------------------------------------------------
        # 1. reads the train_csv and trains the random forest model

        train_data = pd.read_csv(self.train_data_path)
        test_data = pd.read_csv(self.test_data_path)

        # 1. removing the text, unnamed: 0, columns from the csvs 
        # 2. joining the author and title to form content column 
        # 3. dropping the author and title columns as we dont need them anymore

        train_data = train_data.fillna("")
        train_data = train_data.drop(columns=["id", "text"])
        train_data["content"] = train_data["author"] + " " + train_data["title"]
        train_data = train_data.drop(columns=["title", "author"])

        # print(train_data)  ---> [20800 rows x 2 columns]
        
        test_data = test_data.fillna("")
        test_data = test_data.drop(columns=["id", "text", "Unnamed: 0"])
        test_data["content"] = test_data["author"] + " " + test_data["title"]
        test_data = test_data.drop(columns=["title", "author"])
        
        # print(test_data) ---> [5200 rows x 2 columns]
        
        # our data is in text form so we will have to convert it to vectors so our computer can understand it

        port_stem = PorterStemmer()

        def stemming(sentance):
            stemmed_sentance = re.sub("[^a-zA-Z]", " ", sentance).lower().split()
            stemmed_sentance = [port_stem.stem(word) for word in stemmed_sentance if not word in stopwords.words('english')]
            stemmed_sentance = " ".join(stemmed_sentance)
            return stemmed_sentance

        # print(train_data["content"])
        # print()
        # print(test_data["content"])

        print("stemming_start")
        train_data["content"] = train_data["content"].apply(stemming)
        test_data["content"] = test_data["content"].apply(stemming)
        print("stemming_done")
        

        x = train_data["content"].values

        tf_idf_vectorizer = TfidfVectorizer()

        tf_idf_vectorizer.fit(x)

        x = tf_idf_vectorizer.transform(x)

        # X_train = tf_idf_vectorizer.transform(tf_idf_vectorizer.fit(train_data["content"]))
        # X_test = tf_idf_vectorizer.transform(tf_idf_vectorizer.fit(test_data["content"]))

        Y_train = train_data["label"]
        # Y_test = test_data["label"]

        print("training the random forest model")

        self.random_forest_model = RandomForestClassifier(n_jobs=2, random_state=1).fit(x, Y_train)
        self.train_score = self.random_forest_model.score(x, Y_train)
        print(self.train_score)


s = fake_news_predictor()




