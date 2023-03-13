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
from sklearn.model_selection import train_test_split
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
            1. reads the the_big_one handles nan values and splits the data
            2. shows the waiting msg to the user and trains the model getting the score
            3. makes the main_window with three frames that asks for the title and author
            4. checks if the information sent is valid  or not
            5. if the information is valid it will stem the data and transform it using stemming and tf-idf vectorizer

        '''

        # -------------------------------------------------------------------------------------------------------------
        # 0. initializes all the path variables

        self.the_big_one_path = "F://aryans_code_notes//machine_learning//fake_news_predictor//the_big_one.csv"
        self.tile_image_path = "F://aryans_code_notes//machine_learning//fake_news_predictor//tile_image.ico"
        self.icon_path = "F://aryans_code_notes//machine_learning//fake_news_predictor//icon.ico"

        # -------------------------------------------------------------------------------------------------------------
        # 1. reads the the_big_one handles nan values and splits the data

        train_data = pd.read_csv(self.the_big_one_path)
        train_data.fillna(" ", inplace=True)
        train_data.drop(columns="Unnamed: 0", inplace=True)


        X_ = train_data["content"]

        tf_idf_vectorizer = TfidfVectorizer()

        tf_idf_vectorizer.fit(X_)

        X_ = tf_idf_vectorizer.transform(X_)

        Y_ = train_data["label"]

        X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=0.2)


        # -------------------------------------------------------------------------------------------------------------
        # 2. shows the waiting msg to the user and trains the model getting the score

        msg_root = Tk()
        msg_root.withdraw()
        tmsg.showinfo("Fitting model", "The model is being fitted please wait, the program takes approx 1 min....\n\nCLOSE THIS MSG TO START TRAINING THE MODEL.")
        msg_root.destroy()

        self.random_forest_model = RandomForestClassifier(n_jobs=2, random_state=1).fit(X_train,Y_train)
        self.train_score = self.random_forest_model.score(X_train,Y_train)
        self.test_score = self.random_forest_model.score(X_test, Y_test)
        # self.random_forest_model.fit(X_test, Y_test)
        
        # -------------------------------------------------------------------------------------------------------------
        # 3. makes the main_window with three frames that asks for the title and author

        root = Tk() 
        root.geometry("1000x700")
        root.wm_iconbitmap(self.icon_path)
        root.title("Fake_News_Recognizer By Aryan Rathore")
        root.config(bg = "#3C2A21")

        top_frame = Frame(root, bg = "#1A120B")
        top_frame.pack(fill=X)

        form_frame = Frame(root, bg="#3C2A21")
        form_frame.pack(side=TOP,fill=BOTH)

        bottom_frame = Frame(root, bg="#3C2A21")
        bottom_frame.pack(side=BOTTOM)    

        tile_image = Image.open(self.tile_image_path).resize((150,150), Image.ANTIALIAS)
        tile_image = ImageTk.PhotoImage(tile_image)

        photo_label = Label(top_frame,image=tile_image,borderwidth=10,relief=RIDGE,anchor=CENTER)
        photo_label.image = tile_image
        photo_label.pack(side=TOP,pady=(25,0))

        Label(top_frame,text="welcome to Fake_News_Recognizer",font=("Cascadia Mono SemiBold", 27, "bold italic"),anchor=CENTER,bg='#D5CEA3',fg='black',borderwidth=5,relief=RIDGE).pack(side=TOP,pady=(25,0),fill=X)

        # change these:
        # self.train_score = 0.991231254124123
        # self.test_score = 0.9111011231234123
        Label(top_frame,text=f"Enter the title and author of the news\nthe model's train score: {self.train_score:.2f}, test score: {self.test_score:.2f}",font=("Cascadia Mono SemiBold", 20, "bold italic"),fg='black',bg="#D5CEA3",anchor=CENTER).pack(side=TOP,pady=(10,25),fill=X)

        author_var = StringVar()
        title_var = StringVar()

        Label(form_frame,text="name of the author",font=("Cascadia Mono SemiBold",20, "bold italic "),fg='white',bg="#3C2A21").grid(row=0,column=0,padx=(200,10),pady=(75,25),sticky=W)
        Entry(form_frame,textvariable=author_var,font=("Cascadia Mono SemiBold",15, "bold italic ")).grid(row=0,column=1,padx=(25,25),pady=(75,25),sticky=W)

        Label(form_frame,text="title of the article",font=("Cascadia Mono SemiBold",20, "bold italic "),fg='white',bg="#3C2A21").grid(row=1,column=0,padx=(200,10),pady=(15,50),sticky=W)
        Entry(form_frame,textvariable=title_var,font=("Cascadia Mono SemiBold",15, "bold italic ")).grid(row=1,column=1,padx=(25,25),pady=(15,50),sticky=W)


        # --------------------------------------------------------------------------------------------------------
        # 4. checks if the information sent is valid  or not

        def check_title_author():
            
            if author_var.get() == "":
                tmsg.showerror("Invalid information","Please enter the author's name. ")
            elif title_var.get() == "":
                tmsg.showerror("Invalid information","Please enter title of the article. ")
            else:
                

                # --------------------------------------------------------------------------------------------------------
                # 5. if the information is valid it will stem the data and transform it using stemming and tf-idf vectorizer

                port_stem = PorterStemmer()
                sentance = author_var.get() + " " + title_var.get()
                stemmed_sentance = re.sub("[^a-zA-Z]", " ", sentance).lower().split()
                stemmed_sentance = [port_stem.stem(word) for word in stemmed_sentance if not word in stopwords.words('english')]
                stemmed_sentance = " ".join(stemmed_sentance)
                
                tf_vec = TfidfVectorizer()
                
                tf_vec.fit([stemmed_sentance])

                transformed_sentance = tf_vec.transform([stemmed_sentance])

                ans = self.random_forest_model.predict(transformed_sentance)

                print(ans)

                # if int(ans[0]) == 0:
                #     tmsg.showinfo("Prediction successfull", "The news is Real")
                # else:
                #     tmsg.showinfo("Prediction successfull", "The news is Fake")


        Button(bottom_frame, text="submit", bg='#D5CEA3', fg='black', borderwidth=5,relief=RAISED,font=("Cascadia Mono SemiBold",15, "bold italic "),command=check_title_author).pack(side=BOTTOM,pady=(0,10))
    
        root.mainloop()

s = fake_news_predictor()




