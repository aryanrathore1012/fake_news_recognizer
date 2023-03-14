# fake_news_recognizer

a project that uses machine learning algorithms and natural language processing to predict if a news is fake or not 

### about the dataset

the csv has 2 datasets train, and the_big_one

### Train dataset:-

| # | name | Non-Null Count | Dtype |
| :-: | :-----:| :---: | :----: |
| 0   | id     | 20800 non-null | int64  |
| 1   | title  | 20242 non-null | object |
| 2   | author | 18843 non-null | object |
| 3   | text   | 20761 non-null | object |
| 4   | label  | 20800 non-null | int64  |

1. id: unique id for a news article
2. title: the title of a news article
3. author: author of the news article
4. text: the text of the article; could be incomplete
5. label: a label that marks whether the news article is real or fake (1 - Fake news), (0 - real News)

### The_big_one dataset :-

|#    | Column  | Non-Null Count | Dtype  |
| :---:  | :------:  | :--------------: | :-----:  |
| 0   | label   | 26000 non-null | int64  |
| 1   | content | 26000 non-null | object |

1. label: a label that marks whether the news article is real or fake (1 - Fake news), (0 - real News)

2. content : is the stemmed version of author + title Column in train csv

### the main python program uses the_big_one stemmed content to train while the ipynb uses the train, test, submit data to train and analys the models.

# Roadmap of the project:-

### 1.Preliminary data analysis:
Edit the data to prepare it for further analysis, describe the key features of the data, and summarize the results.

* the program extract the submit labels and joins them with test

### 2.Exploratory data analysis:

Investigate data sets and summarize their main characteristics, often employing data visualization methods

* the dataset in the project is filled with text only so there was no visualization needed.

### 3. Data pre-processing:

The dataset is preprocessed in order to check missing values, noisy data, and other inconsistencies before executing it to the algorithm.

* all three train, test and the_big_one have missing values so they are replaced with "" (empty strings) and labels was typecasted to int64

### 4. Model development & comparison:

Model comparison involves comparing the performance of different models on a given task to identify which model is most effective.

* the models used with their training and training testing scores sorted by testing scores are as follows :-

 ![Screenshot 2023-03-14 191713](https://user-images.githubusercontent.com/91218998/225022815-93613445-703d-4594-9c21-e2156fa0d9cd.png)
 
 ### training and testing score graph:-
 
 ![image](https://user-images.githubusercontent.com/91218998/225023238-44bc9cbb-e252-4ea3-bd65-9d2048013b2c.png)
 
# ---------------------------------------

# inputs and outputs of main_fake_news_predictor.py :-

### i would reccomond you watch the demo_video attached in the files as it would give you a clear image on what the project looks like but here are some Screenshots of the GUI

### input:-

![image](https://user-images.githubusercontent.com/91218998/225027285-f32bd20d-3101-48b7-b2fa-5afbaf8b41b2.png)

### output:-

![image](https://user-images.githubusercontent.com/91218998/225027477-23d12df8-814a-43c0-a5d3-c036439abb99.png)

### msg box at the beggining:-

![image](https://user-images.githubusercontent.com/91218998/225028257-be9de10d-7535-4574-948f-b5ec88145d50.png)

# Limitations:-

* the upgraded model of the project may throw an error sometimes related to number of features. if you know know a fix please let me know

# credits and contact info:-

* made by Aryan Rathore
* LinkedIn : https://www.linkedin.com/in/aryan-rathore-b15459215/
* email: aryanrathore13572002@gmail.com, aryan.rathore2021@vitbhopal.ac.in

# ---------------------------------------
 


