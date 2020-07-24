import pandas_profiling as pp
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns

from matplotlib import pyplot as plt
from textblob import TextBlob

from numpy import nan
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Algorithms
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#cross validator
from sklearn.model_selection import cross_val_score

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

@st.cache(allow_output_mutation=True)
def loadData():
	data = pd.read_csv('movie_dataset.csv')
	return data

#change column order
def changeColumnOrder(column,new_index,data):
    rev = data[column]
    data.drop(labels=[column], axis=1,inplace = True)
    data.insert(new_index, column, rev)

def getCorr(data):
    corrmat = data.corr()
    fig = plt.figure(figsize = (50, 50))
    sns.heatmap(corrmat, vmax = 1, square = True,annot=True)

def round_int(x):
    if x == float("inf") or x == float("-inf"):
        return float('nan') # or x or return whatever makes sense
    return int(round(x))

@st.cache(suppress_st_warning=True)
def preprocessing(data):
    df = data

    #Collection column creation
    data['has_collection'] = data['collection']
    df_collection = data['has_collection']
    df_collection[df_collection.notnull()] = 1
    data['has_collection'].fillna(0, inplace=True)

    changeColumnOrder('revenue',0,data)
    changeColumnOrder('has_collection',1,data)

    #Drop any movie with less than 1 Million in revenue
    data =data[df['revenue'] > 1000000]

    #drop adult column because all are none adult & video because they are false
    data = data.drop(columns=['adult','video','collection','vote_count','vote_average','popularity'])

    #Encoding languages
    le = LabelEncoder()
    data['original_language_enc']= le.fit_transform(data['original_language'])

    #Add encoded languages to dataframe
    original_language_encoded = pd.DataFrame()
    original_language_encoded['original_language_enc'] = data.original_language_enc.unique()
    original_language_encoded['language'] = le.inverse_transform(original_language_encoded['original_language_enc'])

    for x in range(4, 8):
        text = "genre_"+str(x)
        data = data.drop(columns=[text])

    for x in range(1, 4):
        text = "genre_"+str(x)
        data[text].fillna("none", inplace=True)

    for x in range(1, 4):
        text = "genre_"+str(x)
        text_enc = text+"_enc"
        data[text_enc]= le.fit_transform(data[text])

    for x in range(1, 4):
        text = "genre_"+str(x)
        data = data.drop(columns=[text])

    genre_1_encoded = pd.DataFrame()
    genre_1_encoded['genre_1_enc'] = data.genre_1_enc.unique()
    genre_1_encoded['genre'] = le.inverse_transform(genre_1_encoded['genre_1_enc'])

    genre_2_encoded = pd.DataFrame()
    genre_2_encoded['genre_2_enc'] = data.genre_2_enc.unique()
    genre_2_encoded['genre'] = le.inverse_transform(genre_2_encoded['genre_2_enc'])

    genre_3_encoded = pd.DataFrame()
    genre_3_encoded['genre_3_enc'] = data.genre_3_enc.unique()
    genre_3_encoded['genre'] = le.inverse_transform(genre_3_encoded['genre_3_enc'])

    data['has_homepage'] = data['homepage']
    df_homepage = data['has_homepage']
    df_homepage[df_homepage.notnull()] = 1
    data['has_homepage'].fillna(0, inplace=True)

    #drop columns as they have been encoded or are no longer needed
    data = data.drop(columns = ['homepage','original_language','original_title','status'])

    changeColumnOrder('has_homepage',2,data)

    #drop columns
    for x in range(4, 22):
        text = "production_company_"+str(x)
        data = data.drop(columns=[text])

    #replace nan with none
    for x in range(1, 4):
        text = "production_company_"+str(x)
        data[text].fillna("none", inplace=True)

    #encode
    for x in range(1, 4):
        text = "production_company_"+str(x)
        text_enc = text+"_enc"
        data[text_enc]= le.fit_transform(data[text])

    production_company_1_encoded = pd.DataFrame()
    production_company_1_encoded['production_company_1'] = data.production_company_1.unique()
    production_company_1_encoded['production_company_1_enc'] = le.fit_transform(production_company_1_encoded['production_company_1'])

    production_company_2_encoded = pd.DataFrame()
    production_company_2_encoded['production_company_2'] = data.production_company_2.unique()
    production_company_2_encoded['production_company_2_enc'] = le.fit_transform(production_company_2_encoded['production_company_2'])

    production_company_3_encoded = pd.DataFrame()
    production_company_3_encoded['production_company_3'] = data.production_company_3.unique()
    production_company_3_encoded['production_company_3_enc'] = le.fit_transform(production_company_3_encoded['production_company_3'])

    #drop encoded columns
    for x in range(1, 4):
        text = "production_company_"+str(x)
        data = data.drop(columns=[text])

    #drop columns
    for x in range(4, 11):
        text = "spoken_language_"+str(x)
        data = data.drop(columns=[text])

    #replace nan with none
    for x in range(1, 4):
        text = "spoken_language_"+str(x)
        data[text].fillna("none", inplace=True)

    #encode
    for x in range(1, 4):
        text = "spoken_language_"+str(x)
        text_enc = text+"_enc"
        data[text_enc]= le.fit_transform(data[text])

    spoken_language_1_encoded = pd.DataFrame()
    spoken_language_1_encoded['spoken_language_1'] = data.spoken_language_1.unique()
    spoken_language_1_encoded['spoken_language_1_enc'] = le.fit_transform(spoken_language_1_encoded['spoken_language_1'])

    spoken_language_2_encoded = pd.DataFrame()
    spoken_language_2_encoded['spoken_language_2'] = data.spoken_language_2.unique()
    spoken_language_2_encoded['spoken_language_2_enc'] = le.fit_transform(spoken_language_2_encoded['spoken_language_2'])

    spoken_language_3_encoded = pd.DataFrame()
    spoken_language_3_encoded['spoken_language_3'] = data.spoken_language_3.unique()
    spoken_language_3_encoded['spoken_language_3_enc'] = le.fit_transform(spoken_language_3_encoded['spoken_language_3'])

    #drop encoded columns
    for x in range(1, 4):
        text = "spoken_language_"+str(x)
        data = data.drop(columns=[text])

    #drop columns
    for x in range(4, 20):
        text = "production_country_"+str(x)
        data = data.drop(columns=[text])

    #replace nan with none
    for x in range(1, 4):
        text = "production_country_"+str(x)
        data[text].fillna("none", inplace=True)

    #encode
    for x in range(1, 4):
        text = "production_country_"+str(x)
        text_enc = text+"_enc"
        data[text_enc]= le.fit_transform(data[text])

    production_country_1_encoded = pd.DataFrame()
    production_country_1_encoded['production_country_1'] = data.production_country_1.unique()
    production_country_1_encoded['production_country_1_enc'] = le.fit_transform(production_country_1_encoded['production_country_1'])

    production_country_2_encoded = pd.DataFrame()
    production_country_2_encoded['production_country_2'] = data.production_country_2.unique()
    production_country_2_encoded['production_country_2_enc'] = le.fit_transform(production_country_2_encoded['production_country_2'])

    production_country_3_encoded = pd.DataFrame()
    production_country_3_encoded['production_country_3'] = data.production_country_3.unique()
    production_country_3_encoded['production_country_3_enc'] = le.fit_transform(production_country_3_encoded['production_country_3'])

    #drop encoded columns
    for x in range(1, 4):
        text = "production_country_"+str(x)
        data = data.drop(columns=[text])

    #create a new column called profit with revenue - budget
    data['profit'] = nan
    data['percentage_profit'] = nan

    data['profit'] = data['profit'].fillna(0).astype(int)
    data['percentage_profit'] = data['percentage_profit'].fillna(0).astype(int)

    #get rounded off profit
    counter = 0
    while counter < len(data):
        revenue = data['revenue'].iloc[counter]
        budget = data['budget'].iloc[counter]
        profit = revenue - budget
        profit_percent = round_int(profit/budget*100)
        data['profit'].iloc[counter] = profit
        data['percentage_profit'].iloc[counter] = profit_percent
        counter += 1

    changeColumnOrder('profit',0,data)
    changeColumnOrder('percentage_profit',1,data)

    changeColumnOrder('percentage_profit',1,data)
    getCorr(data)

    data = data[data['budget'] > 0]
    data = data[data['profit'] > 0]

    #adding categories
    #bins = [0, 50, 100, 200, 300, np.inf]
    # bins = [-50,-25,0,50,100,150,200,250,300,350,400,450,500,np.inf]
    # names = ['less than -25','-25 to 0','0 - 50','50 - 100','100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450', '450-500','500+']
	# bins = [-50,-25,0,50,100,200,300,400,500,np.inf]

    # bins = [-50,-25,0,50,100,200,300,400,500,np.inf]
    # names = ['less than -25','-25 to 0','0 - 50','50 - 100','100-200', '200-300', '300-400', '400-500','500+']

    # bins = [np.NINF,0,100,400,500,np.inf]
    # names = ['less than 0','0 - 100','100-300','400-500','500+']

    bins = [np.NINF,-100,0,100,np.inf]
    names = ['less than -100','-100 to 0','0 to 100','more than 100']

    # bins = [np.NINF,0,100,300,500,np.inf]
    # names = ['Less than zero','0-100','100-300','300-500','500+']

    data['profit_cat'] = pd.cut(data['percentage_profit'], bins, labels=names)

    data.profit_cat = data.profit_cat.astype(str)
    data['profit_cat_enc']= le.fit_transform(data['profit_cat'])

    changeColumnOrder('profit_cat',0,data)
    changeColumnOrder('profit_cat_enc',0,data)

    profit_buckets = pd.DataFrame()
    profit_buckets['profit_cat'] = data.profit_cat_enc.unique()
    profit_buckets['profit_cat_fixed'] = le.inverse_transform(profit_buckets['profit_cat'])

    data["release_date"] = data["release_date"].astype("datetime64")
    data =data[data['release_date'].dt.year < 2020]
    pd.to_datetime(data["release_date"]).dt.year.value_counts()

    data.profit.groupby(data["release_date"].dt.day).sum().plot(kind="line")
    plt.title('Profit per day of the month')
    plt.xlabel('Day of Month')
    plt.ylabel('Profit')

    # In[25]:
    data.profit.groupby(data["release_date"].dt.month).sum().plot(kind="line")
    plt.title('Profit per month')
    plt.xlabel('Month of the year')
    plt.ylabel('Profit')

    # In[26]:
    data.profit.groupby(data["release_date"].dt.year).sum().plot(kind="line")
    plt.title('Profit per year')
    plt.xlabel('Year')
    plt.ylabel('Profit')

    # In[27]:
    data.revenue.groupby(data["release_date"].dt.year).sum().plot(kind="line")
    plt.title('Revenue per year')
    plt.xlabel('Year')
    plt.ylabel('Revenue')

    data["release_month"] = nan
    data["release_day"] = nan

    for index, x in enumerate(data["release_date"].dt.month):
	    data["release_month"][index] =  x
    for index, x in enumerate(data["release_date"].dt.day):
	    data["release_day"][index] = x

    # In[28]:
    data = data.drop(columns=['release_date'])

    #change NaN to none
    data['tagline'].fillna("none", inplace=True)
    data['overview'].fillna("none", inplace=True)
    data['title'].fillna("none", inplace=True)

    # In[30]:
    data["tagline_polarity"] = nan
    data["tagline_subjectivity"] = nan

    data["overview_polarity"] = nan
    data["overview_subjectivity"] = nan

    data["title_polarity"] = nan
    data["title_subjectivity"] = nan

    data["tagline_polarity"] = data["tagline_polarity"].astype(float)
    data["tagline_subjectivity"] = data["tagline_subjectivity"].astype(float)

    data["overview_polarity"] = data["overview_polarity"].astype(float)
    data["overview_subjectivity"] = data["overview_subjectivity"].astype(float)

    data["title_polarity"] = data["title_polarity"].astype(float)
    data["title_subjectivity"] = data["title_polarity"].astype(float)

    #do sentiment analysis
    for index, x in enumerate(data["tagline"]):
        tagline_text = TextBlob(x)
        polarity = tagline_text.sentiment.polarity
        subjectivity = tagline_text.sentiment.subjectivity
        data["tagline_polarity"][index] =  polarity
        data["tagline_subjectivity"][index] =  subjectivity

    for index, x in enumerate(data["overview"]):
        tagline_text = TextBlob(x)
        polarity = tagline_text.sentiment.polarity
        subjectivity = tagline_text.sentiment.subjectivity
        data["overview_polarity"][index] =  polarity
        data["overview_subjectivity"][index] =  subjectivity

    for index, x in enumerate(data["title"]):
        tagline_text = TextBlob(x)
        polarity = tagline_text.sentiment.polarity
        subjectivity = tagline_text.sentiment.subjectivity
        data["title_polarity"][index] =  polarity
        data["title_subjectivity"][index] =  subjectivity


    data7 = pd.DataFrame()

    headers = ['tagline','tagline_polarity','tagline_subjectivity']

    for x in headers:
        data7[x] = data[x]

    data = data.drop(columns=['profit_cat'])

    getCorr(data)

    data = data.drop(['overview','tagline','title'], axis =1)

    data6 = pd.DataFrame()
    #data6['tagline_polarity','overview_polarity','has_collection','tagline_subjectivity','title_polarity','genre_1_enc','genre_2_enc','production_country_2_enc','genre_3_enc','overview_subjectivity'] = data['tagline_polarity','overview_polarity','has_collection','tagline_subjectivity','title_polarity','genre_1_enc','genre_2_enc','production_country_2_enc','genre_3_enc','overview_subjectivity']

    links = ['profit_cat_enc','tagline_polarity','overview_polarity','has_collection','tagline_subjectivity','title_polarity','genre_1_enc','genre_2_enc','production_country_2_enc','genre_3_enc','overview_subjectivity']

    for x in links:
        data6[x] = data[x]

    corrmat = data6.corr()
    fig = plt.figure(figsize = (50, 50))
    sns.heatmap(corrmat, vmax = 1, square = True,annot=True)
    plt.show()

    #count number of nan values
    data.fillna(data.mean(), inplace=True)

    data = data.drop(columns=['revenue','profit','percentage_profit'])

    #data = data.drop(columns=['tagline_subjectivity','tagline_polarity','overview_subjectivity','overview_polarity','title_subjectivity','title_polarity'])

    # dividing the X and the Y from the dataset
    #data = data.drop(['release_date','popularity','runtime','vote_average'], axis =1)
    X = data.drop(['profit_cat_enc'], axis = 1)
    Y = data["profit_cat_enc"]

    # st.write("scaled_data")
    # st.write(scaled_df.head())

    #scaled_df['profit_cat_enc'] = test['profit_cat_enc']

    # getting just the values for the sake of processing
    # (its a numpy array with no columns)
    xData = X.values
    yData = Y.values
    yData=yData.astype('float64')
    #Y.info()

    for i,x in enumerate(yData):
        yData[i]=x.astype('float')

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(xData, yData, test_size = 0.2, random_state = 42)

    production_companies = [production_company_1_encoded,production_company_2_encoded,production_company_3_encoded]
    spoken_languages = [spoken_language_1_encoded,spoken_language_2_encoded,spoken_language_3_encoded]
    production_countries = [production_country_1_encoded,production_country_2_encoded,production_country_3_encoded]
    genres =[genre_1_encoded,genre_2_encoded,genre_3_encoded]

    #THIS IS WHERE RETURN SHOULD HAPPEN
    return X_train, X_test, Y_train, Y_test,original_language_encoded,production_companies,production_countries,spoken_languages,genres,profit_buckets

#KNN Classifier
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def random_forest(X_train, X_test, Y_train, Y_test):

	#240 seems to be a sweet spot,340
	forest = RandomForestClassifier(n_estimators=340)
	forest.fit(X_train, Y_train)
	#Y_prediction = forest.predict(X_test)
	#score = round(forest.score(X_train, Y_train) * 100, 2)

	scores = cross_val_score(forest, X_train, Y_train, cv=4)
	#st.write("The scores are:")
	#st.write("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	score = round(scores.mean() * 100, 2)


	return score, forest

# Training Decission Tree for Classification
@st.cache(suppress_st_warning=True)
def decisionTree(X_train, X_test, Y_train):
	# Train the model
	tree = DecisionTreeClassifier(max_leaf_nodes=4, random_state=1)
	tree.fit(X_train, Y_train)
	#y_pred = tree.predict(X_test)
	#score = round(tree.score(X_train, Y_train) * 100, 2)
	#score = metrics.accuracy_score(y_test, y_pred) * 100
	#report = classification_report(y_test, y_pred)

	scores = cross_val_score(tree, X_train, Y_train, cv=4)
	#st.write("The scores are:")
	#st.write("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	score = round(scores.mean() * 100, 2)

	return score, tree

#KNN Classifier
@st.cache(suppress_st_warning=True)
def knn_classifier(X_train, X_test, Y_train, Y_test):

	clf = KNeighborsClassifier(n_neighbors=65)
	clf.fit(X_train, Y_train)
	#y_pred = clf.predict(X_test)
	#score = metrics.accuracy_score(Y_test, y_pred) * 100
	#score = round(clf.score(X_train, Y_train) * 100, 2)
	#report = classification_report(Y_test, y_pred)
	scores = cross_val_score(clf, X_train, Y_train, cv=4)
	#st.write("The scores are:")
	#st.write("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	score = round(scores.mean() * 100, 2)
	#st.write(report)

	return score, clf

# Accepting user data for predicting its Member Type
def accept_user_data(original_language_encoded,production_companies,production_countries,spoken_languages,genres):

	output_array = []

	collection_input = st.radio("Is it part of a collection",('Yes', 'No'))

	if collection_input == "Yes":
		collection = 1
	else:
		collection = 0

	homepage_input = st.radio("Does your movie have a website?",('Yes', 'No'))

	if homepage_input == "Yes":
		homepage = 1
	else:
		homepage = 0

	budget = st.slider('What is the movie budget (USD) ?', 1000000, 700000000, 200000000)
	runtime = st.slider('How long will the movie run in minutes?',30,500,60)

	#TODO how are languages encoded, genre,production company,spoken language, production country?
	#ORIGINAL LANGUAGE
	original_language_dict = original_language_encoded.set_index('language').T.to_dict('list')
	language = list(original_language_dict.keys())

	original_language_input = st.selectbox("Select the language: ", language)
	original_language_value = original_language_dict[original_language_input][0]

	output_array.extend([collection,homepage,budget,runtime,original_language_value])

	#GENRE
	st.subheader('Genres')

	genres_dict = genres[0].set_index('genre').T.to_dict('list')
	genres_1 = list(genres_dict.keys())
	genres_1_input = st.selectbox("Select the first production: ", genres_1)
	genres_1_value = genres_dict[genres_1_input][0]

	genres_dict = genres[1].set_index('genre').T.to_dict('list')
	genres_2 = list(genres_dict.keys())
	genres_2_input = st.selectbox("Select the second production: ", genres_2)
	genres_2_value = genres_dict[genres_2_input][0]

	genres_dict = genres[2].set_index('genre').T.to_dict('list')
	genres_3 = list(genres_dict.keys())
	genres_3_input = st.selectbox("Select the third production: ", genres_3)
	genres_3_value = genres_dict[genres_3_input][0]

	output_array.extend([genres_1_value,genres_2_value,genres_3_value])


	#PRODUCTION COMPANIES
	st.subheader('Production companies')

	production_companies_dict = production_companies[0].set_index('production_company_1').T.to_dict('list')
	production_company_1 = list(production_companies_dict.keys())
	production_company_1_input = st.selectbox("Select the first production: ", production_company_1)
	production_company_1_value = production_companies_dict[production_company_1_input][0]

	#production_company_2_enc
	production_companies_dict = production_companies[1].set_index('production_company_2').T.to_dict('list')
	production_company_2 = list(production_companies_dict.keys())
	production_company_2_input = st.selectbox("Select the second production: ", production_company_2)
	production_company_2_value = production_companies_dict[production_company_2_input][0]

	#production_company_3_enc
	production_companies_dict = production_companies[2].set_index('production_company_3').T.to_dict('list')
	production_company_3 = list(production_companies_dict.keys())
	production_company_3_input = st.selectbox("Select the third production: ", production_company_3)
	production_company_3_value = production_companies_dict[production_company_3_input][0]

	output_array.extend([production_company_1_value,production_company_2_value,production_company_3_value])

	#SPOKEN LANGUAGE
	st.subheader('Spoken Languages')

	spoken_language_dict = spoken_languages[0].set_index('spoken_language_1').T.to_dict('list')
	spoken_language_1 = list(spoken_language_dict.keys())
	spoken_language_1_input = st.selectbox("Select the first spoken language: ", spoken_language_1)
	spoken_language_1_value = spoken_language_dict[spoken_language_1_input][0]

	spoken_language_dict = spoken_languages[1].set_index('spoken_language_2').T.to_dict('list')
	spoken_language_2 = list(spoken_language_dict.keys())
	spoken_language_2_input = st.selectbox("Select the second spoken language: ", spoken_language_2)
	spoken_language_2_value = spoken_language_dict[spoken_language_1_input][0]

	spoken_language_dict = spoken_languages[2].set_index('spoken_language_3').T.to_dict('list')
	spoken_language_3 = list(spoken_language_dict.keys())
	spoken_language_3_input = st.selectbox("Select the third spoken language: ", spoken_language_3)
	spoken_language_3_value = spoken_language_dict[spoken_language_1_input][0]

	output_array.extend([spoken_language_1_value,spoken_language_2_value,spoken_language_3_value])

	#PRODUCTION COUNTRY
	st.subheader('Production country')

	production_countries_dict = production_countries[0].set_index('production_country_1').T.to_dict('list')
	production_country_1 = list(production_countries_dict.keys())
	production_country_1_input = st.selectbox("Select the first production country: ", production_country_1)
	production_country_1_value = production_countries_dict[production_country_1_input][0]

	production_countries_dict = production_countries[1].set_index('production_country_2').T.to_dict('list')
	production_country_2 = list(production_countries_dict.keys())
	production_country_2_input = st.selectbox("Select the second production country: ", production_country_2)
	production_country_2_value = production_countries_dict[production_country_2_input][0]

	production_countries_dict = production_countries[2].set_index('production_country_3').T.to_dict('list')
	production_country_3 = list(production_countries_dict.keys())
	production_country_3_input = st.selectbox("Select the third production country: ", production_country_3)
	production_country_3_value = production_countries_dict[production_country_3_input][0]

	output_array.extend([production_country_1_value,production_country_2_value,production_country_3_value])


	#PRODUCTION COUNTRY
	st.subheader('Final details')

	st.write("**FYI:**")
	st.markdown("_The **subjectivity** of a phrase is a float within the range 0 and 1 where 0 is very objective and 1 is very subjective_")
	st.markdown("_**Polarity** is float which lies in the range of -1 and 1 where 1 means positive statement and -1 means a negative statement_")

	tagline = st.text_input('Movie tagline', '')
	tagline_polarity, tagline_subjectivity = run_sentiment_analysis(tagline)
	st.write('Polarity: ',tagline_polarity)
	st.write('subjectivity: ',tagline_subjectivity)
	output_array.extend([tagline_polarity, tagline_subjectivity])

	overview = st.text_area('Movie Overview', '')
	overview_polarity, overview_subjectivity = run_sentiment_analysis(overview)
	st.write('Polarity: ',overview_polarity)
	st.write('subjectivity: ',overview_subjectivity)
	output_array.extend([overview_polarity,overview_subjectivity])

	title = st.text_input('Movie title', '')
	title_polarity, title_subjectivity = run_sentiment_analysis(title)
	st.write('Polarity: ',title_polarity)
	st.write('subjectivity: ',title_subjectivity)
	output_array.extend([title_polarity,title_subjectivity])

	day = st.slider('Which day of the month will the movie be released?', 0, 30, 15)
	month = st.slider('Which month will the movie be released',0,11,5)

	output_array.extend([day,month])

	user_prediction_data = np.array(output_array).reshape(1,-1)

	return user_prediction_data

def run_sentiment_analysis(txt):
	text_blob = TextBlob(txt)
	polarity = text_blob.sentiment.polarity
	subjectivity = text_blob.sentiment.subjectivity
	return polarity,subjectivity

def use_user_prediction(pred,prob,profit_buckets):
	result = profit_buckets.loc[profit_buckets['Class'] == pred[0]]['Profit Range']

	st.write("The movie is expected to make: ", result.iloc[0], "%")

	result_class = profit_buckets.loc[profit_buckets['Class'] == pred[0]]['Profit Range']

	st.write("The probabilities of each class are:")
	prob_df = pd.DataFrame(prob)
	sorted_buckets = profit_buckets.sort_values('Profit Range')
	prob_df.columns = sorted_buckets['Profit Range']
	st.table(prob)
	st.bar_chart(prob_df.transpose())


def main():
	st.title("Predicting the success of a Hollywood movie")

	st.write("How much profit will your movie make based?")
	st.write("Depending how you answer the next questions, this model will determine how much profit your movie is expected to make compared to your initial budget.")
	data = loadData()
	X_train, X_test, Y_train, Y_test,original_language_encoded,production_companies,production_countries,spoken_languages,genres,profit_buckets = preprocessing(data)

	# Insert Check-Box to show the snippet of the data.
	if st.checkbox('Show Raw Data'):
		st.subheader("Raw data")
		st.write(data.head())


	# ML Section
	choose_model = st.sidebar.selectbox("Choose the ML Model",
		["Decision Tree","Random Forest", "K-Nearest Neighbours"])

	st.sidebar.subheader("ABOUT")
	st.sidebar.markdown("This is a classification exercise. It shall classify movies according to the expected groups for profit.")
	st.sidebar.markdown("Profit is calculated as:")
	st.sidebar.markdown("**(REVENUE - BUDGET / BUDGET) x 100**")

	profit_buckets.columns = ['Class', 'Profit Range']

	profit_buckets.set_index('Class')

	st.sidebar.table(profit_buckets.sort_values('Class'))

	if(choose_model == "Decision Tree"):
		
		score, tree = decisionTree(X_train, X_test, Y_train)
		st.text("Accuracy of Decision Tree model is: ")
		st.write(score,"%")
		# st.text("Report of Decision Tree model is: ")
		# st.write(report)
		if(st.checkbox("Predict your own",value=False)):

			user_prediction_data = accept_user_data(original_language_encoded,production_companies,production_countries,spoken_languages,genres)

			if st.button('Predict'):

				pred = tree.predict(user_prediction_data)
				prob = tree.predict_proba(user_prediction_data)

				use_user_prediction(pred,prob,profit_buckets)

	elif(choose_model == "Random Forest"):
		st.subheader("Using Random Forest")
		score, forest = random_forest(X_train, X_test, Y_train, Y_test)
		st.text("Accuracy of Random Forest model is: ")
		st.write(score,"%")

		if(st.checkbox("Predict your own",value=True)):

			user_prediction_data = accept_user_data(original_language_encoded,production_companies,production_countries,spoken_languages,genres)

			if st.button('Predict!'):


				pred = forest.predict(user_prediction_data)
				prob = forest.predict_proba(user_prediction_data)

				use_user_prediction(pred,prob,profit_buckets)

	elif(choose_model == "K-Nearest Neighbours"):
		st.subheader("Using KNN")
		score, clf = knn_classifier(X_train, X_test, Y_train, Y_test)
		st.text("Accuracy of K-Nearest Neighbour model is: ")
		st.write(score,"%")

		if(st.checkbox("Predict your own",value=True)):

			user_prediction_data = accept_user_data(original_language_encoded,production_companies,production_countries,spoken_languages,genres)

			if st.button('Predict!'):


				pred = clf.predict(user_prediction_data)
				prob = clf.predict_proba(user_prediction_data)

				use_user_prediction(pred,prob,profit_buckets)

if __name__ == "__main__":
	main()
