"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import seaborn as sns
import numpy as np


## nlpk import 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import TreebankWordTokenizer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
## other imports
from PIL import Image
import re
import string
import plotly.figure_factory as ff
import plotly.graph_objects as go
import base64

#audio imports
#from gtts import gTTs
import os
import altair as alt
from streamlit_folium import folium_static
import folium
from branca.colormap import linear, LinearColormap


import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
_lock = RendererAgg.lock


#suppress cell_warnings
import warnings
warnings.filterwarnings("ignore")


# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load data 
train_df = pd.read_csv("resources/train.csv")

# defining a data Processing and cleaning  function
def tweet_processor(user_input):
	
	if isinstance(user_input, str):
		x_val= user_input.lower()
		x_val= ''.join([x for x in x_val if x not in string.punctuation])

		x_val =[x_val]



	if isinstance(user_input, pd.DataFrame):
		x_val = user_input['message'].astype(str)
		
	

	return x_val

# 
# Loading prediction model using its path
def load_model(path_):
	model_ = joblib.load(open(os.path.join(path_),"rb"))
	return model_


# Prediction classification function
def predict_class(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key
# The main function where we will build the actual app
def main():

	
	"""Tweet Classifier App with Streamlit """

    ### Loading Company logo
	row1_space1, center_, row1_space2 = st.beta_columns((.5, 1, .2, ))
	with center_,_lock :

		file_ = open('resources/imgs/Company_logo.gif', "rb")
		contents = file_.read()
		data_url = base64.b64encode(contents).decode("utf-8")
		file_.close()
		st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True,)	



	pages = ["Prediction Page","Sentiment Visualization","Climate Change", "Company Information, Background & Team"]
	selection = st.sidebar.selectbox("Choose Page :", pages)

	#Building the "Climate Change" page

	if selection == "Climate Change":
		st.header("Climate Changes Between 1980-2020")

		st.info("""This page contains maps of climate change sentiment worldwide.""")
	
		st.write("""
		Climate change has had highly variable effects in different places.
		This dashboard lets you see the climate impacts so far. For each city, you can see changes in:

		1. Daily high temperatures 
		2. Daily low temperatures
		3. Total precipitation

		Use the map for quick comparisons. Then use the menu at the bottom to drill into single city data.
		Results are aggregations of [this raw daily weather data](https://docs.opendata.aws/noaa-ghcn-pds/readme.html). This page doesn't show extreme weather events. Changes in extreme weather are more important, but that topic deserves a more detailed review than this page.
		""")

		hide_menu_style = """
				<style>
				#MainMenu {visibility: hidden;}
				</style>
				"""
		st.markdown(hide_menu_style, unsafe_allow_html=True)

		@st.cache
		def get_annual_stats():
			raw_data = pd.read_parquet( r'weather_1980_to_2020.parquet')
			raw_data['date'] = pd.to_datetime(raw_data.date)
			raw_data['year'] = raw_data.date.dt.year
			annual_stats = raw_data.fillna(0).groupby(['station_id', 'year']).agg(
									{'max_temp_c': ['mean'],
									'min_temp_c': ['mean'],
									'precip_mm': ['sum']}
			).reset_index()
			annual_stats.columns = ['station_id', 'year', 'max_temp_c', 'min_temp_c', 'precip_mm']
			return annual_stats

		@st.cache
		def read_base_file():
			station_stats = pd.read_csv(r'station_stats.csv')
			station_stats['pct_precip_change'] = (100 * station_stats.slope_total_precip_pct.round(3))
			station_stats['slope_max_temp'] = station_stats['slope_max_temp'].round(2)
			station_stats['slope_min_temp'] = station_stats['slope_min_temp'].round(2)
			return station_stats


		metric_descs = {'slope_max_temp': 'Annual change (degrees Celsius) in average daily high temp',
						'slope_min_temp': 'Annual change (degrees Celsius) in average daily low temp',
						'pct_precip_change': 'Annual percent change in precipitation',}

		metric_units = {'slope_max_temp': '° / year',
						'slope_min_temp': '° / year',
						'pct_precip_change': '% / year',}

		name_in_annual_data = {'slope_max_temp': 'max_temp_c',
							'slope_min_temp': 'min_temp_c',
							'pct_precip_change': 'precip_mm',}

		graph_descriptions = {'max_temp_c': 'Average daily high temperature (Celsius)',
							'min_temp_c': 'Average daily low temperature (Celsius)',
							'precip_mm': 'Total Precipitation (millimeters)', 
		}

		reverse_colormap = {'slope_max_temp': True,
							'slope_min_temp': True,
							'pct_precip_change': False}



		@st.cache(allow_output_mutation=True)
		def make_city_graphs(allow_output_mutation=True):
			# Folium converts vegalite scatters to line graphs in an ugly way. So, make a graph
			# that looks good after the conversion, even if it looks different from standalone
			def make_one_city_map_graphs(annual_this_city, city_name, annual_data_field):
				graph = alt.Chart(annual_this_city, title=city_name).mark_line().encode(
					alt.X('year', scale=alt.Scale(zero=False), axis=alt.Axis(format="d")),
					alt.Y(annual_data_field, scale=alt.Scale(zero=False)),
				)
				return graph

			def make_one_city_standalone_graphs(annual_this_city, city_name, annual_data_field):
				graph = alt.Chart(annual_this_city, title=city_name).mark_point().encode(
					alt.X('year', scale=alt.Scale(zero=False), axis=alt.Axis(format="d")),
					alt.Y(annual_data_field, scale=alt.Scale(zero=False), title=''),
				)
				return graph + graph.transform_regression('year', annual_data_field).mark_line()

			out = {'for_map': {},
				'standalone': {}}
			for _, city in station_stats.iterrows():
				annual_this_city = annual_stats.loc[annual_stats.station_id == city.station_id]
				city_name = city.municipality
				station_id = city.station_id
				out['for_map'][station_id] = {summary_stat: make_one_city_map_graphs(annual_this_city, city_name, annual_stat) 
								for summary_stat, annual_stat in name_in_annual_data.items()}
				out['standalone'][city_name] = {annual_stat: make_one_city_standalone_graphs(annual_this_city, city_name, annual_stat) 
								for summary_stat, annual_stat in name_in_annual_data.items()}
			return out

			
		@st.cache(hash_funcs={folium.folium.Map: lambda _: None}, allow_output_mutation=True)
		def make_map(field_to_color_by):
			main_map = folium.Map(location=(39, -77), zoom_start=1)
			colormap = linear.RdYlBu_08.scale(station_stats[field_to_color_by].quantile(0.05),
											station_stats[field_to_color_by].quantile(0.95))
			if reverse_colormap[field_to_color_by]:
				colormap = LinearColormap(colors=list(reversed(colormap.colors)),
										vmin=colormap.vmin,
										vmax=colormap.vmax)
			colormap.add_to(main_map)
			metric_desc = metric_descs[field_to_color_by]
			metric_unit = metric_units[field_to_color_by]
			colormap.caption = metric_desc
			colormap.add_to(main_map)
			for _, city in station_stats.iterrows():
				icon_color = colormap(city[field_to_color_by])
				city_graph = city_graphs['for_map'][city.station_id][field_to_color_by]
				folium.CircleMarker(location=[city.lat, city.lon],
							tooltip=f"{city.municipality}\n  value: {city[field_to_color_by]}{metric_unit}",
							fill=True,
							fill_color=icon_color,
							color=None,
							fill_opacity=0.7,
							radius=5,
							popup = folium.Popup().add_child(
													folium.features.VegaLite(city_graph)
													)
							).add_to(main_map)
			return main_map

		annual_stats = get_annual_stats()
		station_stats = read_base_file()
		city_graphs = make_city_graphs()

		#st.header('Map')
		#st.write("""
		#You can zoom into the map, or get a city's history by clicking on it.
		#""")
		#metric_for_map = st.selectbox('Climate metric for map',
		#							options=list(metric_descs.keys()),
		#							index=0,
		#							format_func=lambda m: metric_descs[m])
		#main_map = make_map(metric_for_map)

		#folium_static(main_map)

		#st.write("""

		#*Details:*

		#1. Cities are color-coded based on coefficients of a linear regression model. 
		#2. Cities are included iff they have data available for 99% of days since 1980. 
		#""")
		st.markdown("""---""")
		#st.header('Single Location Drilldown')
		region = st.selectbox("Region", sorted(station_stats.region.unique()), index=0)
		city_name = st.selectbox("City", sorted(station_stats.loc[station_stats.region == region].municipality.unique()))
		for graph_name, graph in city_graphs['standalone'][city_name].items():
			st.write(graph_descriptions[graph_name])
			st.write(graph)

	# Building out the "Company Information, Background & Team" page

	if selection == "Company Information, Background & Team":
		st.title("Company Information, Background and Team")
		st.info('Discover the mission and vision that keeps us going as well as the amazing team that pulled this project together and how we started.')

		st.header('Our Mission')		
		st.write('To use AI to combat climate change within Africa, securing the futures of the generations of now and tomorrow.')

		st.header('Our Vision')
		st.write('A better and more intelligent Africa which is able to adapt to the fourth industrial revolution by using Data Science, for social good.')

		st.header('Our Amazing Team')
		st.write('A team of 6 passionate AI solutionists.')
		#First row of pictures

		col1, col2,col3 = st.beta_columns(3)
		Ric_Pic =Image.open('resources/imgs/Rickie_pic.png') 
		col1.image(Ric_Pic,caption="Rickie Mogale Mohale", width=150)
		col1.write('Tech-lead and software developer.')
        
		Cot_Pic =Image.open('resources/imgs/courtney_pic.png') 
		col2.image(Cot_Pic,caption="Courtney Murugan", width=150)
		col2.write('Machine learning engineer')

		Cot_Pic =Image.open('resources/imgs/jacques_pic.png') 
		col3.image(Cot_Pic,caption="Jacques Stander", width=150)
		col3.write('Project manager')

        #Second row of pictures
		col4, col5,col6 = st.beta_columns(3)
		vesh_Pic =Image.open('resources/imgs/veshen_pic.png') 
		col4.image(vesh_Pic,caption="Veshen Naidoo", width=150)
		col4.write('UX/UI Designer')
        
		Phiw_Pic =Image.open('resources/imgs/phiwe_pic.png') 
		col5.image(Phiw_Pic,caption="Phiweka Mthini", width=150)
		col5.write('Digital marketer ')

		nor_Pic =Image.open('resources/imgs/nour_pic.png') 
		col6.image(nor_Pic,caption="Nourhan Alfalous", width=150)
		col6.write('Database architect')

		#Third row of picture 
		col7, col8,col9 = st.beta_columns(3)

		st.header('How we started?')
		st.write('African Intelligence started as a group of 6 students who met each other on a university project. The students bonded together around a love for solving problems with the help of AI. ')	
		st.write('These students all graduated with flying colours and entered successful carreers, but they never forgot the joys of solving real world problems.')
		st.write('A few years later they decided to meet up again and started working part time on this project which they call: AI Africa.')
	

	# Building out the predication page
	if selection == "Prediction Page":
		
		row1_space1, center_, row1_space2 = st.beta_columns((.3, 1, .2, ))
		st.title('Climate Change Sentiment Tracker')
		
		st.info('This page uses machine learning models to predict an entity or an individual\'s sentiment on climate change based on the tweet that they input.')
		
		row1_space1, center_, row1_space2 = st.beta_columns((.1, 1, .1, ))
		st.subheader('To make predictions, please follow the three steps below:')
	
		
		#selecting input text
		text_type_selection = ['Single tweet input','Dataframe input'] 
		text_selection = st.selectbox('Step 1 ) : Select type of tweet input', text_type_selection)


		

		Models = ["Logistic Regression","Linear SVC","Naive Bayes multinomial","Ridge classifier"]
		selected_model = st.radio("Step 2 ) : Choose prediction model ",Models)
		# User selecting prediction model
		#Models = ["Logistic regression","Decision tree","Random Forest Classifier","Naive Bayes","XGboost","Linear SVC"]
		#selected_model =st.selectbox("Step 3 ) : Choose prediction model ",Models )
		
		if text_selection == 'Single tweet input':
			st.warning('To make  accurate prediction\'s your tweet should  have at least 5 words')
			user_input = st.text_area("Step 3 ) : Enter Your Single Text Below :") 
            ### SINGLE TWEET CLASSIFICATION ###
			
            # Creating a text box for user input

			prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
			if st.button("Classify"):
				## showing the user original text
				#st.text("Input tweet is :\n{}".format(user_input))

				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([user_input]).toarray()
				

            	#M Model_ Selection
				if selected_model == "Logistic Regression":

					predictor = load_model("resources/logreg_count.pkl")
					X_input =tweet_processor(user_input)
					prediction = predictor.predict(X_input)
               	    # st.write(prediction)
				elif selected_model == "Linear SVC":

					predictor = load_model("resources/Lsvc_tfidf.pkl")
					X_input =tweet_processor(user_input)
					prediction = predictor.predict(X_input)
                    # st.write(prediction)
				elif selected_model == "Naive Bayes multinomial":
					predictor = load_model("resources/nbm_count.pkl")
					X_input =tweet_processor(user_input)
					prediction = predictor.predict(X_input)
                    # st.write(prediction)
				elif selected_model == "Ridge classifier":
					predictor = load_model("resources/ridge_count.pkl")
					X_input =tweet_processor(user_input)
					prediction = predictor.predict(X_input)

				# st.write(prediction)
			    # When model has successfully run, will print prediction
			    # You can use a dictionary or similar structure to make this output
			    # more human interpretable.
			    # st.write(prediction)
				final_result = predict_class(prediction,prediction_labels)
				st.success("Input tweet is :{}".format(user_input))
				st.success("Tweet Categorized as : {}".format(final_result))

				#Audio code

				#language = 'en'
				#myobj = gTTS(text=mytext, lang=language, slow=False)
				#myobj.save("welcome.mp3")
				#os.system("mpg321 welcome.mp3")


				#text_en = "this article is non compliant"
				#ta_tts = gTTS(text_en)
				#ta_tts.save("trans.mp3")
				#audio_file = open("trans.mp3", "rb")
				#audio_bytes = audio_file.read()
				#st.audio(audio_bytes, format="audio/ogg",start_time=0)

				#st.audio(audio_bytes, format="audio/ogg")

		if text_selection == 'Dataframe input':

			csv_file =st.file_uploader("Step 3 ) : Upload csv file here", type=None, accept_multiple_files=False, key=None, help=None)
			
				
		
			#df = pd.read_csv(csv_file)
			#if df.empty:
			#	st.write("You did not upload file")

			prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}

			if st.button("Classify"):
				
				if csv_file == None:
					st.warning("You did not upload a file ")
					raise Exception(" please upload your csv file and then classify")

				#Loading the user csv as a dataframe

				input_df = pd.read_csv(csv_file)

				if selected_model == "Logistic Regression":

					predictor = load_model("resources/logreg_count.pkl")
					X_input =tweet_processor(input_df)
					prediction = predictor.predict(X_input)

               	    # st.write(prediction)
				elif selected_model == "Linear SVC":

					predictor = load_model("resources/lsvc_tfidf.pkl")
					X_input =tweet_processor(input_df )
					prediction = predictor.predict(X_input)
                    # st.write(prediction)
				elif selected_model == "Naive Bayes multinomial":
					predictor = load_model("resources/nbm_count.pkl")
					X_input =tweet_processor(input_df)
					prediction = predictor.predict(user_input)
                    # st.write(prediction)
				elif selected_model == "Ridge classifier":
					predictor = load_model("resources/ridge_count.pkl")
					X_input =tweet_processor(input_df)
					prediction = predictor.predict(X_input)







			
 
			st.markdown("![Alt Text](https://media2.giphy.com/media/k4ZItrTKDPnSU/giphy.gif?cid=ecf05e47un87b9ktbh6obdp7kooy4ish81nxm6n9c19kmnqw&rid=giphy.gif&ct=g)")
    
	# Building out the "Data Visualization" page

	if selection == "Sentiment Visualization" :
		st.title("Sentiment Visualization")
		st.info("This page shows various visuals which display the general sentiment of South-Africa towards climate change.")

		# You can read a markdown file from supporting resources folder
		#.write(train_df [['sentiment', 'message']]) # will write the df to the page
		
  		#Word cloud of all the words used on climate change
		st.header('Twitter Word Cloud')
		st.write('This Twitter bird word cloud represents the most frequently recurring words on climate change of all sentiments in the Twitter dataset.')
		Twitter_word_cloud =Image.open('resources/imgs/Image 1 - Twitter bird.png') 
		st.image(Twitter_word_cloud,caption="Twitter Climate Change Word Cloud", width=750)


 	      # Labeling the target
		train_df['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in train_df['sentiment']]

		showPyplotGlobalUse = False
		
	## 'Number of Tweets Per Sentiment' bargraph
		row1_space1, center_, row1_space2 = st.beta_columns((.5, 1, .2, ))
		st.header('Number of Tweets Per Sentiment')
		st.write('This first chart will display the amount of Tweets in the given dataset that represents each sentiment respectively.')

		fig3 =Figure()
		ax = fig3.subplots()
		colors = ['green', 'blue', 'yellow', 'red']
		sns.countplot(x='sentiment' ,data =train_df ,palette='PRGn',ax=ax)
		ax.set_ylabel('Number Of Tweets')
		plt.title('Number of Tweets Per Sentiment')
		st.pyplot(fig3)
		st.write("")

		## Plotting 'Top metnions on climate change per sentiment
		row1_space1, center_, row1_space2 = st.beta_columns((.2, 1, .2, ))

		train_df['users'] = [''.join(re.findall(r'@\w{,}', line)) if '@' in line else np.nan for line in train_df.message]
	
		st.header('The top 10 mentions by sentiment')
		st.write('These four charts will display the top people or entities that are mentioned in Tweets for each one of the four sentiments under consideration.')
		 
		row1_space1, row1_1, row1_space1, row1_2, row1_space1 = st.beta_columns((.1, 1, .1, 1, .1))
		with row1_1, _lock:
			st.subheader('Top 10 positive mentions')
			fig5 =Figure()
			ax = fig5.subplots()
			sns.countplot(y="users", data=train_df[train_df['sentiment'] == 'Positive'],order=train_df[train_df['sentiment'] == 'Positive'].users.value_counts().iloc[:10].index,ax=ax) 
			plt.ylabel('Total Number Of Mentions')
			st.pyplot(fig5)

		with row1_2, _lock:
			st.subheader("Top 10 negative mentions")
			fig6 =Figure()
			ax = fig6.subplots()
			sns.countplot(y="users", data=train_df[train_df['sentiment'] == 'Negative'],order=train_df[train_df['sentiment'] == 'Negative'].users.value_counts().iloc[:10].index,ax=ax) 
			plt.ylabel('Total Number Of Mentions')
			st.pyplot(fig6)

		row1_space1, center_, row1_space2 = st.beta_columns((.2, 1, .2, ))
		with row1_1,_lock :
			st.subheader( 'Top 10 news mentions') 
			fig7 =Figure()
			ax = fig7.subplots()
			sns.countplot(y="users", data=train_df[train_df['sentiment'] == 'News'],order=train_df[train_df['sentiment'] == 'News'].users.value_counts().iloc[:10].index,ax=ax) 
			plt.xlabel('User')
			plt.ylabel('Total Number Of Mentions')
			st.pyplot(fig7)

		with row1_2,_lock :
			st.subheader( 'Top 10 neutral mentions') 
			fig7 =Figure()
			ax = fig7.subplots()
			sns.countplot(y="users", data=train_df[train_df['sentiment'] == 'Neutral'],order=train_df[train_df['sentiment'] == 'Neutral'].users.value_counts().iloc[:10].index,ax=ax) 
			plt.xlabel('User')
			plt.ylabel('Total Number Of Mentions')
			st.pyplot(fig7)
		
		#Displaying the top hashtags on all four sentiments.
		#train_df['users'] = [''.join(re.findall(r'#\w{,}', line)) if '#' in line else '' for line in train_df.message]
		#train_df['users'] = train_df[train_df['users'] != '']

		#st.header('The top 10 hashtags by sentiment')
		#st.write('These four charts will display the top words hashtagged in Tweets for each one of the four sentiments under consideration.')
 
		#row1_space1, row1_1, row1_space1, row1_2, row1_space1 = st.beta_columns((.1, 1, .1, 1, .1))
		#with row1_1, _lock:
		#	st.subheader('Top 10 positive hashtagged words')
		#	fig15 =Figure()
		#	ax = fig15.subplots()
		#	sns.countplot(y="users", data=train_df[train_df['sentiment'] == 'Positive'],order=train_df[train_df['sentiment'] == 'Positive'].users.value_counts().iloc[:10].index,ax=ax) 
		#	plt.ylabel('Total Number Of Tags')
		#	st.pyplot(fig15)

		#with row1_2, _lock:
		#	st.subheader("Top 10 negative hashtagged words")
		#	fig6 =Figure()
		#	ax = fig6.subplots()
		#	sns.countplot(y="hashtags", data=train_df[train_df['sentiment'] == 'Negative'],order=train_df[train_df['sentiment'] == 'Negative'].users.value_counts().iloc[:10].index,ax=ax) 
		#	plt.ylabel('Total Number Of Tags')
		#	st.pyplot(fig6)

		#row1_space1, center_, row1_space2 = st.beta_columns((.2, 1, .2, ))
		#with row1_1,_lock :
		#	st.subheader( 'Top 10 news hashtagged words') 
		#	fig7 =Figure()
		#	ax = fig7.subplots()
		#	sns.countplot(y="hashtags", data=train_df[train_df['sentiment'] == 'News'],order=train_df[train_df['sentiment'] == 'News'].users.value_counts().iloc[:10].index,ax=ax) 
		#	plt.xlabel('User')
		#	plt.ylabel('Total Number Of Tags')
		#	st.pyplot(fig7)

		#with row1_2,_lock :
		#	st.subheader( 'Top 10 neutral hashtagged') 
		#	fig7 =Figure()
		#	ax = fig7.subplots()
		#	sns.countplot(y="hashtags", data=train_df[train_df['sentiment'] == 'Neutral'],order=train_df[train_df['sentiment'] == 'Neutral'].users.value_counts().iloc[:10].index,ax=ax) 
		#	plt.xlabel('User')
		#	plt.ylabel('Total Number Of Tags')
		#	st.pyplot(fig7)
       
		#
		#corpus = re.sub("climate change", ''," ".join(tweet.strip() for tweet in train_df['clean'][working_df['sentiment'] == 'Positive']))
		#wordcloud = WordCloud(font_path='../input/droidsansmonottf/droidsansmono.ttf', background_color="white",width = 1920, height = 1080, colormap="viridis").generate(corpus)
		#plt.figure(dpi=260)
		#plt.imshow(wordcloud, interpolation='bilinear')
		#plt.axis("off")
		#plt.show()








        ## C

        





			

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
