import streamlit as st
import pickle as p
import numpy as np
from streamlit_option_menu import option_menu

adaboost_model = p.load(open('adaboost_model','rb'))
xgboost_model = p.load(open('xgboost_model','rb'))
lightgbm_model = p.load(open('light_gbm_model','rb'))
catboost_model = p.load(open('catboost_model','rb'))
stoch_model = p.load(open('stochastic_gradient_descent','rb'))
decision_model = p.load(open('decision_tree','rb'))
logreg_model = p.load(open('logreg_model','rb'))
randfor_model = p.load(open('random_forest','rb'))
knn_model = p.load(open('knn_model','rb'))
perc_model = p.load(open('perceptron_model','rb'))
multperc_model = p.load(open('Multperc_model','rb'))
selected = ""
selected1 = ''
st.title('Song Recommendation System')


# with st.tabs:
#     st.subheader('Model')
#     st.write('Model')
#     st.write(option_menu(adaboost_model, 'adaboost_model'))

with st.sidebar:

    st.subheader('Options')
    opted = option_menu('Sections',['Model_predictions','Model_Metrics'])
    if opted == 'Model_predictions':
        selected = ""
        selected1 = ''
        with st.sidebar:

            selected = option_menu('Multiple models for song recommendation',
                            ['Ada_Boost_Model','Catboost_model','lightgbm_model','xgboost_model','knn_model',
                            'logreg_model','stoch_model','decision_model','randfor_model','Multilayer_perc','Perceptron'],
                            
                            default_index=0)
    if opted == 'Model_Metrics':
        selected = ""
        selected1 = ''
        with st.sidebar:

            selected1 = option_menu('Models metrics',
                            ['Ada_Boost_Model','Catboost_model','lightgbm_model','xgboost_model','knn_model',
                            'logreg_model','stoch_model','decision_model','randfor_model','Multilayer_perc','Perceptron'],
                            default_index=0)
            
if selected == 'Multilayer_perc':
    
    st.title("Prediction using Multilayer Perceptron")

    col1,col2,col3= st.columns(3)

    with col1:
        member_id = st.text_input("Enter the Member id:")
    with col2:
        song_id = st.text_input("Enter the song id:")
    with col3:
        source_system_tab = st.text_input("Enter the source tab")
    col4,col5,col6 = st.columns(3)
    with col4:
        source_screen_name = st.text_input("Enter the source screen name")
    with col5:
        source_type = st.text_input("Enter the source type")
    with col6:
        song_length = st.text_input("Enter the song length")
    col7,col8,col9 =  st.columns(3)
    with col7:
        artist_name = st.text_input("Enter the artist name")
    with col8:
        composer = st.text_input("Enter the composer")
    with col9:
        lyricist = st.text_input("Enter the lyricist")
    col10,col11,col12 =  st.columns(3)
    with col10:
        language = st.text_input("Enter the language")
    with col11:
        genre_id = st.text_input("Enter the genre id")
    with col12:
        city = st.text_input("Enter the city")
    col13,col14,col15 =st.columns(3)
    with col13:
        bd = st.text_input("Enter the birth day")
    with col14:
        gender = st.text_input("Enter the gender")
    with col15:
        registered_via = st.text_input("Enter the registered via name")
    col16,col17 = st.columns(2)
    with col16:
        registration_init_time = st.text_input("Enter Registration time")
    with col17:
        expiration_date = st.text_input("Enter the expiration date")

    text_message = ''

    if st.button('Predict'):

        
        mulprec_prediction = multperc_model.predict([[member_id,song_id,source_system_tab,source_screen_name,
                                        source_type,song_length,artist_name,composer,lyricist,
                                        language,genre_id,city,bd,gender,registered_via,registration_init_time,expiration_date]])
        
        if mulprec_prediction == 1:
            text_message = "The user will listen to the song again"
        else:
            text_message = "The user won't listen to the song again"
    
        st.success(text_message)

if selected == 'Perceptron':
    
    st.title("Prediction using Perceptron")

    col1,col2,col3= st.columns(3)

    with col1:
        member_id = st.text_input("Enter the Member id:")
    with col2:
        song_id = st.text_input("Enter the song id:")
    with col3:
        source_system_tab = st.text_input("Enter the source tab")
    col4,col5,col6 = st.columns(3)
    with col4:
        source_screen_name = st.text_input("Enter the source screen name")
    with col5:
        source_type = st.text_input("Enter the source type")
    with col6:
        song_length = st.text_input("Enter the song length")
    col7,col8,col9 =  st.columns(3)
    with col7:
        artist_name = st.text_input("Enter the artist name")
    with col8:
        composer = st.text_input("Enter the composer")
    with col9:
        lyricist = st.text_input("Enter the lyricist")
    col10,col11,col12 =  st.columns(3)
    with col10:
        language = st.text_input("Enter the language")
    with col11:
        genre_id = st.text_input("Enter the genre id")
    with col12:
        city = st.text_input("Enter the city")
    col13,col14,col15 =st.columns(3)
    with col13:
        bd = st.text_input("Enter the birth day")
    with col14:
        gender = st.text_input("Enter the gender")
    with col15:
        registered_via = st.text_input("Enter the registered via name")
    col16,col17 = st.columns(2)
    with col16:
        registration_init_time = st.text_input("Enter Registration time")
    with col17:
        expiration_date = st.text_input("Enter the expiration date")

    text_message = ''

    if st.button('Predict'):

        
        perc_prediction = perc_model.predict([[member_id,song_id,source_system_tab,source_screen_name,
                                        source_type,song_length,artist_name,composer,lyricist,
                                        language,genre_id,city,bd,gender,registered_via,registration_init_time,expiration_date]])
        
        if perc_prediction == 1:
            text_message = "The user will listen to the song again"
        else:
            text_message = "The user won't listen to the song again"
    
        st.success(text_message)

if selected == 'randfor_model':
    
    st.title("Prediction using Random Forest")

    col1,col2,col3= st.columns(3)

    with col1:
        member_id = st.text_input("Enter the Member id:")
    with col2:
        song_id = st.text_input("Enter the song id:")
    with col3:
        source_system_tab = st.text_input("Enter the source tab")
    col4,col5,col6 = st.columns(3)
    with col4:
        source_screen_name = st.text_input("Enter the source screen name")
    with col5:
        source_type = st.text_input("Enter the source type")
    with col6:
        song_length = st.text_input("Enter the song length")
    col7,col8,col9 =  st.columns(3)
    with col7:
        artist_name = st.text_input("Enter the artist name")
    with col8:
        composer = st.text_input("Enter the composer")
    with col9:
        lyricist = st.text_input("Enter the lyricist")
    col10,col11,col12 =  st.columns(3)
    with col10:
        language = st.text_input("Enter the language")
    with col11:
        genre_id = st.text_input("Enter the genre id")
    with col12:
        city = st.text_input("Enter the city")
    col13,col14,col15 =st.columns(3)
    with col13:
        bd = st.text_input("Enter the birth day")
    with col14:
        gender = st.text_input("Enter the gender")
    with col15:
        registered_via = st.text_input("Enter the registered via name")
    col16,col17 = st.columns(2)
    with col16:
        registration_init_time = st.text_input("Enter Registration time")
    with col17:
        expiration_date = st.text_input("Enter the expiration date")

    text_message = ''

    if st.button('Predict'):

        
        rand_prediction = randfor_model.predict([[member_id,song_id,source_system_tab,source_screen_name,
                                        source_type,song_length,artist_name,composer,lyricist,
                                        language,genre_id,city,bd,gender,registered_via,registration_init_time,expiration_date]])
        
        if rand_prediction == 1:
            text_message = "The user will listen to the song again"
        else:
            text_message = "The user won't listen to the song again"
    
        st.success(text_message)

            
if selected == 'decision_model':
    
    st.title("Prediction using Decision Tree")

    col1,col2,col3= st.columns(3)

    with col1:
        member_id = st.text_input("Enter the Member id:")
    with col2:
        song_id = st.text_input("Enter the song id:")
    with col3:
        source_system_tab = st.text_input("Enter the source tab")
    col4,col5,col6 = st.columns(3)
    with col4:
        source_screen_name = st.text_input("Enter the source screen name")
    with col5:
        source_type = st.text_input("Enter the source type")
    with col6:
        song_length = st.text_input("Enter the song length")
    col7,col8,col9 =  st.columns(3)
    with col7:
        artist_name = st.text_input("Enter the artist name")
    with col8:
        composer = st.text_input("Enter the composer")
    with col9:
        lyricist = st.text_input("Enter the lyricist")
    col10,col11,col12 =  st.columns(3)
    with col10:
        language = st.text_input("Enter the language")
    with col11:
        genre_id = st.text_input("Enter the genre id")
    with col12:
        city = st.text_input("Enter the city")
    col13,col14,col15 =st.columns(3)
    with col13:
        bd = st.text_input("Enter the birth day")
    with col14:
        gender = st.text_input("Enter the gender")
    with col15:
        registered_via = st.text_input("Enter the registered via name")
    col16,col17 = st.columns(2)
    with col16:
        registration_init_time = st.text_input("Enter Registration time")
    with col17:
        expiration_date = st.text_input("Enter the expiration date")

    text_message = ''

    if st.button('Predict'):

        
        des_prediction = decision_model.predict([[member_id,song_id,source_system_tab,source_screen_name,
                                        source_type,song_length,artist_name,composer,lyricist,
                                        language,genre_id,city,bd,gender,registered_via,registration_init_time,expiration_date]])
        
        if des_prediction == 1:
            text_message = "The user will listen to the song again"
        else:
            text_message = "The user won't listen to the song again"
    
        st.success(text_message)

if selected == 'stoch_model':
    
    st.title("Prediction using Stochastic Gradient Descent")

    col1,col2,col3= st.columns(3)

    with col1:
        member_id = st.text_input("Enter the Member id:")
    with col2:
        song_id = st.text_input("Enter the song id:")
    with col3:
        source_system_tab = st.text_input("Enter the source tab")
    col4,col5,col6 = st.columns(3)
    with col4:
        source_screen_name = st.text_input("Enter the source screen name")
    with col5:
        source_type = st.text_input("Enter the source type")
    with col6:
        song_length = st.text_input("Enter the song length")
    col7,col8,col9 =  st.columns(3)
    with col7:
        artist_name = st.text_input("Enter the artist name")
    with col8:
        composer = st.text_input("Enter the composer")
    with col9:
        lyricist = st.text_input("Enter the lyricist")
    col10,col11,col12 =  st.columns(3)
    with col10:
        language = st.text_input("Enter the language")
    with col11:
        genre_id = st.text_input("Enter the genre id")
    with col12:
        city = st.text_input("Enter the city")
    col13,col14,col15 =st.columns(3)
    with col13:
        bd = st.text_input("Enter the birth day")
    with col14:
        gender = st.text_input("Enter the gender")
    with col15:
        registered_via = st.text_input("Enter the registered via name")
    col16,col17 = st.columns(2)
    with col16:
        registration_init_time = st.text_input("Enter Registration time")
    with col17:
        expiration_date = st.text_input("Enter the expiration date")

    text_message = ''

    if st.button('Predict'):

        
        sto_prediction = stoch_model.predict([[member_id,song_id,source_system_tab,source_screen_name,
                                        source_type,song_length,artist_name,composer,lyricist,
                                        language,genre_id,city,bd,gender,registered_via,registration_init_time,expiration_date]])
        
        if sto_prediction == 1:
            text_message = "The user will listen to the song again"
        else:
            text_message = "The user won't listen to the song again"
    
        st.success(text_message)


if selected == 'logreg_model':
    
    st.title("Prediction using Logistic Regression")

    col1,col2,col3= st.columns(3)

    with col1:
        member_id = st.text_input("Enter the Member id:")
    with col2:
        song_id = st.text_input("Enter the song id:")
    with col3:
        source_system_tab = st.text_input("Enter the source tab")
    col4,col5,col6 = st.columns(3)
    with col4:
        source_screen_name = st.text_input("Enter the source screen name")
    with col5:
        source_type = st.text_input("Enter the source type")
    with col6:
        song_length = st.text_input("Enter the song length")
    col7,col8,col9 =  st.columns(3)
    with col7:
        artist_name = st.text_input("Enter the artist name")
    with col8:
        composer = st.text_input("Enter the composer")
    with col9:
        lyricist = st.text_input("Enter the lyricist")
    col10,col11,col12 =  st.columns(3)
    with col10:
        language = st.text_input("Enter the language")
    with col11:
        genre_id = st.text_input("Enter the genre id")
    with col12:
        city = st.text_input("Enter the city")
    col13,col14,col15 =st.columns(3)
    with col13:
        bd = st.text_input("Enter the birth day")
    with col14:
        gender = st.text_input("Enter the gender")
    with col15:
        registered_via = st.text_input("Enter the registered via name")
    col16,col17 = st.columns(2)
    with col16:
        registration_init_time = st.text_input("Enter Registration time")
    with col17:
        expiration_date = st.text_input("Enter the expiration date")

    text_message = ''

    if st.button('Predict'):

        
        log_prediction = logreg_model.predict([[member_id,song_id,source_system_tab,source_screen_name,
                                        source_type,song_length,artist_name,composer,lyricist,
                                        language,genre_id,city,bd,gender,registered_via,registration_init_time,expiration_date]])
        
        if log_prediction == 1:
            text_message = "The user will listen to the song again"
        else:
            text_message = "The user won't listen to the song again"
    
        st.success(text_message)

if selected == 'knn_model':
    
    st.title("Prediction using KNN")

    col1,col2,col3= st.columns(3)

    with col1:
        member_id = st.text_input("Enter the Member id:")
    with col2:
        song_id = st.text_input("Enter the song id:")
    with col3:
        source_system_tab = st.text_input("Enter the source tab")
    col4,col5,col6 = st.columns(3)
    with col4:
        source_screen_name = st.text_input("Enter the source screen name")
    with col5:
        source_type = st.text_input("Enter the source type")
    with col6:
        song_length = st.text_input("Enter the song length")
    col7,col8,col9 =  st.columns(3)
    with col7:
        artist_name = st.text_input("Enter the artist name")
    with col8:
        composer = st.text_input("Enter the composer")
    with col9:
        lyricist = st.text_input("Enter the lyricist")
    col10,col11,col12 =  st.columns(3)
    with col10:
        language = st.text_input("Enter the language")
    with col11:
        genre_id = st.text_input("Enter the genre id")
    with col12:
        city = st.text_input("Enter the city")
    col13,col14,col15 =st.columns(3)
    with col13:
        bd = st.text_input("Enter the birth day")
    with col14:
        gender = st.text_input("Enter the gender")
    with col15:
        registered_via = st.text_input("Enter the registered via name")
    col16,col17 = st.columns(2)
    with col16:
        registration_init_time = st.text_input("Enter Registration time")
    with col17:
        expiration_date = st.text_input("Enter the expiration date")

    text_message = ''

    if st.button('Predict'):

        
        knn_prediction = knn_model.predict([[member_id,song_id,source_system_tab,source_screen_name,
                                        source_type,song_length,artist_name,composer,lyricist,
                                        language,genre_id,city,bd,gender,registered_via,registration_init_time,expiration_date]])
        
        if knn_prediction == 1:
            text_message = "The user will listen to the song again"
        else:
            text_message = "The user won't listen to the song again"
    
        st.success(text_message)

if selected == 'Ada_Boost_Model':

    st.title("Prediction using ADA Boosting")

    col1,col2,col3= st.columns(3)

    with col1:
        member_id = st.text_input("Enter the Member id:")
    with col2:
        song_id = st.text_input("Enter the song id:")
    with col3:
        source_system_tab = st.text_input("Enter the source tab")
    col4,col5,col6 = st.columns(3)
    with col4:
        source_screen_name = st.text_input("Enter the source screen name")
    with col5:
        source_type = st.text_input("Enter the source type")
    with col6:
        song_length = st.text_input("Enter the song length")
    col7,col8,col9 =  st.columns(3)
    with col7:
        artist_name = st.text_input("Enter the artist name")
    with col8:
        composer = st.text_input("Enter the composer")
    with col9:
        lyricist = st.text_input("Enter the lyricist")
    col10,col11,col12 =  st.columns(3)
    with col10:
        language = st.text_input("Enter the language")
    with col11:
        genre_id = st.text_input("Enter the genre id")
    with col12:
        city = st.text_input("Enter the city")
    col13,col14,col15 =st.columns(3)
    with col13:
        bd = st.text_input("Enter the birth day")
    with col14:
        gender = st.text_input("Enter the gender")
    with col15:
        registered_via = st.text_input("Enter the registered via name")
    col16,col17 = st.columns(2)
    with col16:
        registration_init_time = st.text_input("Enter Registration time")
    with col17:
        expiration_date = st.text_input("Enter the expiration date")

    text_message = ''

    if st.button('Predict'):

        
        ada_prediction = adaboost_model.predict([[member_id,song_id,source_system_tab,source_screen_name,
                                        source_type,song_length,artist_name,composer,lyricist,
                                        language,genre_id,city,bd,gender,registered_via,registration_init_time,expiration_date]])
        
        if ada_prediction == 1:
            text_message = "The user will listen to the song again"
        else:
            text_message = "The user won't listen to the song again"
    
        st.success(text_message)

if selected == 'Catboost_model':
    
    st.title("Prediction using CAT Boosting")

    col1,col2,col3= st.columns(3)

    with col1:
        member_id = st.text_input("Enter the Member id:")
    with col2:
        song_id = st.text_input("Enter the song id:")
    with col3:
        source_system_tab = st.text_input("Enter the source tab")
    col4,col5,col6 = st.columns(3)
    with col4:
        source_screen_name = st.text_input("Enter the source screen name")
    with col5:
        source_type = st.text_input("Enter the source type")
    with col6:
        song_length = st.text_input("Enter the song length")
    col7,col8,col9 =  st.columns(3)
    with col7:
        artist_name = st.text_input("Enter the artist name")
    with col8:
        composer = st.text_input("Enter the composer")
    with col9:
        lyricist = st.text_input("Enter the lyricist")
    col10,col11,col12 =  st.columns(3)
    with col10:
        language = st.text_input("Enter the language")
    with col11:
        genre_id = st.text_input("Enter the genre id")
    with col12:
        city = st.text_input("Enter the city")
    col13,col14,col15 =st.columns(3)
    with col13:
        bd = st.text_input("Enter the birth day")
    with col14:
        gender = st.text_input("Enter the gender")
    with col15:
        registered_via = st.text_input("Enter the registered via name")
    col16,col17 = st.columns(2)
    with col16:
        registration_init_time = st.text_input("Enter Registration time")
    with col17:
        expiration_date = st.text_input("Enter the expiration date")

    text_message = ''

    if st.button('Predict'):
        cat_prediction = catboost_model.predict([[member_id,song_id,source_system_tab,source_screen_name,
                                        source_type,song_length,artist_name,composer,lyricist,
                                        language,genre_id,city,bd,gender,registered_via,registration_init_time,expiration_date]])
        
        if cat_prediction == 1:
            text_message = "The user will listen to the song again"
        else:
            text_message = "The user won't listen to the song again"
    
        st.success(text_message)


if selected == 'lightgbm_model':
    
    st.title("Prediction using LIGHT_GBM Boosting")

    col1,col2,col3= st.columns(3)

    with col1:
        member_id = st.text_input("Enter the Member id:")
    with col2:
        song_id = st.text_input("Enter the song id:")
    with col3:
        source_system_tab = st.text_input("Enter the source tab")
    col4,col5,col6 = st.columns(3)
    with col4:
        source_screen_name = st.text_input("Enter the source screen name")
    with col5:
        source_type = st.text_input("Enter the source type")
    with col6:
        song_length = st.text_input("Enter the song length")
    col7,col8,col9 =  st.columns(3)
    with col7:
        artist_name = st.text_input("Enter the artist name")
    with col8:
        composer = st.text_input("Enter the composer")
    with col9:
        lyricist = st.text_input("Enter the lyricist")
    col10,col11,col12 =  st.columns(3)
    with col10:
        language = st.text_input("Enter the language")
    with col11:
        genre_id = st.text_input("Enter the genre id")
    with col12:
        city = st.text_input("Enter the city")
    col13,col14,col15 =st.columns(3)
    with col13:
        bd = st.text_input("Enter the birth day")
    with col14:
        gender = st.text_input("Enter the gender")
    with col15:
        registered_via = st.text_input("Enter the registered via name")
    col16,col17 = st.columns(2)
    with col16:
        registration_init_time = st.text_input("Enter Registration time")
    with col17:
        expiration_date = st.text_input("Enter the expiration date")

    text_message = ''

    if st.button('Predict'):
        lightgbm_prediction = lightgbm_model.predict([[member_id,song_id,source_system_tab,source_screen_name,
                                        source_type,song_length,artist_name,composer,lyricist,
                                        language,genre_id,city,bd,gender,registered_via,registration_init_time,expiration_date]])
        
        if lightgbm_prediction == 1:
            text_message = "The user will listen to the song again"
        else:
            text_message = "The user won't listen to the song again"
    
        st.success(text_message)


if selected == 'xgboost_model':
    
    st.title("Prediction using XG Boosting")

    col1,col2,col3= st.columns(3)

    with col1:
        member_id = st.text_input("Enter the Member id:")
    with col2:
        song_id = st.text_input("Enter the song id:")
    with col3:
        source_system_tab = st.text_input("Enter the source tab")
    col4,col5,col6 = st.columns(3)
    with col4:
        source_screen_name = st.text_input("Enter the source screen name")
    with col5:
        source_type = st.text_input("Enter the source type")
    with col6:
        song_length = st.text_input("Enter the song length")
    col7,col8,col9 =  st.columns(3)
    with col7:
        artist_name = st.text_input("Enter the artist name")
    with col8:
        composer = st.text_input("Enter the composer")
    with col9:
        lyricist = st.text_input("Enter the lyricist")
    col10,col11,col12 =  st.columns(3)
    with col10:
        language = st.text_input("Enter the language")
    with col11:
        genre_id = st.text_input("Enter the genre id")
    with col12:
        city = st.text_input("Enter the city")
    col13,col14,col15 =st.columns(3)
    with col13:
        bd = st.text_input("Enter the birth day")
    with col14:
        gender = st.text_input("Enter the gender")
    with col15:
        registered_via = st.text_input("Enter the registered via name")
    col16,col17 = st.columns(2)
    with col16:
        registration_init_time = st.text_input("Enter Registration time")
    with col17:
        expiration_date = st.text_input("Enter the expiration date")

    text_message = ''

    if st.button('Predict'):
        xg_prediction = xgboost_model.predict([[int(member_id),int(song_id),int(source_system_tab),int(source_screen_name),
                                        int(source_type),int(song_length),int(artist_name),int(composer),int(lyricist),
                                        int(language),int(genre_id),int(city),int(bd),int(gender),int(registered_via),int(registration_init_time),int(expiration_date)]])
        
        if xg_prediction == 1:
            text_message = "The user will listen to the song again"
        else:
            text_message = "The user won't listen to the song again"
    
        st.success(text_message)




# ['Ada_Boost_Model','Catboost_model','lightgbm_model','xgboost_model','knn_model',
#                             'logreg_model','stoch_model','decision_model','randfor_model']

if selected1 == 'Ada_Boost_Model':
    st.write('Metrics of Ada boost Model')
    col1,col2 = st.columns(2)
    with col1:
        st.metric(label='Accuracy Score',value="65%")
    with col2:
        st.metric(label='Precision',value="63%")

if selected1 == 'xgboost_model':
    st.write('Metrics of Ada boost Model')
    col1,col2 = st.columns(2)
    with col1:
        st.metric(label='Accuracy Score',value="74%")
    with col2:
        st.metric(label='Precision',value="72%")

if selected1 == 'Catboost_model':
    st.write('Metrics of Ada boost Model')
    col1,col2 = st.columns(2)
    with col1:
        st.metric(label='Accuracy Score',value="73%")
    with col2:
        st.metric(label='Precision',value="70%")

if selected1 == 'lightgbm_model':
    st.write('Metrics of Lightgbm Model')
    col1,col2 = st.columns(2)
    with col1:
        st.metric(label='Accuracy Score',value="73%")
    with col2:
        st.metric(label='Precision',value="72%")

if selected1 == 'knn_model':
    st.write('Metrics of KNN Model')
    col1,col2 = st.column(2)
    with col1:
        st.metric(label='Accuracy Score',value="")
    with col2:
        st.metric(label='Precision',value="")

if selected1 == 'logreg_model':
    st.write('Metrics of Logistic Regression Model')

    col1,col2 = st.columns(2)
    with col1:
        st.metric(label='Accuracy Score',value="50%")
    with col2:
        st.metric(label='Precision',value="44%")

if selected1 =='stoch_model':

    st.write('Metrics of Stochastic Gradient Descent Model')
    col1,col2 = st.columns(2)
    with col1:
        st.metric(label='Accuracy Score',value="43%")
    with col2:
        st.metric(label='Precision',value="42%")

if selected1 == 'decision_model':
    st.write('Metrics of Decision Tree Model')
    col1,col2 = st.columns(2)
    with col1:
        st.metric(label='Accuracy Score',value="64%")
    with col2:
        st.metric(label='Precision',value="62%")

if selected1 == 'randfor_model':
    st.write('Metrics of Random Forest Model')
    col1,col2 = st.columns(2)
    with col1:
        st.metric(label='Accuracy Score',value="74%")
    with col2:
        st.metric(label='Precision',value="74%")

if selected1 == 'Multilayer_perc':
    st.write('Metrics of Multilayer Perceptron Model')
    col1,col2 = st.columns(2)
    with col1:
        st.metric(label='Accuracy Score',value=45)
    with col2:
        st.metric(label='Precision',value=0)

if selected1 == 'Perceptron':
    st.write('Metrics of Perceptron Model')
    col1,col2 = st.columns(2)
    with col1:
        st.metric(label='Accuracy Score',value="48")
    with col2:
        st.metric(label='Precision',value="51")