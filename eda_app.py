import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def run_eda_app() :
    st.subheader('EDA 화면입니다.')

    car_df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding='ISO-8859-1')
   
    radio_menu = ['데이터프레임','통계치']
    selected_radio = st.radio('선택하세요',radio_menu)

    if selected_radio == '데이터프레임' :
        st.dataframe(car_df)

    elif selected_radio == '통계치' :
        st.dataframe(car_df.describe())

    columns = car_df.columns
    columns = list(columns)

    selected_cols = st.multiselect('컬럼을 선택하세요.' , columns)
    if len(selected_cols) != 0 :
        st.dataframe(car_df[selected_cols])
    else :
        st.write('선택한 컬럼이 없습니다.')
    

    # 상관계수를 화면에 보여주도록 만든다.
    # 멀티셀렉트에 컬럼명 보여주고, 해당 컬럼들에 대한 상관계수를 보여준다.
    # 단, 컬럼들은 숫자 컬럼들만 멀티 셀렉트에 나타나야 한다.

    corr_cols = car_df.columns[car_df.dtypes != object]
    selected_corr = st.multiselect('상관계수 컬럼을 선택하세요.' , corr_cols)

    if len(selected_corr) != 0 :
        st.dataframe(car_df[selected_corr].corr())
        
        # 위에서 선택한 컬럼들을 이용해서, sns.pairplot을 그린다.

        fig = sns.pairplot(data = car_df[selected_corr])
        st.pyplot(fig)

    else :
        st.write('선택한 컬럼이 없습니다.')

    

    # 컬럼을 하나만 선택하면, 해당 컬럼의 max 와 min에 해당하는 사람의 데이터를 화면에 보여준다. (수치 데이터만)
    
    min_max_cols = car_df.columns[car_df.dtypes == float]
    selected_min_max = st.selectbox('Max/Min 값을 볼 컬럼을 선택하세요.', min_max_cols)

    st.write('Max', car_df.loc[car_df[selected_min_max]==car_df[selected_min_max].max(), ])
    st.write('Min', car_df.loc[car_df[selected_min_max]==car_df[selected_min_max].min(), ])


    # 고객 이름을 검색하면, 해당 고객의 데이터를 화면에 보여준다.

    word = st.text_input('고객의 이름을 입력하세요')
    #                                                              대소문자 구분 없이!
    result = car_df.loc[car_df['Customer Name'].str.contains(word, case = False)]
    if len(word) != 0 :
        st.write(word+'님의 데이터를 검색하셨습니다.')
        st.dataframe(result)
        





    


    