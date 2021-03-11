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

import joblib

def run_ml_app():
    st.subheader('Machine Learning')

    # 1. 유저한테 입력을 받는다.
    # 성별
    gender = st.radio('성별을 선택하세요.', ['여자', '남자'])
    if gender == '여자' :
        gender = '0'
    else :
        gender = '1'

    # 나이
    age = st.number_input('나이를 입력하세요.', min_value = 0, max_value = 120)

    # 연봉
    salary = st.number_input('연봉을 입력하세요.', min_value = 0)

    # 카드 빚
    debt = st.number_input('카드 빚을 입력하세요.', min_value = 0)

    # 자산
    worth = st.number_input('자산을 입력하세요.', min_value = 0)

    # 2. 예측한다.
    # 2-1. 모델을 불러온다.
    model = tensorflow.keras.models.load_model('data/car_ai.h5')

    # 2-2. np.array를 만든다.
    new_data  = np.array([gender, age, salary, debt, worth])

    # 2-3. 피처 스케일링을 한다.
    new_data = new_data.reshape(1,-1)

    sc_X = joblib.load('data/sc_X.pkl')

    new_data = sc_X.transform(new_data)
    # 2-4. 예측한다.

    y_pred = model.predict(new_data)

    # 예측 결과는, 스케일링 된 결과이므로 다시 돌려야한다.
    st.write(y_pred[0][0])

    sc_y = joblib.load('data/sc_y.pkl')

    y_pred_original = sc_y.inverse_transform(y_pred)

    # st.write(y_pred_original)

    # 3. 결과를 화면에 보여준다.
    bnt = st.button('결과 보기')
    if bnt :
        st.write('예측 결과입니다. {:,.1f}달러의 차를 살 수 있습니다.'.format(y_pred_original[0,0]))

