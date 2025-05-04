import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

model = tf.keras.models.load_model('Keputusan_konsumen_model.h5') 
scaler = StandardScaler()

st.title('Prediksi Keputusan Pembeli')
Age = st.number_input('Input Usia', min_value=18, max_value=100, step=1)
EstimatedSalary = st.number_input('Input PerKiraan Gaji', min_value=1000, max_value=200000, step=100)

predict = ''

if st.button('Prediksi'):
    input_data = np.array([[Age, EstimatedSalary]])
    input_data_scaled = scaler.fit_transform(input_data)

    prediction = model.predict(input_data_scaled)

    if prediction > 0.5: 
        st.write('Prediksi Keputusan Konsumen : Akan Membeli')
    else:
        st.write('Prediksi Keputusan Konsumen : Tidak Akan Membeli')