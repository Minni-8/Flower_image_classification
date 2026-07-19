
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
model = load_model("D:/new project/Flower_image_classification/flower_classifier.keras")
data_cat = ['lily', 'lotus', 'orchid', 'sunflower', 'tulip']

img_width =  180
img_height = 180
st.header('Image Classification Model')
uploaded_file = st.file_uploader("Choose a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_width, img_height))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr,0)

    predict = model.predict(img_bat)

    score = tf.nn.softmax(predict)

    st.image(uploaded_file)

    st.write(
        'Flower in Image is {} with accuracy of {:0.2f}'.format(
        data_cat[np.argmax(score)],
        np.max(score)*100
        )
    )
