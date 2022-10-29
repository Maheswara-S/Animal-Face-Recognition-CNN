import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('animal_face.h5')

st.header('Is it cat? Dog? or wild animal?')

input = st.file_uploader("Pleas input animal image")

if st.button('Submit'):

    st.image(input)

    image = Image.open(input)
    image = np.array(image)
    image_rescale = image/255 
    image_resize = cv2.resize(image_rescale,(224,224))

    pred = model.predict(np.array([image_resize]))
    np.argmax(pred)

    if np.argmax(pred) == 0:
        ouput = "It's a Cat! Nyaaaa~~~"
    elif np.argmax(pred) == 1:
        ouput = "It's a Dog! Woof!"
    else:
        ouput = "It's a wild animal, take precautions!"

    st.text(f'{ouput}')