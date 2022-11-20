import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input


model = tf.keras.models.load_model("mdl_wt.hdf5")

st.title('Animal Classification Project')
st.header('This app will classify 6 animals')
# st.subheader('This app will classify 6 animals (dog, horse, elephant, butterfly, chicken, cat)')

map_dict = {0: 'Dog',
            1: 'Horse',
            2: 'Elephant',
            3: 'Butterfly',
            4: 'Chicken',
            5: 'Cat',
}

st.write(map_dict)


uploaded_file = st.file_uploader("Choose a image file", type="jpeg")


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized) #The preprocess_input function is meant to adequate your image to the format the model requires.
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        
        st.subheader("Predicted Label for the image is {}".format(map_dict [prediction]))            