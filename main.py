import streamlit as st 
import os
from PIL import Image
import tensorflow
import  pickle
import numpy as np
import cv2 
from numpy.linalg import norm
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors



feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',  include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
        model,
        GlobalMaxPooling2D()
])

st.title("Fashion Recommendation System")

def save_file(uploaded_file):
    try:
        with open (os.path.join('uploads',uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def camera_save(img_capture):
    try:
        with open(os.path.join('uploads', img_capture.name), 'wb') as f:
            f.write(img_capture.getbuffer())
        return 1
    except:
        return 0

cam = cv2.VideoCapture(0)
cv2.namedWindow('test')
img_counter = 0

def camera_capture(img_capture):
    while True:
        ret, frame = cam.read()
        if not ret:
            print('failed')
            break
        cv2.imshow('test', frame)

        k = cv2.waitKey(1)

        if k%256 ==27:
            print("escape hit")
            break

        elif k%256 ==32:
            img_name = 'camera\img{}.jpg'.format(img_counter)
            cv2.imwrite(img_name, frame)
            print('Image Captured')
            img_counter +=1

    cam.release()
    cam.destroyAllWindows()
    cv2.waitKey(1)

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_arrays = image.img_to_array(img)
    expand_img_array = np.expand_dims(img_arrays, axis=0)
    prepross_img = preprocess_input(expand_img_array)
    result = model.predict(prepross_img).flatten()
    normalized_result = result/norm(result)

    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distance, indices = neighbors.kneighbors([features])

    return indices


uploaded_file = st.file_uploader("Choose an image")
img_capture = st.camera_input("Capture image")




if uploaded_file is not None:
    if save_file(uploaded_file):
       display_img = Image.open(uploaded_file)
       st.image(display_img)
       st.caption('Search Image')
       features = feature_extraction(os.path.join("uploads",uploaded_file.name), model)
    #    st.text(features)
       indices = recommend(features, feature_list)
       col1, col2, col3, col4, col5 = st.columns(5)

       with col1:
        st.image(filenames[indices[0][0]])
        st.caption('recommendation 1')

       with col2:
        st.image(filenames[indices[0][1]])
        st.caption('recommendation 2')

       with col3:
        st.image(filenames[indices[0][2]])
        st.caption('recommendation 3')

       with col4:
        st.image(filenames[indices[0][3]])
        st.caption('recommendation 4')

       with col5:
        st.image(filenames[indices[0][4]])
        st.caption('recommendation 5')

    else:
        st.header('Some error occured in file upload')


elif img_capture is not None:
    if save_file(img_capture):
       display_img = Image.open(img_capture)
       st.image(display_img)
       st.caption('Search Image')
       features = feature_extraction(os.path.join("uploads", img_capture.name), model)

       indices = recommend(features, feature_list)
       col1, col2, col3, col4, col5 = st.columns(5)

       with col1:
        st.image(filenames[indices[0][0]])
        st.caption('recommendation 1')

       with col2:
        st.image(filenames[indices[0][1]])
        st.caption('recommendation 2')
       with col3:
        st.image(filenames[indices[0][2]])
        st.caption('recommendation 3')

       with col4:
        st.image(filenames[indices[0][3]])
        st.caption('recommendation 4')

       with col5:
        st.image(filenames[indices[0][4]])
        st.caption('recommendation 5')
    else:
        st.header('Camera is not working')

