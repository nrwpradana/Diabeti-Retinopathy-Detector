import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
from keras.models import load_model
from time import sleep
import tensorflow.keras.backend as K
import gc

@st.cache(allow_output_mutation=True)
def load_the_model():
    model = load_model('Finished_Model.h5')
    model.summary()  # included to make it visible when model is reloaded
    return model


st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)


st.title("Diabetic Retinopathy Detector")
st.subheader("by Nadhiar")
#st.text("Unggah gambar fundus disini ")

model = load_the_model()

#file_bytes
#opencv_image
#copy_image
#file
#result
#result_arr
#result_array
#message
#decimal




def callback_func():
  
  try:
      
      file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
      opencv_image = cv2.imdecode(file_bytes, 1)
      copy_image = cv2.resize(opencv_image, (320, 240), interpolation = cv2.INTER_LANCZOS4)
      st.image(copy_image, channels="BGR")

      file = cv2.resize(opencv_image, (224, 224), interpolation = cv2.INTER_LANCZOS4)
      file = file.astype('float32')
      file /= 255
      file = file.reshape(1, 224, 224, 3)

      result = model.predict(file).tolist()
      result = result[0]
      result_array = np.array(result)
      result_array *= 100
      result = result_array.tolist()

      #st.write(result)
  
      result_arr = []
      for i in result:
        result_arr.append(str(i)[0:2] + '.' + str(i)[3] + '% confidence')

      og = 0
      og_counter = 0
      counter = 0
      for i in result:
        if i > og:
          og = i
          og_counter = counter
        counter += 1

      message = str(og)[0:2]
      decimal = str(og)[3]

      if og_counter == 0:
        message = '"Normal"(0) dengan level confidence ' + str(message) + '.' + str(decimal) + '% confidence.'
      elif og_counter == 1:
        message = '"Mild Nonproliferative Retinopathy"(1) dengan level confidence ' + str(message) + '.' + str(decimal) + '% confidence.'
      elif og_counter == 2:
        message = '"Moderate Nonproliferative Retinopathy"(2) dengan level confidence ' + str(message) + '.' + str(decimal) + '% confidence.'
      elif og_counter == 3:
        message = '"Severe Nonproliferative Retinopathy"(3) dengan level confidence ' + str(message) + '.' + str(decimal) + '% confidence.'
      elif og_counter == 4:
        message = '"Proliferative Retinopathy"(4) dengan level confidence ' + str(message) + '.' + str(decimal) + '% confidence.'
      #st.write(result_arr)

      st.markdown('<p class="big-font">' + message + '</p>', unsafe_allow_html=True)
      
      del file_bytes
      del opencv_image
      del copy_image
      del file
      del result
      del result_arr
      del result_array
      del message
      del decimal
      gc.collect()
      
  except:
      sleep(5)





def img2arr(filepath):
  im = cv2.imread(filepath)
  im = cv2.resize(im, (224, 224), interpolation = cv2.INTER_LANCZOS4)
    
  return im

uploaded_file = st.file_uploader("Upload Gambar Fundus", on_change=callback_func())

if uploaded_file is not None: 
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  copy_image = cv2.resize(opencv_image, (320, 240), interpolation = cv2.INTER_LANCZOS4)
  st.image(copy_image, channels="BGR")

  file = cv2.resize(opencv_image, (224, 224), interpolation = cv2.INTER_LANCZOS4)
  file = file.astype('float32')
  file /= 255
  file = file.reshape(1, 224, 224, 3)

  result = model.predict(file).tolist()
  result = result[0]
  result_array = np.array(result)
  result_array *= 100
  result = result_array.tolist()
  
  #st.write(result)
  result_arr = []
  for i in result:
    result_arr.append(str(i)[0:2] + '.' + str(i)[3] + '% confidence')
  
  og = 0
  og_counter = 0
  counter = 0
  for i in result:
    if i > og:
      og = i
      og_counter = counter
    counter += 1

  message = str(og)[0:2]
  decimal = str(og)[3]

  if og_counter == 0:
    message = '"Normal"(0) dengan level confidence ' + str(message) + '.' + str(decimal) + '% confidence.'
  elif og_counter == 1:
    message = '"Mild Nonproliferative Retinopathy"(1) dengan level confidence ' + str(message) + '.' + str(decimal) + '% confidence.'
  elif og_counter == 2:
    message = '"Moderate Nonproliferative Retinopathy"(2) dengan level confidence ' + str(message) + '.' + str(decimal) + '% confidence.'
  elif og_counter == 3:
    message = '"Severe Nonproliferative Retinopathy"(3) dengan level confidence ' + str(message) + '.' + str(decimal) + '% confidence.'
  elif og_counter == 4:
    message = '"Proliferative Retinopathy"(4) dengan level confidence ' + str(message) + '.' + str(decimal) + '% confidence.'
  #st.write(result_arr)

  st.markdown('<p class="big-font">' + message + '</p>', unsafe_allow_html=True)
  
  del file_bytes
  del opencv_image
  del copy_image
  del file
  del result
  del result_arr
  del result_array
  del message
  del decimal
  
  gc.collect()
