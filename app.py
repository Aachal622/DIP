import streamlit as st 
from PIL import Image
import pickle
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from werkzeug.utils import secure_filename
st.set_option('deprecation.showfileUploaderEncoding', False)

html_temp = """
   <div class="" style="background-color:gray;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Digital Image Processing lab</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Perform Translation on Image
         """
         )
file = st.file_uploader("Please upload image", type=("jpg", "png"))

import cv2
from  PIL import Image, ImageOps
def import_and_predict():

  img1=cv2.imread(file,1)
  image = cv.cvtColor(img_T, cv.COLOR_BGR2RGB)
  #@title Perform Translation on Images {run:"auto"} 
  Operation = '-50' #@param ["-50", "150"] {allow-input: true}
  if Operation=='-50':
    M1 = np.float32([[1, 0, -50],[0, 1, 100], [0, 0, 1]])
    img1 = cv.warpPerspective(image, M1, (image.shape[1]*2, image.shape[0]*2))
    

  if Operation=='150':
    M2 = np.float32([[1, 0, 50],[0, 1, 150],[0, 0, 1]])
    img1 = cv.warpPerspective(img1, M2, (image.shape[1]*2, image.shape[0]*2))
    

  st.image(image_data, use_column_width=True)
  return 0

if file is None:
  st.text("Please upload an Image file")
else:
  file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  st.image(file,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Translation -50 in X-direction"):
  M1 = np.float32([[1, 0, -50],[0, 1, 100], [0, 0, 1]])
  img1 = cv.warpPerspective(image, M1, (image.shape[1]*2, image.shape[0]*2))
  print(img1)

if st.button("Translation 150 in Y-direction"):
   M2 = np.float32([[1, 0, 50],[0, 1, 150],[0, 0, 1]])
   img1 = cv.warpPerspective(img1, M2, (image.shape[1]*2, image.shape[0]*2))
   print(img1)
  
if st.button("About"):
  st.subheader("Student, Department of Computer Engineering")
st.subheader("Dhruv Sevak")
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Digital Image processing Experiment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
