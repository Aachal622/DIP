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
        Addition and Substraction on Image
         """
         )
url = st.file_uploader("Please upload image", type=("jpg", "png"))

import cv2
from  PIL import Image, ImageOps
def import_and_predict(image):

  img1=cv2.imread(url,1)
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

if url is None:
  st.text("Please upload an Image file")
else:
  file_bytes = np.asarray(bytearray(url.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  st.image(url,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Translation 50 in X-direction"):
   img1=cv2.imread(url,1)
   img2=np.ones(img1.shape, dtype="uint8")*100
   img=img1+img2
   print(img)

if st.button("Translation 150 in Y-direction"):
   img1=cv2.imread(url,1)
   img2=np.ones(img1.shape, dtype="uint8")*100
   img=img1-img2
   print(img)
  
if st.button("About"):
  st.header(" Aachal Kala")
  st.subheader("Student, Department of Computer Engineering")
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
