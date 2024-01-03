import streamlit as st
import pandas as pd  
from PIL import Image
import pdfplumber
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="File Chatbot")

@st.cache_resource
def initialize_model():
    text_model = genai.GenerativeModel('gemini-pro')   
    vision_model = genai.GenerativeModel('gemini-pro-vision')
    return text_model, vision_model

text_model, vision_model = initialize_model()

model = initialize_model()

def get_response(file,input_text,prompt):

    if file.type == "image/jpeg": 
        if file is not None:
            # Read the file into bytes
            bytes_data = file.getvalue()

            image_parts = [
            {
                "mime_type": file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        # return image_parts 
        # image_data = [file.read()]   
        return vision_model.generate_content([input_text, image_parts,prompt]).text

    elif file.type == "text/csv":
        df = pd.read_csv(file)  
        return text_model.generate_content([input_text, str(df),prompt]).text

    elif file.suffix == ".pdf":
        pdf = pdfplumber.open(file)
        text = "\n\n".join([page.extract_text() for page in pdf.pages])  
        return text_model.generate_content([input_text, text,prompt]).text
        
    elif file.type == "text/plain":
        raw_text = str(file.read(),"utf-8")  
        return text_model.generate_content([input_text, raw_text,prompt]).text

st.header("File Chatbot")

file = st.file_uploader("Choose file", type=["jpg","jpeg","png","csv","pdf","txt"])  
prompt = """
               You are an expert in understanding csv.
               You will receive input images as csv &
               you will have to answer questions based on the input image
               your answer should strictly be related to csv file uploaded
               if you are asked with complex query then go through the csv carefully and answer correctly
               """
if file:

    input_text = st.text_input("Ask a question about the file:", key="input")

    if st.button("Get Response"):
        response = get_response(file, input_text,prompt)  
        st.write("Response:", response)