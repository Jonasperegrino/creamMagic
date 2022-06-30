import streamlit as st
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="CREAM Magic",
    page_icon="🍨",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# models
topic = load_model("keras/topic", custom_objects=ak.CUSTOM_OBJECTS)
regional = load_model("keras/regional", custom_objects=ak.CUSTOM_OBJECTS)
# conversion = load_model("keras/topic", custom_objects=ak.CUSTOM_OBJECTS)
reach = load_model("keras/reach", custom_objects=ak.CUSTOM_OBJECTS)
engagement = load_model("keras/engagement", custom_objects=ak.CUSTOM_OBJECTS)
antichurn = load_model("keras/antichurn", custom_objects=ak.CUSTOM_OBJECTS)

st.title("🍨 CREAM Magic")
# title = st.text_input('Title')
body = st.text_area("Artikeltext",height=500)

# magic
if st.button("Los"):
    text = np.array([body])
    st.header(f"Regionalität: {int(round(regional.predict(tf.expand_dims(text, -1))[0][0].astype(float),2)*100)}%")
    st.header(f"Reach: {int(round(reach.predict(tf.expand_dims(text, -1))[0][0].astype(float),2)*100)}%")
    st.header(f"Engagement: {int(round(engagement.predict(tf.expand_dims(text, -1))[0][0].astype(float),2)*100)}%")
    st.header(f"Antichurn: {int(round(antichurn.predict(tf.expand_dims(text, -1))[0][0].astype(float),2)*100)}%")
else:
    st.write("Artikeltext oben einfügen und Los drücken!")
