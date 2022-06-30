import autokeras as ak
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="CREAM Magic",
    page_icon="🍨",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# models
# topic = load_model("keras/topic", custom_objects=ak.CUSTOM_OBJECTS)
regional = load_model("keras/regional", custom_objects=ak.CUSTOM_OBJECTS)
# conversion = load_model("keras/topic", custom_objects=ak.CUSTOM_OBJECTS)
reach = load_model("keras/reach", custom_objects=ak.CUSTOM_OBJECTS)
engagement = load_model("keras/engagement", custom_objects=ak.CUSTOM_OBJECTS)
antichurn = load_model("keras/antichurn", custom_objects=ak.CUSTOM_OBJECTS)

st.title("🍨 CREAM Magic")
# title = st.text_input('Title')
body = st.text_area(
    "Artikeltext", placeholder="Hier Artikeltext einfügen und Los drücken!", height=500
)


def scorer(badge, value):
    if value > 0.5:
        return badge


# magic
if st.button("Los"):
    text = np.array([body])
    regional_badge = scorer("Regionalität",regional.predict(tf.expand_dims(text, -1))[0][0].astype(float))
    reach_badge = scorer("Reach",reach.predict(tf.expand_dims(text, -1))[0][0].astype(float))
    engagement_badge = scorer("Engagement",engagement.predict(tf.expand_dims(text, -1))[0][0].astype(float))
    antichurn_badge = scorer("Anti-Churn",antichurn.predict(tf.expand_dims(text, -1))[0][0].astype(float))
    st.header(f"Klassifizierung: {regional_badge or ''} {reach_badge or ''} {engagement_badge or ''} {antichurn_badge or ''}")
    st.write(f"{body[0:500]}...")
