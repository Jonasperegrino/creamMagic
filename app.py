import autokeras as ak
import numpy as np
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
topic = load_model("keras/topic", custom_objects=ak.CUSTOM_OBJECTS)
regional = load_model("keras/regio", custom_objects=ak.CUSTOM_OBJECTS)
# conversion = load_model("keras/topic", custom_objects=ak.CUSTOM_OBJECTS)
reach = load_model("keras/reach", custom_objects=ak.CUSTOM_OBJECTS)
engagement = load_model("keras/engagement", custom_objects=ak.CUSTOM_OBJECTS)
antichurn = load_model("keras/antichurn", custom_objects=ak.CUSTOM_OBJECTS)

topics = [
    "gesellschaft",
    "gesellschaft/bildung",
    "gesellschaft/familie",
    "gesellschaft/mensch",
    "gesellschaft/religion",
    "gesundheit",
    "gesundheit/krankheit",
    "heimat",
    "heimat/bodensee",
    "heimat/heimatliebe",
    "heimat/mensch",
    "heimat/schweiz",
    "heimat/stadtentwicklung",
    "heimat/veranstaltung",
    "heimat/verein",
    "heimat/wein",
    "kultur",
    "kultur/film",
    "leben",
    "leben/alltagshilfe",
    "leben/freizeit",
    "leben/grundversorgung",
    "leben/wohnen",
    "natur",
    "natur/tiere",
    "natur/umwelt",
    "natur/wetter",
    "politik",
    "politik/ausland",
    "politik/bund",
    "politik/land",
    "politik/lokal",
    "politik/lokal/buergermeisterwahl",
    "sicherheit",
    "sicherheit/blaulicht",
    "sicherheit/justiz",
    "sicherheit/kriminalitaet",
    "sport",
    "sport/eishockey",
    "sport/fussball",
    "sport/fussball/wuerzburger-kickers",
    "verkehr",
    "verkehr/autoverkehr",
    "verkehr/flugverkehr",
    "verkehr/oepv",
    "wirtschaft",
    "wirtschaft/arbeit",
    "wirtschaft/branchen",
    "wirtschaft/branchen/einzelhandel",
    "wirtschaft/branchen/gastronomie",
    "wirtschaft/branchen/landwirtschaft",
    "wirtschaft/branchen/tourismus",
    "wirtschaft/energie",
    "wirtschaft/unternehmen",
    "wirtschaft/verbraucher",
    "wissen",
    "wissen/geschichte",
    "wissen/wissenschaft",
]


st.title("🍨 CREAM Magic")
# title = st.text_input('Title')
body = st.text_area(
    "Artikeltext", placeholder="Hier Artikeltext einfügen und Los drücken!"
)


def scorer(badge, threshold, prediction):
    value = prediction[0][0].astype(float)
    if value > threshold:
        return badge
    elif badge == "Regional":
        return f"Nicht {badge}"


# magic
if st.button("Los"):
    text = tf.expand_dims(np.array([body]), -1)

    topic_prob = topic.predict(text)
    topic_badge = topics[np.argmax(topic_prob)]

    regional_badge = scorer("Regional", 0.7, regional.predict(text))
    reach_badge = scorer("Reach", 0.3, reach.predict(text))
    engagement_badge = scorer(
        "Engagement",
        0.3,
        engagement.predict(text),
    )
    antichurn_badge = scorer(
        "Anti-Churn",
        0.3,
        antichurn.predict(text),
    )
    st.header(f"Thema: {topic_badge.title()}")
    st.header(f"{regional_badge or ''}")
    st.header(
        f"Stärke: {reach_badge or ''} {engagement_badge or ''} {antichurn_badge or ''}"
    )
    st.write(f"{body[0:500]}...")
