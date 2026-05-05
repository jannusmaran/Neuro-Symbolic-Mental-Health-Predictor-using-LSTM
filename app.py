import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

st.set_page_config(page_title="Mental Health Predictor", layout="centered")


@st.cache_resource
def load_files():
    model = tf.keras.models.load_model("emotion_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return model, tokenizer, le

model, tokenizer, le = load_files()


st.title(" Mental Health Predictor")
st.write("Analyze your text and detect emotions")

text = st.text_area("Enter your text")


if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        # Preprocess
        text = text.lower()

        seq = tokenizer.texts_to_sequences([text])
        pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=40)

        # Prediction
        pred = model.predict(pad)
        pred_index = np.argmax(pred, axis=1)[0]

        emotion = le.inverse_transform([pred_index])[0]

      
        st.subheader("Prediction")
        st.success(f"Emotion: {emotion}")

    
        st.subheader("Prediction Probabilities")

        prob_dict = {
            le.classes_[i]: float(pred[0][i])
            for i in range(len(le.classes_))
        }

        st.bar_chart(prob_dict)