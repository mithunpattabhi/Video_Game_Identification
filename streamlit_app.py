import streamlit as st
from PIL import Image
import numpy as np, json, os
import tensorflow as tf
from tensorflow import keras
import pandas as pd

st.set_page_config("Game Image Identifier", layout="centered")
MODEL_PATH = "Model/game_effnetb0.h5"
CLASS_JSON = "Model/class_names.json"
IMG_SIZE = 224

@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    classes = json.load(open(CLASS_JSON))
    return model, classes

model, classes = load_model()
st.title("🎮 Game Image Identifier")
uploaded = st.file_uploader("Upload a Video Game Screenshot from GTA5 or Indiana Jones or Tomb Raider or Spiderman", type=["png","jpg","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    st.image(img, use_column_width=True)
    x = np.array(img)
    x = preprocess_input = tf.keras.applications.efficientnet.preprocess_input(x)
    preds = model.predict(np.expand_dims(x,0))[0]
    idxs = preds.argsort()[::-1]
    top1 = classes[idxs[0]]
    st.markdown(f"## Predicted: **{top1}** — {preds[idxs[0]]*100:.2f}%")
    df = pd.DataFrame({"Game":[classes[i] for i in idxs], "Probability":[f"{preds[i]*100:.2f}%" for i in idxs]})
    st.table(df)
else:
    st.info("Upload a Video Game(from the 4 mentioned above) screenshot to predict.")

st.markdown("Developed by Mithun Pattabhi. Visit my Github for more projects: [https://github.com/mithunpattabhi]")