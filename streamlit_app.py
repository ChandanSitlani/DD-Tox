import tensorflow as tf
import numpy as np
import json
import streamlit as st

st.set_page_config("Drug Toxicity Classifier")
st.title("Drug Toxicity Classifier")
model=tf.keras.models.load_model("model_clintox-0.9130_us.h5")
with open("vocab_tox.json", "r") as f:
    voc=json.load(f)
    

def encode(row):
  r2=row
  for i in range(len(row)):
    try:
        r2[i]=voc[row[i]]
    except:
        r2[i]=1
  return r2
def one_hot(x):
  x3d=np.zeros((x.shape[0], x.shape[1], len(voc)+1))
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x3d[i][j][x[i][j]]=1
  
  return x3d
def predict(d):
  l=list(d)
  e=encode(l)
  e=np.expand_dims(e, axis=0)
  p=x_train=tf.keras.preprocessing.sequence.pad_sequences(e, 210)
  oh=one_hot(p)
  pred=model.predict(oh)
  p=np.argmax(pred, axis=-1)
  return p, pred

drug=st.text_input("Enter Drug Smiles")

if st.button("Check"):
    p, pred=predict(drug)
    pr=round(float(pred[0][1])*100, 2)
    sts="Safe" if pr>60 else "Unsafe"
    st.metric("Safety", sts, delta=pr, delta_color='normal')