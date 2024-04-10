from tensorflow.keras.models import model_from_json
import numpy as np
import streamlit as st
import librosa
json_file = open('D:/Imp Files/Jupyter nootbook/Model/CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("D:/Imp Files/Jupyter nootbook/Model/CNN_model_weights.weights.h5")
print("Loaded model from disk")
import pickle

with open('D:/Imp Files/Jupyter nootbook/Model/scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)
    
with open('D:/Imp Files/Jupyter nootbook/Model/encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

    
print("Done") 

import librosa
def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data,frame_length=2048,hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

def get_predict_feat(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    res=extract_features(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,2376))
    i_result = scaler2.transform(result)
    final_result=np.expand_dims(i_result, axis=2)
    
    return final_result

emotions1={1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',8:'Surprise'}
def prediction(path1):
    res=get_predict_feat(path1)
    predictions=loaded_model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    return y_pred[0][0]


st.title('Speech Emotion Recognition')
audio_file = st.file_uploader("Upload an audio file", type=['wav'])
if audio_file is not None:
    st.audio(audio_file)
    # Check the prediction
    model = prediction(audio_file)
    st.write('Predicted emotion:', model)
