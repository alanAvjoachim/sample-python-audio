import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import soundfile
import librosa

app = Flask(__name__)
CORS(app)

ureaModel = pickle.load(open('pickles/urea.pkl','rb'))
mopModel = pickle.load(open('pickles/mop.pkl','rb'))
tspModel = pickle.load(open('pickles/tsp.pkl','rb'))
audioEmotionModel = pickle.load(open('pickles/audio-emotion.pkl','rb'))

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

@app.route("/")
def index():
    return '<h1>Hellow</h1>'

# ROUTES
@app.route('/audio-emotion', methods=['POST'])
def audioEmotionPrediction():
    
    data=request.get_json(force=True)
    
    feature=extract_feature(data['data'], mfcc=True, chroma=True, mel=True)
    feature=feature.reshape(1,-1)

    prediction = audioEmotionModel.predict(feature)
    print(prediction[0])
    return jsonify(prediction[0])

if __name__ == "__main__":
  app.run()