import io
import torch
import ffmpeg
import librosa
import numpy as np
import soundfile as sf
from flask import Flask,request,render_template
from flask_cors import cross_origin
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

application = Flask(__name__)


print("Loading dutch_processor model...")
dutch_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-dutch")
print("Loading dutch_pretrained model...")
dutch_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-dutch")
print("Loading english_processor model...")
english_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
print("Loading english_pretrained model...")
english_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
print("Loading french_processor model...")
french_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
print("Loading french_pretrained model...")
french_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
print("All models loaded Ready to transcribe.")



def dutch_transcribe(file_to_transcribe):
    input_values = dutch_processor(file_to_transcribe, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
            print("Starting inference from dutch model...")
            logits = dutch_model(input_values).logits
            print("Inference ended..")
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = dutch_processor.decode(predicted_ids[0])
    print("sending response...")
    return transcription


def english_transcribe(file_to_transcribe):
    input_values = english_processor(file_to_transcribe, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
            print("Starting inference from english model...")
            logits = english_model(input_values).logits
            print("Inference ended..")
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = english_processor.decode(predicted_ids[0])
    print("sending response...")
    return transcription

def french_transcribe(file_to_transcribe):
    input_values = french_processor(file_to_transcribe, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
            print("Starting inference from french model...")
            logits = french_model(input_values).logits
            print("Inference ended..")
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = french_processor.decode(predicted_ids[0])
    print("sending response...")
    return transcription


api_v2_cors_config = {
  "origins": ["*"],
  "methods": ["OPTIONS", "GET", "POST"],
  "allow_headers": ["Content-Type"]
}

@application.route('/file_to_transcribe',methods = ['POST','GET'])
@cross_origin(**api_v2_cors_config)
def result():
    try:
        outputs = {}
        data = []
        if request.method=='POST':
            print("Request Recieved...")

            language = request.files['language'].read().decode('utf-8')
            fileType = request.files['audioFile'].filename

            if fileType[-3:] == 'mp3' or fileType[-3:] =='m4a' or fileType[-3:] =='wma':
                with open("audio_file.mp3","wb") as f:
                    f.write(request.files['audioFile'].read())
                data,samplerate = librosa.load("audio_file.mp3",sr=16000,mono=True)

            elif fileType[-3:] == 'wav' :
                data, samplerate = librosa.load(io.BytesIO(request.files['audioFile'].read()), sr=16000, mono=True)

            else:
                raise Exception("File Type Not Supported!")
            if language=='english':
                outputs['transcript'] = english_transcribe(data)
            elif language=='dutch':
                outputs['transcript'] = dutch_transcribe(data)
            elif language=='french':
                outputs['transcript'] = french_transcribe(data)
                
        return outputs
    except Exception as e:
        return {
            'output': '',
            'error': str(e)
        }



if __name__ =='__main__':
    from waitress import serve
    serve(application, host="0.0.0.0", port=3200)