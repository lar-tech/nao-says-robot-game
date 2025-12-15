from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import soundfile as sf

# load model
model_path = "./models"   
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path,dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# read audio file
audio, sr = sf.read("test.wav")
inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to(model.device)
with torch.no_grad():
    ids = model.generate(**inputs)

text = processor.batch_decode(ids, skip_special_tokens=True)[0]
print(text)