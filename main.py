import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo", low_cpu_mem_usage=True).to(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")

data, _ = sf.read("test2.mp3")

inputs = processor.feature_extractor(data, return_tensors="pt", sampling_rate=16_000).input_features.to(device)
predicted_ids = model.generate(inputs, max_length=540_000)
text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)[0]

print("------------------------------Transcribed Text----------------------")
print(text)
print("----------------------------------------------------------------")
