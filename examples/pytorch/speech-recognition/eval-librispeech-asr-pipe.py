from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch
from jiwer import wer

# see reference results
# https://discuss.huggingface.co/t/wav2vec2-base-task-performance/14881/5

# librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")
librispeech_eval = load_dataset("librispeech_asr", "other", split="test")

model_path = 'facebook/wav2vec2-base-100h'
# model_path = 'facebook/wav2vec2-base-960h'

model = Wav2Vec2ForCTC.from_pretrained(model_path).to("cuda")
processor = Wav2Vec2Processor.from_pretrained(model_path)

def map_to_array(batch):
    # speech, _ = sf.read(batch["file"])
    # batch["speech"] = speech
    batch["speech"] = batch['audio']['array']
    return batch

librispeech_eval = librispeech_eval.map(map_to_array)

def map_to_pred(batch):
    input_values = processor(batch["speech"], return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch

result = librispeech_eval.map(map_to_pred, batched=True, batch_size=1, remove_columns=["speech"])

print("WER:", wer(result["text"], result["transcription"]))