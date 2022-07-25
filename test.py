import torch
from datasets import load_dataset, load_metric
import re
import torchaudio
from train import PROCESSOR, MODEL


wer = load_metric("wer")

common_voice_test_transcription = load_dataset("common_voice", "hi", data_dir="./cv-corpus-6.1-2020-12-11", split="test")

test_dataset = load_dataset("common_voice", "hi", split="test")

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“]'  # TODO: adapt this list to include all special characters you removed from the data
resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
	batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
	speech_array, sampling_rate = torchaudio.load(batch["path"])
	batch["speech"] = resampler(speech_array).squeeze().numpy()
	return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def evaluate(batch):
    inputs = PROCESSOR(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = MODEL(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
    
    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = PROCESSOR.batch_decode(pred_ids)
    return batch

result = test_dataset.map(evaluate, batched=True, batch_size=8)

print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))