import os
import torch
import evaluate
from datasets import load_dataset, Features, Audio, Value
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperFeatureExtractor, WhisperTokenizer

MODEL_NAME = "openai/whisper-small"
OUTPUT_DIR = "./whisper-salaman"
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Menggunakan device: {device}")

feature_schema = Features({
    "file_name": Audio(sampling_rate=16000),
    "text": Value("string"),
})

print("Memuat dataset train dan validation...")
dataset = load_dataset("csv", data_files={"train": "train.csv", "validation": "validation.csv"}, features=feature_schema)


feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="Indonesian", task="transcribe")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="Indonesian", task="transcribe")

def prepare_dataset(batch):
    audio = batch["file_name"]
    
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


print("\nMemproses dataset audio...")
dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=1)

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=1e-5,
    warmup_steps=5,
    max_steps=20,
    logging_steps=5,
    save_steps=10,
    eval_steps=10,
    fp16=True,
    gradient_checkpointing=True,
    do_eval=True,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nMemulai proses fine-tuning dengan validasi...")
trainer.train()

print(f"\nPelatihan selesai. Model disimpan di folder: {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
processor.save_pretrained(OUTPUT_DIR)