from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

dataset = load_dataset('glue', 'sst2')
metric = load_metric('glue', 'sst2')

tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast = True)

def preprocessor(examples):
    return tokenizer(examples['sentence'], truncation=True)

encoded_dataset = dataset.map(preprocessor, batched=True)


model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels = 2)
batch_size = 16
args = TrainingArguments(
    output_dir = 'roberta-finetuned-sst2',
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    per_device_train_batch_size= batch_size,
    per_device_eval_batch_size= batch_size*4,
    num_train_epochs = 3,
    weight_decay = 0.01,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis = 1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = encoded_dataset["train"],
    eval_dataset = encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics = compute_metrics
)

def preprocess_test_dataset(examples):
    examples["label"] = (examples["label"] + 1)/2
    return examples

test_encoded_dataset = encoded_dataset['test'].map(preprocess_test_dataset)
print(trainer.evaluate(test_encoded_dataset))
# print((encoded_dataset['test']['label']+1)/2)
# print(encoded_dataset['train']['label'][0:10])
# trainer.train()