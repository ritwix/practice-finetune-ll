#!/usr/bin/env python
# coding: utf-8

# In[2]:


from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np


# In[3]:


dataset = load_dataset('glue', 'cola')
metric = load_metric('glue', 'cola')


# In[4]:


tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast = True)
def preprocessor(examples):
    return tokenizer(examples['sentence'], truncation=True)
encoded_dataset = dataset.map(preprocessor, batched=True)


# In[5]:


# min(encoded_dataset["test"]['label'])


# In[6]:


model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels = 2)


# In[8]:


batch_size = 16
args = TrainingArguments(
    output_dir = 'roberta-finetuned-cola',
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    per_device_train_batch_size= batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate = 2e-5,
    num_train_epochs = 5,
    weight_decay = 0.01,
    load_best_model_at_end=True,
    metric_for_best_model='matthews_correlation',
)


# In[9]:


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis = 1)
    return metric.compute(predictions=predictions, references=labels)


# In[10]:


trainer = Trainer(
    model = model,
    args = args,
    train_dataset = encoded_dataset["train"],
    eval_dataset = encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics = compute_metrics
)


# In[ ]:


trainer.train()


# In[12]:


#def convert_test_label(examples):
#    examples["label"] = (examples["label"] + 1)/2
#    return examples

# test_encoded_dataset = encoded_dataset['test'].map(convert_test_label)
val_results = trainer.evaluate(encoded_dataset['validation'])
from pickle import dump
with open('val_results_roberta_cola.pkl', 'wb') as fout:
    dump(val_results, fout)

