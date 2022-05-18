#!/usr/bin/env python
# coding: utf-8

# In[2]:


from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np


# In[3]:


dataset = load_dataset('glue', 'sst2')
metric = load_metric('glue', 'sst2')


# In[4]:


tokenizer = AutoTokenizer.from_pretrained('./roberta-finetuned-sst2/checkpoint-4212', use_fast = True)
def preprocessor(examples):
    return tokenizer(examples['sentence'], truncation=True)
encoded_dataset = dataset.map(preprocessor, batched=True)


# In[5]:


min(encoded_dataset["test"]['label'])


# In[6]:


model = AutoModelForSequenceClassification.from_pretrained('./roberta-finetuned-sst2/checkpoint-4212', num_labels = 2)


# In[8]:


batch_size = 16
args = TrainingArguments(
    output_dir = 'roberta-finetuned-sst2',
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    per_device_train_batch_size= batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate = 2e-5,
    num_train_epochs = 5,
    weight_decay = 0.01,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
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


# trainer.train()


# In[12]:


#def convert_test_label(examples):
#    examples["label"] = (examples["label"] + 1)/2
#    return examples

# test_encoded_dataset = encoded_dataset['test'].map(convert_test_label)
print(trainer.evaluate(encoded_dataset['validation']))

