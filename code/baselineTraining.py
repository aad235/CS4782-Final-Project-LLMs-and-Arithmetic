
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import tensorflow as tf
import random
tokenizer = T5Tokenizer.from_pretrained('t5-small')
baseline = T5ForConditionalGeneration.from_pretrained('t5-small')
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset


# Load in-distribution dataset
def load_dataset_from_file(filename):
    dataset = []
    with open(filename, "r") as file:
        for line in file:
            problem, solution = line.strip().split("\t")
            dataset.append((problem, solution))
    return dataset
train_dataset = load_dataset_from_file("t5_dataset_in_distribution.txt")

#print(train_dataset)

#Split data set for training
train_dataset, eval_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

# Convert to Dataset objects
train_dataset_hf = Dataset.from_dict({"problem": [x[0] for x in train_dataset], "solution": [x[1] for x in train_dataset]})
eval_dataset_hf = Dataset.from_dict({"problem": [x[0] for x in eval_dataset], "solution": [x[1] for x in eval_dataset]})

# Tokenize the datasets
tokenized_train = train_dataset_hf.map(preprocess_function, batched=True, num_proc=16)
tokenized_eval = eval_dataset_hf.map(preprocess_function, batched=True, num_proc=16)


#Define model

model = T5ForConditionalGeneration.from_pretrained("t5-small")
# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Directory to store model checkpoints and results
    evaluation_strategy="epoch",     # Evaluate model at the end of each epoch
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,   # Adjust based on available GPU memory
    per_device_eval_batch_size=16,
    num_train_epochs=200,              # Number of training epochs
    weight_decay=0.01                # Weight decay for regularization
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval  # Use the tokenized evaluation dataset
)

#Train and save the model
trainer.train()
trainer.save_model('./results/final_model')