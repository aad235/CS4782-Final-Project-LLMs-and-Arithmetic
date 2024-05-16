from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import torch
from datasets import Dataset

# Load the dataset from file and split into training and evaluation sets
def load_dataset_from_file(filename):
    dataset = []
    with open(filename, "r") as file:
        for line in file:
            problem, solution = line.strip().split(" = ")
            dataset.append({"problem": problem, "solution": solution})
    return dataset

train_dataset = load_dataset_from_file("t5_dataset_in_distribution.txt")
train_dataset, eval_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

train_dataset_hf = Dataset.from_dict({"problem": [x['problem'] for x in train_dataset], "solution": [x['solution'] for x in train_dataset]})
eval_dataset_hf = Dataset.from_dict({"problem": [x['problem'] for x in eval_dataset], "solution": [x['solution'] for x in eval_dataset]})

# Map the dataset to tokenized inputs using the tokenizer
def preprocess_function(examples):
    inputs = [f"add: {problem} =" for problem in examples['problem']]
    model_inputs = tokenizer(inputs, max_length=16, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['solution'], max_length=16, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset_hf.map(preprocess_function, batched=True, num_proc=16)
tokenized_eval = eval_dataset_hf.map(preprocess_function, batched=True, num_proc=16)

# Set up and configure the T5 model for training
model = T5ForConditionalGeneration.from_pretrained("t5-small")
training_args = TrainingArguments(
    output_dir='./results',          # Directory to store model checkpoints and results
    evaluation_strategy="epoch",     # Evaluate model at the end of each epoch
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,  # Adjust based on available GPU memory
    per_device_eval_batch_size=16,
    num_train_epochs=200,            # Number of training epochs
    weight_decay=0.01                # Weight decay for regularization
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval
)

# Train the model and save the final model
trainer.train()
trainer.save_model('./results/final_model')
