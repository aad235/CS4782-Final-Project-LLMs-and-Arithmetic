from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import random
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Define functions to generate data
def generate_random_labels(digits):
    letters = random.sample([chr(i) for i in range(65, 91)], digits)
    return letters

def format_number_with_specified_markers(number, digits, labels):
    number_str = f"{number:0{digits}d}"
    return ' '.join(f"{label} {digit}" for label, digit in zip(labels, number_str))

def generate_addition_example_with_variable_digits(min_digits, max_digits):
    digits1 = random.randint(min_digits, max_digits)
    digits2 = random.randint(min_digits, max_digits)
    num1 = random.randint(0, 10**digits1 - 1)
    num2 = random.randint(0, 10**digits2 - 1)
    max_digits_num1_num2 = max(digits1, digits2)
    labels = generate_random_labels(max_digits_num1_num2)
    formatted_num1 = format_number_with_specified_markers(num1, max_digits_num1_num2, labels)
    formatted_num2 = format_number_with_specified_markers(num2, max_digits_num1_num2, labels)
    problem = f"{formatted_num1} + {formatted_num2}"
    solution = num1 + num2
    solution_digits = max_digits_num1_num2 + (solution >= 10**max_digits_num1_num2)
    if solution_digits > max_digits_num1_num2:
        new_label = random.choice([chr(i) for i in range(65, 91) if chr(i) not in labels])
        labels.insert(0, new_label)
    solution_formatted = format_number_with_specified_markers(solution, solution_digits, labels)
    return (problem, solution_formatted)

def generate_addition_examples_with_variable_digits(num_samples, min_digits, max_digits):
    return [generate_addition_example_with_variable_digits(min_digits, max_digits) for _ in range(num_samples)]

def save_dataset_to_file(dataset, filename):
    with open(filename, "w") as file:
        for problem, solution in dataset:
            file.write(f"{problem} = {solution}\n")

def generate_and_save_datasets(num_samples, filename):
    in_distribution_dataset = generate_addition_examples_with_variable_digits(num_samples // 2, 1, 5)
    save_dataset_to_file(in_distribution_dataset, f"{filename}_in_distribution.txt")
    out_of_distribution_dataset = generate_addition_examples_with_variable_digits(num_samples // 2, 6, 10)
    save_dataset_to_file(out_of_distribution_dataset, f"{filename}_out_of_distribution.txt")

# Parameters
num_samples = 10000
filename = "t5_dataset"

# Generate and save the datasets
generate_and_save_datasets(num_samples, filename)

# Define preprocessing function
def preprocess_function(examples):
    problems = examples['problem']
    solutions = examples['solution']
    inputs = [f"add: {problem} =" for problem in problems]
    model_inputs = tokenizer(inputs, max_length=16, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(solutions, max_length=16, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load in-distribution dataset
def load_dataset_from_file(filename):
    dataset = []
    with open(filename, "r") as file:
        for line in file:
            problem, solution = line.strip().split(" = ")
            dataset.append({"problem": problem, "solution": solution})
    return dataset

train_dataset = load_dataset_from_file("t5_dataset_in_distribution.txt")

# Split dataset into training and evaluation sets
train_dataset, eval_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

# Convert datasets to Hugging Face datasets and tokenize
train_dataset_hf = Dataset.from_dict({"problem": [x['problem'] for x in train_dataset], "solution": [x['solution'] for x in train_dataset]})
eval_dataset_hf = Dataset.from_dict({"problem": [x['problem'] for x in eval_dataset], "solution": [x['solution'] for x in eval_dataset]})
tokenized_train = train_dataset_hf.map(preprocess_function, batched=True, num_proc=16)
tokenized_eval = eval_dataset_hf.map(preprocess_function, batched=True, num_proc=16)

# Load model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=200,
    weight_decay=0.01
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval
)

# Train model
trainer.train()

# Save the final model
trainer.save_model('./results/final_model')

# Load the model from the final checkpoint
model_path = './results/final_model'
trained_model = T5ForConditionalGeneration.from_pretrained(model_path)
trained_tokenizer = T5Tokenizer.from_pretrained("t5-small")

def add_numbers(expression, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = f"add: {expression} ="
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=16, truncation=True)
    input_ids = input_ids.to(device)
    outputs = model.generate(input_ids)
    predicted_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_result

example_expression = "B 1 A 1 + B 2 A 9"
result = add_numbers(example_expression, model, trained_tokenizer)
print(f"{example_expression} = {result}")

example_expression = "B 6 + A 3 + B 4 A 6"
result = add_numbers(example_expression, model, trained_tokenizer)
print(f"{example_expression} = {result}")
