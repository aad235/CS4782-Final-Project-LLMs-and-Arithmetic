from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import tensorflow as tf
import random
tokenizer = T5Tokenizer.from_pretrained('t5-small')
baseline = T5ForConditionalGeneration.from_pretrained('t5-small')
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split


"""
We trained and saved our models in google colab so when generating plots we
pulled from google drive


from google.colab import drive
drive.mount('/content/drive')


#Grab model from google drive
model_path = 'if reimplementing replace with location'  #this is where we saved our model in google drive
trained_model = T5ForConditionalGeneration.from_pretrained(model_path)
trained_tokenizer = T5Tokenizer.from_pretrained(model_path)
"""

# Example usage of the trained model
def add_numbers(expression, model, tokenizer):
    input_text = f"add: {expression} ="
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=16, truncation=True)

    outputs = model.generate(input_ids)
    predicted_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_result

example_expression = "777+777"
result = add_numbers(example_expression, trained_model, trained_tokenizer)
print(f"{example_expression} = {result}")

import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Generate a single addition example
def generate_addition_example(num_digits):
    num1 = random.randint(10**(num_digits-1), 10**num_digits - 1)
    num2 = random.randint(10**(num_digits-1), 10**num_digits - 1)
    return f"{num1}+{num2}"

# Generate multiple examples for each digit class
def generate_examples_per_class(num_digits, num_examples=100):
    return [generate_addition_example(num_digits) for _ in range(num_examples)]

# Evaluate the model's performance across multiple classes
def evaluate_model_performance(trained_model, trained_tokenizer, num_classes=30, examples_per_class=100):
    accuracy_per_class = []

    for num_digits in range(1, num_classes + 1):
        correct_predictions = 0
        examples = generate_examples_per_class(num_digits, examples_per_class)

        for example in tqdm(examples, desc=f"Class {num_digits}"):
            predicted_result = add_numbers(example, trained_model, trained_tokenizer)
            expected_result = str(eval(example))

            if predicted_result == expected_result:
                correct_predictions += 1

        accuracy = correct_predictions / examples_per_class
        accuracy_per_class.append(accuracy)

    return accuracy_per_class

# Function to add numbers using the trained model
def add_numbers(expression, model, tokenizer):
    input_text = f"add: {expression} ="
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=16, truncation=True)

    outputs = model.generate(input_ids)
    predicted_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_result

accuracy_per_class = evaluate_model_performance(trained_model, trained_tokenizer, num_classes=30)

# Plotting the accuracy vs. class
plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), accuracy_per_class, marker='o')
plt.xlabel('Class (Number of Digits)')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs Class (Number of Digits)')
plt.xticks(range(1, 31))
plt.grid(True)
plt.show()

