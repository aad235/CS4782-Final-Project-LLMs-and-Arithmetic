import random 
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import tensorflow as tf
tokenizer = T5Tokenizer.from_pretrained('t5-small')
baseline = T5ForConditionalGeneration.from_pretrained('t5-small')
from sklearn.model_selection import train_test_split

"""
In this file you will find the steps needed to create the baseline addition dataset.
Use this data set to train the baseline t5-small model. You can specify how many training 
examples you want, up to how many digits you want, and what alpha value you want. The
alpha value determines the repititon level of the dataset.
"""


def generate_number(digit_length):
    """Generate a random number with the specified digit length."""
    if digit_length == 0:
        return 0
    return random.randint(10**(digit_length - 1), 10**digit_length - 1)

def generate_addition_examples(num_samples, k_digits, alpha):
    examples = []
    for _ in range(num_samples):
        digit_length = random.randint(1, k_digits) 
        num1 = generate_number(digit_length)
        num2 = generate_number(digit_length)
        problem = f"{num1}+{num2}"
        solution = num1 + num2
        examples.append((problem, str(solution)))
        # Add repetition with probability alpha
        for _ in range(k_digits - 1):
            if random.random() < alpha:
                num2 = num1
            else:
                num2 = generate_number(digit_length) 
            problem = f"{num1}+{num2}"
            solution = num1 + num2
            examples.append((problem, str(solution)))
            num1 = num2
    return examples

# Save the dataset to a file
def save_dataset_to_file(dataset, filename):
    with open(filename, "w") as file:
        for example in dataset:
            file.write(example[0] + "\t" + example[1] + "\n")

# Generate dataset
def generate_dataset(num_samples, k_digits, alpha):
    in_distribution_dataset = generate_addition_examples(num_samples, k_digits, alpha)
    out_of_distribution_dataset = generate_addition_examples(num_samples, k_digits + 1, alpha)
    return in_distribution_dataset, out_of_distribution_dataset

# Generate and save the dataset
def generate_and_save_dataset(num_samples, k_digits, alpha, filename):
    in_distribution_dataset, out_of_distribution_dataset = generate_dataset(num_samples, k_digits, alpha)
    save_dataset_to_file(in_distribution_dataset, filename + "_in_distribution.txt")
    save_dataset_to_file(out_of_distribution_dataset, filename + "_out_of_distribution.txt")

# Parameters
num_samples = 2000  # Number of samples
k_digits = 5        # Maximum number of digits
alpha = 0.1         # Repetition level

# Generate and save the dataset
generate_and_save_dataset(num_samples, k_digits, alpha, "t5_dataset")


def preprocess_function(examples):
    problems = examples['problem']
    solutions = examples['solution']

    # Format the problems into model inputs
    inputs = [f"add: {problem} =" for problem in problems]
    model_inputs = tokenizer(
        inputs,
        max_length=16,
        truncation=True,
        padding='max_length'
    )

    # Prepare labels for the model
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            solutions,
            max_length=64,
            truncation=True,
            padding='max_length'
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


