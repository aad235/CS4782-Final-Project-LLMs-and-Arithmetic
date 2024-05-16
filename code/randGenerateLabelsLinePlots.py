import random
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define the maximum number of digits to be evaluated
num_digits_max = 25

def generate_random_labels(digits):
    """Generates a random set of unique labels for the given number of digits."""
    letters = random.sample([chr(i) for i in range(65, 91)], digits)
    return letters

def format_number_with_specified_markers(number, digits, labels):
    """Formats the number with leading zeros and attaches specified markers to each digit."""
    number_str = f"{number:0{digits}d}"
    return ' '.join(f"{label} {digit}" for label, digit in zip(labels, number_str))

# Generate formatted addition examples for testing with expected results also formatted
def generate_formatted_addition_example(num_digits):
    num1 = random.randint(10**(num_digits-1), 10**num_digits - 1)
    num2 = random.randint(10**(num_digits-1), 10**num_digits - 1)

    labels = generate_random_labels(num_digits)

    formatted_problem = f"{format_number_with_specified_markers(num1, num_digits, labels)} + {format_number_with_specified_markers(num2, num_digits, labels)}"

    solution = num1 + num2
    solution_digits = len(str(solution))

    if solution_digits > num_digits:
        # Add one additional random label if the solution length increases
        possible_labels = [chr(i) for i in range(65, 91) if chr(i) not in labels]
        if possible_labels:
            new_label = random.choice(possible_labels)
            labels.insert(0, new_label)  # Insert the new label at the beginning

    formatted_solution = format_number_with_specified_markers(solution, solution_digits, labels)

    return (formatted_problem, formatted_solution)

def extract_digits(formatted_string):
    """Extracts and returns only the digits from a formatted string."""
    return ''.join(filter(str.isdigit, formatted_string))

# Evaluate model performance on formatted data for multiple digit numbers
def evaluate_formatted_model_performance(trained_model, trained_tokenizer, num_classes=num_digits_max, examples_per_class=100):
    accuracy_per_class = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    print("Testing individual examples...")

    for num_digits in range(1, num_classes + 1):
        correct_predictions = 0
        examples = [generate_formatted_addition_example(num_digits) for _ in range(examples_per_class)]

        for problem, expected_result in tqdm(examples, desc=f"Evaluating {num_digits}-digit examples"):
            input_text = f"add: {problem} ="
            input_ids = trained_tokenizer.encode(input_text, return_tensors="pt", max_length=64, truncation=True).to(device)
            outputs = trained_model.generate(input_ids)
            predicted_result = trained_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract digits for comparison
            expected_digits = extract_digits(expected_result)
            predicted_digits = extract_digits(predicted_result)

            if predicted_digits == expected_digits:
                correct_predictions += 1

        accuracy = correct_predictions / examples_per_class
        accuracy_per_class.append(accuracy)

    return accuracy_per_class

# Load your model and tokenizer from a specific path
model_path = '/content/drive/My Drive/your_folder_name'
trained_model = T5ForConditionalGeneration.from_pretrained(model_path)
trained_tokenizer = T5Tokenizer.from_pretrained(model_path)

# Perform the evaluation for all classes
accuracy_per_class = evaluate_formatted_model_performance(trained_model, trained_tokenizer)

# Plotting the accuracy vs. number of digits
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_digits_max + 1), accuracy_per_class, marker='o')
plt.xlabel('Number of Digits')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs. Number of Digits')
plt.xticks(range(1, num_digits_max + 1))
plt.grid(True)
plt.show()
