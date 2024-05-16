import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch

# Function to format numbers with markers
def format_number_with_markers(number, digits):
    """Formats the number with leading zeros and attaches markers to each digit according to its position."""
    number_str = f"{number:0{digits}d}"
    return ' '.join(f"{chr(64 + digits - i)} {digit}" for i, digit in enumerate(number_str))

# Generate formatted addition examples for testing with expected results also formatted
def generate_formatted_addition_example(num_digits):
    num1 = random.randint(10**(num_digits-1), 10**num_digits - 1)
    num2 = random.randint(10**(num_digits-1), 10**num_digits - 1)
    formatted_problem = f"{format_number_with_markers(num1, num_digits)} + {format_number_with_markers(num2, num_digits)}"
    solution = num1 + num2
    solution_digits = len(str(solution))
    formatted_solution = format_number_with_markers(solution, solution_digits)
    return (formatted_problem, formatted_solution)

# Evaluate model performance on formatted data for multiple digit numbers
def evaluate_formatted_model_performance(trained_model, trained_tokenizer, num_classes=30, examples_per_class=100):
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

            print(f"Problem: {problem} | Expected: {expected_result} | Predicted: {predicted_result}")

            if predicted_result == expected_result:
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
plt.plot(range(1, 31), accuracy_per_class, marker='o')
plt.xlabel('Number of Digits')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs. Number of Digits')
plt.xticks(range(1, 31))
plt.grid(True)
plt.show()
