import json
import re
import argparse


def calculate_accuracy(file_path):
    total = 0
    correct = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse the JSON line
            data = json.loads(line)
            # Extract the ground truth and model output
            ground_truth = data['pure_ground_truth'].strip()
            model_output = data['model_out'].strip()
            # Extract the last integer (positive or negative) from the model output as the answer
            numbers = re.findall(r'-?\d+', model_output)
            if numbers:
                model_answer = numbers[-1]
            else:
                model_answer = ""
            # Increment total and check if the answers match
            total += 1
            if model_answer == ground_truth:
                correct += 1
    # Calculate and return the accuracy
    return correct / total if total > 0 else 0

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
args = parser.parse_args()

# Example usage:
accuracy = calculate_accuracy(args.path)
print(f"The accuracy is: {accuracy:.2%}")