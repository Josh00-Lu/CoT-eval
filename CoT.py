import json
import re
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

import argparse

tokenizer = AutoTokenizer.from_pretrained("./chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm2-6b", trust_remote_code=True).half().cuda()
model = model.eval()

CoT_list = [
    {"question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."},
    {"question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."},
    {"question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."},
    {"question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?", "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."},
    {"question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?", "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."},
    {"question": "There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?", "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29."},
    {"question": "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?", "answer": "Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."},
    {"question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?", "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 = 8. The answer is 8."},
]

CoT_history = [(item['question'], item['answer']) for item in CoT_list]
CoT_history_label = [('question:' + item['question'], 'answer:' + item['answer']) for item in CoT_list]
CoT_string = "\n".join(f"question: {item['question']}\nanswer: {item['answer']}" for item in CoT_list) + "\n"

def read_jsonl(file_path):
    """Reads a .jsonl file and returns a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def write_jsonl(data_list, file_path):
    """Writes a list of dictionaries to a .jsonl file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data_list:
            file.write(json.dumps(item) + '\n')

def generate_cot_prompt(base_prompt, new_question):
    """Generates a Chain of Thought prompt by appending a new question to the base prompt."""
    return f"{base_prompt}\nquestion: {new_question}, answer: "

def query_model(prompt, history):
    """Simulates querying a model. Replace this function with actual model querying logic."""
    # Assuming the model returns an answer, this is a placeholder function.
    response, history = model.chat(tokenizer, prompt, history=history)
    return response

def main(cot_type):
    # cot_type = "history_label"
    # cot_type = "one_string"
    
    # File path to your test.jsonl file
    file_path = './grade-school-math/grade_school_math/data/test.jsonl'
    
    # Base CoT prompt as described
    base_prompt = CoT_string

    # Read questions from the jsonl file
    questions_and_answers = read_jsonl(file_path)
    
    output_data = []
    
    # Process each question
    for idx, question_and_answer in enumerate(tqdm(questions_and_answers)):
        
        question = question_and_answer['question']
        ground_truth = question_and_answer['answer']
        pure_ground_truth = re.search(r'#### (-?\d+)', ground_truth).group(1)
        
        if cot_type == "history":
            prompt = question
            response = query_model(prompt, CoT_history)
        elif cot_type == "history_label":
            prompt = question
            response = query_model(prompt, CoT_history_label)
        elif cot_type == "one_string":
            prompt = generate_cot_prompt(base_prompt, question)
            response = query_model(prompt, [])
        elif cot_type == "none":
            prompt = question
            response = query_model(prompt, [])
        else:
            raise NotImplementedError

        output_data.append({
            "pure_ground_truth": pure_ground_truth,
            "model_out": response
        })
        
        # Print the model's response
        # print(f"---Model Input({idx})---")
        # print(prompt)
        # print(f"---Model Outpu({idx})---")
        # print(response)
        
    write_jsonl(output_data, f"./{cot_type}.jsonl")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cot_type', type=str, default="history")
    
    args = parser.parse_args()
    
    main(args.cot_type)