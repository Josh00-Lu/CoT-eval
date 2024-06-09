import torch
import json
import re
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--cot_type', type=str, default="history")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--ft_path', type=str)
args = parser.parse_args()

FT_PATH = args.ft_path
BASE_PATH = "./chatglm2-6b"
os.system(f"cp {BASE_PATH}/configuration_chatglm.py {FT_PATH}/configuration_chatglm.py")
os.system(f"cp {BASE_PATH}/modeling_chatglm.py {FT_PATH}/modeling_chatglm.py")
os.system(f"cp {BASE_PATH}/quantization.py {FT_PATH}/quantization.py")
os.system(f"cp {BASE_PATH}/tokenization_chatglm.py {FT_PATH}/tokenization_chatglm.py")

path_parts = FT_PATH.split("/")
specific_parts = path_parts[-2:]  # This selects the last two components of the path
specific_parts[0] = "-".join(specific_parts[0].split("-")[-2:])
exp = "-".join(specific_parts)

######### Modifiy Here #########
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(FT_PATH, trust_remote_code=True).half().cuda()
model = model.eval()
######### Modifiy Here #########

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
    return f"{base_prompt}\nquestion: {new_question}\nanswer: "

def get_dataloader(file_path, batch_size=8):
    dataset = read_jsonl(file_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

def build_prompt(question, cot_type):
    if cot_type == "history":
        prompt = tokenizer.build_prompt(question, CoT_history)
    elif cot_type == "history_label":
        prompt = tokenizer.build_prompt(question, CoT_history_label)
    elif cot_type == "one_string":
        prompt = generate_cot_prompt(CoT_string, question)
        prompt = tokenizer.build_prompt(prompt, [])
    elif cot_type == "none":
        prompt = tokenizer.build_prompt(question, [])
    else:
        raise NotImplementedError
    
    return prompt

def main():
    file_path = './grade-school-math/grade_school_math/data/test.jsonl'
    dataloader = get_dataloader(file_path=file_path, batch_size=args.batch_size)
    output_data = []
    
    for batch in tqdm(dataloader):
        questions = batch["question"]
        ground_truths = batch['answer']
        pure_ground_truth = [re.search(r'#### (-?\d+)', ground_truth).group(1) for ground_truth in ground_truths]
        queries = [build_prompt(question, args.cot_type) for question in questions]
        
        inputs = tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=2048).to('cuda')
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
        
        for idx in range(len(outputs)):
            output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
            response = tokenizer.decode(output)
            output_data.append({
                "pure_ground_truth": pure_ground_truth[idx],
                "model_out": response
            })
            
    write_jsonl(output_data, f"./{exp}-{args.cot_type}.jsonl")

if __name__ == "__main__":
    main()