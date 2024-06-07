## Simple Script for LLM Math Evaluation

### Configuration Path
In `CoT.py`:
```python
######### Modifiy Here #########
tokenizer = AutoTokenizer.from_pretrained("./chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm2-6b", trust_remote_code=True).half().cuda()
model = model.eval()
######### Modifiy Here #########
......
######### Modifiy Here #########
def query_model(prompt, history):
    """Simulates querying a model. Replace this function with actual model querying logic."""
    # Assuming the model returns an answer, this is a placeholder function.
    response, history = model.chat(tokenizer, prompt, history=history)
    return response
######### Modifiy Here #########
```

### CoT-test
```shell
python CoT.py --cot_type "history" 
python CoT.py --cot_type "none" ## ablation CoT
```
The above command will generate `history.jsonl` and `none.jsonl`, respectively.

### Generate results from `jsonl`
```shell
python eval.py --path <e.g. history.jsonl>
python eval.py --path <e.g. none.jsonl>
```

## Support Dataset
- [x]  GSM8K
