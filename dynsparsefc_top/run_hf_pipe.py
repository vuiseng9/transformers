import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import ActivationPruneLinear
import time
import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def change_forward_of_ActivationPruneLinear(model, to_sparse=True):
    is_dynamic_sparse = getattr(model.config, 'dynamic_sparse', False)
    if is_dynamic_sparse is False:
        raise ValueError(f"model has not be configured for sparse path")
    
    for n, m in model.named_modules():
        if isinstance(m, ActivationPruneLinear):
            if to_sparse is True:
                m.to_sparse_forward()
            else:
                m.to_dense_forward()

    converted = 0
    for n, m in model.named_modules():
        if isinstance(m, ActivationPruneLinear):
            if m.sparse_execution is True:
                converted += 1

    print(f"Intent to sparse:{to_sparse}, Total #sparse forward now: {converted}")
    return None

def set_config_of_ActivationPruneLinear(model, scap_config):
    is_dynamic_sparse = getattr(model.config, 'dynamic_sparse', False)
    if is_dynamic_sparse is False:
        raise ValueError(f"model has not be configured for sparse path")
    
    if not isinstance(scap_config, dict):
        raise TypeError("scap_config must be a dict")

    configured = 0
    for n, m in model.named_modules():
        if isinstance(m, ActivationPruneLinear):
            if n in scap_config['pre']:
                print(f"Configuring {n} ...")
                configured += 1
                m.target_sparsity = scap_config["pre"][n]["target_sparsity"]
                m.threshold = scap_config["pre"][n]["threshold"]
                m.zero_point = scap_config["pre"][n]["zero_point"]

            else:
                raise ValueError("unsupported mode. wrong matching linear layers")
    return None



# Main
gen_len=32
model_id = "/data1/vchua/run/scap_model/Llama-2-7b-hf"

scap_json = "./sparse_config_llama2-7b_up+gate0.35,down0.5.json"
scap_config = read_json_file(scap_json)


config = AutoConfig.from_pretrained(model_id)
config.dynamic_sparse = True
model = AutoModelForCausalLM.from_pretrained(model_id, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Below are the conventional way of loading
# model = AutoModelForCausalLM.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "what is the meaning of life?"
inputs = tokenizer(text, return_tensors="pt")

# model.state_dict()['model.layers.27.mlp.down_proj.weight'].shape (4096,11008)
# -----------------------
print("-"*100, flush=True)
print("\n\n++ sparse")
set_config_of_ActivationPruneLinear(model, scap_config)
change_forward_of_ActivationPruneLinear(model, to_sparse=True)
# model.state_dict()['model.layers.27.mlp.down_proj.weight'].shape (11008,4096)
t1 = time.time()
outputs = model.generate(**inputs, max_new_tokens=gen_len)
t2 = time.time()
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(t2-t1)
# -----------------------
print("\n\n++ dense")
change_forward_of_ActivationPruneLinear(model, to_sparse=False)
# model.state_dict()['model.layers.27.mlp.down_proj.weight'].shape (4096,11008)
t1 = time.time()
outputs = model.generate(**inputs, max_new_tokens=gen_len)
t2 = time.time()
print(t2-t1)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("end.")
