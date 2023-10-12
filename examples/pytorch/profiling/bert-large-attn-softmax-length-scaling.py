import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm

def filter_report(single_newline_delimited_string, prefix):
    report_lines = single_newline_delimited_string.split('\n')
    report_len = len(report_lines)

    filtered_lines = []
    for line_id, line in enumerate(report_lines):
        if line_id < 3 or line_id >= (report_len-3) or 'marker' in line:
            if line != "":
                filtered_lines.append(f"###seqlen,{prefix},"+line)

    return '\n'.join(filtered_lines)

# model creation
model_id = "assemblyai/bert-large-uncased-sst2"
dtype=torch.float16
model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=dtype)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)

sweep_length = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
n_loop = 10
device = "cuda:0"
cuda_version = torch.version.cuda

print(f"{'-'*100}\nProfiling configuration")
print(f"* Model ID: {model_id}")
print(f"* Datatype: {next(model.parameters()).dtype}")
print(f"* CUDA_VERSION: {cuda_version}")
print(f"* Device: {device}")
print(f"* Sampling loop: {n_loop}")
print(f"* Sweep Length: {sweep_length}")

model.eval()
for sl in sweep_length:
    intensor = torch.rand(1, sl, 1024, dtype=dtype).to(device)
    print("-"*100)
    print(f"input tensor shape: {intensor.shape}")
    for _ in tqdm(range(10), desc="- warming up ..."):
        with torch.no_grad():
            out = model.bert.encoder(intensor)

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # profile_memory=True, 
            # record_shapes=True,
            # with_stack=True, 
            # with_modules=True,
            ) as prof:
        
        for _ in tqdm(range(n_loop), desc=f"\n- profiling ... seqlen {sl}, total loop {n_loop} "):
            with torch.no_grad():
                with torch.profiler.record_function("marker:BERT-large-Encoder"):
                    out = model.bert.encoder(intensor)

    print(f"\n- output shape: {out.last_hidden_state.shape} \n")

    report_strings = prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=40)

    print(filter_report(report_strings, sl))


# to hack the print, pls modify as following
# ..../miniconda3/envs/sixer-2310-lm-eval-dys/lib/python3.9/site-packages/torch/autograd/profiler_util.py
# def _format_time(time_us):
#     """Defines how to format time in FunctionEvent"""
#     US_IN_SECOND = 1000.0 * 1000.0
#     US_IN_MS = 1000.0
#     return ',{:.3f}, ms'.format(time_us / US_IN_MS)
#
#
#
    # for evt in events:
    #     if event_limit == row_limit:
    #         break
    #     if top_level_events_only and evt.cpu_parent is not None:
    #         continue
    #     else:
    #         event_limit += 1
    #     name = evt.key
    #     if max_name_column_width is not None and len(name) >= max_name_column_width - 3:
    #         name = name[:(max_name_column_width - 3)] + "..."
    #     name += ',' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! add this line
    #     row_values = [

# CUDA_VISIBLE_DEVICES=1 python bert-large-attn-softmax-length-scaling.py 2>&1 | tee log.profiling
# grep ^### log.profiling | grep -v '\-\-\-' | grep -vi "self" > profiling.csv