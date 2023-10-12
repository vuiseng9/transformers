from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.profiler import profile, record_function, ProfilerActivity

# pipe = pipeline(model="roberta-large-mnli")
# print(pipe("This restaurant is awesome")) # [{'label': 'NEUTRAL', 'score': 0.7313136458396912}]

# sentiment_analysis = pipeline("sentiment-analysis",model="assemblyai/bert-large-uncased-sst2")
# print(sentiment_analysis("I love this!"))


model_id = "assemblyai/bert-large-uncased-sst2"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    # with_stack=True, 
    # profile_memory=True, 
    # with_modules=True,
    # record_shapes=True,
    ) as prof:
    output = pipe("I love this!")

print(output)

report_strings = prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=40)
print(report_strings)
# , row_limit=10))
def filter_report(single_newline_delimited_string):
    report_lines = single_newline_delimited_string.split('\n')
    report_len = len(report_lines)

    filtered_lines = []
    for line_id, line in enumerate(report_lines):
        if line_id < 3 or line_id >= (report_len-3) or 'marker' in line:
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)

print(filter_report(report_strings))
print()

print("joto")