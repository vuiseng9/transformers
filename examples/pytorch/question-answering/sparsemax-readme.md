### Setup SparseMax-enabled BERT

```bash
# create/activate a python environment (Optional)

git clone https://github.com/vuiseng9/transformers
cd transformers
git checkout integrate-sparsemax
pip install -e .
cd examples/pytorch/question-answering
pip install -r requirements.txt
```

### Run - Evaluate BERT-Large fine-tuned on SQuAD dataset (Baseline - softmax)
F1: 93.2%
```bash
cd transformers/examples/pytorch/question-answering
python3 run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./eval-bert-large-squad-softmax \
    --overwrite_output_dir
```
### Run - Evaluate BERT-Large fine-tuned on SQuAD dataset (sparsemax)
F1: 57.1%
```bash
cd transformers/examples/pytorch/question-answering
python3 run_qa.py \
    --model_name_or_path bert-large-uncased-whole-word-masking-finetuned-squad \
    --dataset_name squad \
    --sparsemax \
    --do_eval \
    --per_device_eval_batch_size 128 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./eval-bert-large-squad-sparsemax \
    --overwrite_output_dir
```