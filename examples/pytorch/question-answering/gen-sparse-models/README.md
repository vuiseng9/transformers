# Generate Sparse-Quantized BERT models

# Setup
```bash
https://github.com/vuiseng9/nncf (nncf-apr)
https://github.com/vuiseng9/transformers (gen-sparse-models)
```
# Run
```bash
cd transformers/examples/pytorch/question-answering/gen-sparse-models

# Generate nncf config and bash scripts
python gen-nncf-cfg-cmd.py

# Update Variable at the begining of the script
vim run_gen_sparse_quantized_model.sh

# Relax and Viola
source run_gen_sparse_quantized_model.sh
```