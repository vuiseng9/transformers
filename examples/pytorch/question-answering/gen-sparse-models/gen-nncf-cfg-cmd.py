import json
import torch
import os

DEBUG=False

# Keep following for easy reference
# nncfcfg['compression'][1]["activations"] = {
#                 "mode": "symmetric"
#             }
# nncfcfg['compression'][1]["weights"] = {
#                 "mode": "symmetric",
#                 "signed": True,
#                 "per_channel": False
#             }
# torch.save(nncfcfg, "./squad_nncf_config.pth")
def override_input_info(nncfcfg, n):
    if "input_info" not in nncfcfg:
        raise ValueError("invalud input nncfcfg")
    nncfcfg["input_info"]=[dict(sample_size=[1,384], type="long")]*n
    return nncfcfg

def get_fast_quantization_cfg(nncfcfg):
    fast_initializer = {
        "range": {
                "num_init_samples": 4,
            },
            "batchnorm_adaptation": {
                "num_bn_adaptation_samples": 2
            }
        }
    if isinstance(nncfcfg['compression'], list):
        quantize_cfg_id = None
        for i, algo in enumerate(nncfcfg['compression']):
            if algo['algorithm'] == 'quantization':
                quantize_cfg_id = i
        if quantize_cfg_id is not None:
            nncfcfg['compression'][quantize_cfg_id]['initializer']=fast_initializer
    elif isinstance(nncfcfg['compression'], dict) and nncfcfg['compression']['algorithm'] == 'quantization':
        nncfcfg['compression']['initializer']=fast_initializer
    else:
        raise ValueError("input nncfcfg has no quantization section initialized")
    return nncfcfg

def set_init_sparsity(nncfcfg, sparsity_level):
    if isinstance(nncfcfg['compression'], list):
        magsparse_cfg_id = None
        for i, algo in enumerate(nncfcfg['compression']):
            if algo['algorithm'] == 'magnitude_sparsity':
                magsparse_cfg_id = i
        if magsparse_cfg_id is not None:
            nncfcfg['compression'][magsparse_cfg_id]['sparsity_init'] = sparsity_level

    elif isinstance(nncfcfg['compression'], dict) and nncfcfg['compression']['algorithm'] == 'magnitude_sparsity':
        nncfcfg['compression']['sparsity_init']=sparsity_level
    else:
        raise ValueError("input nncfcfg has no magnitude_sparsity section initialized")
    return nncfcfg

def gen_cmd():
    pass

if DEBUG is True:
    nncfcfg = torch.load("./squad_nncf_config.pth")
    fastq_nncfcfg = get_fast_quantization_cfg(nncfcfg)
    print(json.dumps(fastq_nncfcfg, indent=2))

pretrained_model_dict = {
    "bert-large": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-base":  "vuiseng9/bert-base-uncased-squad",
    "distilbert": "distilbert-base-cased-distilled-squad",
}

input_nport = {
    "bert-large": 3,
    "bert-base":  3,
    "distilbert": 2,
}

nncfcfg_dir = "sweep_sparsity_nncfcfg"
os.makedirs(nncfcfg_dir, exist_ok=True)

nncfcfg = torch.load("./squad_nncf_config.pth")
fastq_nncfcfg = get_fast_quantization_cfg(nncfcfg)

sparsity_list = [0.50, 0.60, 0.70, 0.80, 0.90]

cmdlist=[]

for label, hfckpt in pretrained_model_dict.items():
    for sparsity in sparsity_list:
        nncfcfg_filepth = "{}/{}_{}%_sparse_quantized.json".format(nncfcfg_dir, label, int(sparsity*100))

        final_nncfcfg = override_input_info(
                            set_init_sparsity(fastq_nncfcfg, sparsity),
                            input_nport[label]
                        )

        with open(nncfcfg_filepth, "w") as f:
            json.dump(final_nncfcfg, f, indent=4)

        # generate run cmd
        output_dir="{}-sparse-quantized/{}%".format(label, int(sparsity*100))

        msg=("echo \"[Info]: Creating {}\"\n".format(output_dir))
        cmd=(
            "python run_qa.py \\\n"
            "    --model_name_or_path {ckpt} \\\n"
            "    --dataset_name squad \\\n"
            "    --do_eval \\\n"
            "    --per_device_eval_batch_size 128 \\\n"
            "    --max_seq_length 384 \\\n"
            "    --doc_stride 128 \\\n"
            "    --output_dir $OUTROOT/{outdir} \\\n"
            "    --nncf_config $CFGROOT/{cfgpth} \\\n"
            "    --overwrite_output_dir \\\n"
            "    --gen_onnx \\\n"
            "\n".format(
                ckpt=hfckpt, outdir=output_dir, cfgpth=nncfcfg_filepth
            )
        )

        cmdlist.append((msg, cmd))

script_path="run_gen_sparse_quantized_model.sh"

with open(script_path, "w") as f:
    f.write('#!/usr/bin/env bash\n\n')
    f.write('WORKDIR=/path/to/workdir\n')
    f.write('OUTROOT=/path/to/outroot\n')
    f.write('CFGROOT=/path/to/cfgroot\n')
    f.write('mkdir -p $OUTROOT\n')
    f.write('\ncd $WORKDIR\n\n')
    
    for msg, cmd in cmdlist:
        f.write(msg)
        f.write(cmd)

print("yatta")