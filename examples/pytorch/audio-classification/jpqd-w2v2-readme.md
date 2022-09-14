### Apply JPQD on wav2vec2/keyword detection

### create an environment 
```bash 
conda create -n <env_name> python=3.8 
``` 

### Set up JPQD-NNCF 
```bash 
cd <workdir> 
git clone https://github.com/vuiseng9/nncf 
cd nncf 
git checkout nncf-mvmt-p3-ac
python setup.py develop 
``` 

### Set up JPQD-transformers 
```bash 
cd <workdir> 
git clone https://github.com/vuiseng9/transformers 
cd transformers 
git checkout v4.16.2-w2v2opt 
pip install -e . 
``` 

### Set up dependency of HF Audio Classification example
```bash
cd transformers/examples/pytorch/audio-classification
pip install -r requirements.txt
```

### Set up optimum openvino
```
git clone https://github.com/vuiseng9/optimum-openvino
cd optimum-openvino
git checkout -b w2v2-dev
pip install -e .
```

### Install pytorch (not mandatory; this is validated torch version) 
```bash 
pip3 install torch==1.10.2+cu113 \
  torchvision==0.11.3+cu113 \
  torchaudio==0.10.2+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
or
```bash
pip3 install \
  torch==1.9.1+cu102 \
  torchvision==0.10.1+cu102 \
  torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html 
``` 

### Eval pretrained wav2vec2/ks
vscode debug config is enclosed. Move or make a softlink to ```vscode-launch.json``` to ```.vscode/launch.json```.
Config ```[Eval] w2v2 keyword spotting``` should work out of the box with 98.28 accuracy.

### Run JPQD on Wav2vec2/KS
revise and run ```nncf-w2v2-ac.sh*```