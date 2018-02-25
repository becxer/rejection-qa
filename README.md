## Rejection QA

To train or test the model you need the pre-trained ELMo model. It can be downloaded
[here](https://docs.google.com/uc?export=download&id=1vXsiRHxJqsj3HLesUIet0x4Yrjw0S54D).
Then unzip it and store in ~/data/lm (or change config.py to alter its expected location). For example:

```
mkdir -p ~/data/lm
cd ~/data/lm
mv ~/Download/squad-context-concat-skip.tar.gz .
tar -xzf squad-context-concat-skip.tar.gz
rm squad-context-concat-skip.tar.gz
```

### Training
Now the model can be trained using:

`python docqa/reject/ablate_rejection_squad.py`

This code based on allenai's [document-qa](https://github.com/allenai/document-qa)

