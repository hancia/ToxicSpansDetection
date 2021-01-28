Dummy: https://github.com/plutasnyy/Interpretable-Attention/blob/master/random_classifier.py

Naive Bayes: project/naivebayes/

Ortho LSTM: https://github.com/plutasnyy/Interpretable-Attention

BERT: project/span_bert/

SHAP: project/binary_bert/ https://colab.research.google.com/drive/1DQVnQp9asCFJFxvONHhV5Fo0xrra13c6?usp=sharing
  

## How to run   
First, install dependencies   
```bash
conda env create --file environment.yml
 ```   
or
```bash
conda env update --file environment.yml
 ```   
`config.ini`
```ini
[cometml]
apikey = mySecretKey
projectname = myProjectName
workspace = workspaceName
```

Minimal BERT-SPAN params to locally run:
`--no-logger --data-path data/spans --name test --data-cutoff 10 --batch-size 1 --epochs 1 --model bert`
