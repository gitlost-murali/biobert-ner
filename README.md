# BERT-ner

End-to-End NER for BERT variants. Includes everything from processing input files, modelling, evaluating the models (F1-score) to inferring the output.

For downloading pre-trained Transformer models from [HuggingFace](https://huggingface.co/models), use [lfs](https://git-lfs.github.com/)

```
git lfs install
git clone https://huggingface.co/ixa-ehu/ixambert-base-cased
```

```
BASE_MODEL_PATH => Folder path of bert-base-uncased OR biobert-large-cased-v1.1

BASE_MODEL_DIM => Output Dimension of the layer before softmax. 1024 for bioBERT & 768 for BERT

TRAINING_FILE => Use the function read_bilou() or process_data_conll() depending on the format.
```

```
read_bilou(): [
                [w1,w2,w3],[tag_w1,tag_w2,tag_w3],
                [w1,w2],[tag_w1,tag_w2],
              ]
```

```
process_data_conll(): CONLL format

w1  tag_w1
w2  tag_w2
w3  tag_w3

w1  tag_w1
w2  tag_w2
```