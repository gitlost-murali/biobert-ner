import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel


def process_data_conll(data_path):
    """To read CONLL type data

    Args:
        data_path ([str]): File name

    Returns:
        Sentences, Sentence_wise_tags, Encoding Tags
    """
    df = pd.read_csv(data_path, encoding="latin-1", sep="\t")
    sentences = []
    sent_tags = []
    tmp_snt = [] # To stop collecting words after a line break i.e Instance Boundary
    tmp_tag = [] # To stop collecting words after a line break i.e Instance Boundary

    df = df.fillna("-1")

    for wrd, tg in df.values:
        if wrd=="-1" and tg=="-1": 
            sentences.append(tmp_snt)
            sent_tags.append(tmp_tag)
            tmp_snt = []
            tmp_tag = []
            continue

        tmp_snt.append(wrd)
        tmp_tag.append(tg)

    enc_tag = preprocessing.LabelEncoder()
    tags_list = []
    for tags in sent_tags:
        tags_list.extend(tags)
 
    enc_tag.fit(tags_list)
    sent_tags = [enc_tag.transform(tags) for tags in sent_tags]

    return sentences, sent_tags, enc_tag


if __name__ == "__main__":
    sentences, tag, enc_tag = process_data_conll(config.TRAINING_FILE)
    
    meta_data = {
        "enc_tag": enc_tag
    }

    joblib.dump(meta_data, "meta.bin")

    num_tag = len(list(enc_tag.classes_))

    (
        train_sentences,
        test_sentences,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, tag, random_state=42, test_size=0.1)

    train_dataset = dataset.EntityDataset(
        texts=train_sentences, tags=train_tag, O_tag_id= enc_tag.transform(["O"])[0]
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.EntityDataset(
        texts=test_sentences, tags=test_tag, O_tag_id= enc_tag.transform(["O"])[0]
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )
    
    model = EntityModel(num_tag=num_tag)
    device = torch.device("cuda" if config.CUDA else "cpu")
    model.to(device) #BioBERT is taking alot of space

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss
