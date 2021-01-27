import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():  #BioBERT is taking alot of space
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, loss = model(data["ids"],data["mask"], data["token_type_ids"], data["target_tag"])
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items(): #BioBERT is taking alot of space
            data[k] = v.to(device)
        _, loss = model(data["ids"],data["mask"], data["token_type_ids"], data["target_tag"])
        final_loss += loss.item()
    return final_loss / len(data_loader)

def eval_with_metrics(data_loader, model, device, enc_tag):
    model.eval()

    y_pred = []
    y_ground = []

    for data in (data_loader):
        sentence_lengths = data["length"].tolist()
        for k, v in data.items(): #BioBERT is taking alot of space
            data[k] = v.to(device)
        tags, _ = model(data["ids"],data["mask"], data["token_type_ids"], data["target_tag"])
        for ix, tag in enumerate(tags):
            result = enc_tag.inverse_transform(
                    tag.argmax(1).cpu().numpy()
                )

            y_pred.extend(result[:sentence_lengths[ix]])

            ground_truth_seq = data["target_tag"][ix][:sentence_lengths[ix]].tolist()
            ground_truth = enc_tag.inverse_transform(ground_truth_seq)
            y_ground.extend(ground_truth)

    class_labels = (enc_tag.classes_).tolist()
    class_labels.remove("O")
    metrics = classification_report(y_pred=y_pred, y_true=y_ground, labels= class_labels)

    return metrics