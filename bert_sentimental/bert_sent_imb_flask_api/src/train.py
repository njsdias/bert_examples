# import src.config as config
# import pandas as pd
# from sklearn import model_selection
# import src.dataset as dataset
# import torch

import src.config as config
import src.dataset as dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from src.model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def run():
    # load dataset
    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")
    # label sentiment string
    dfx.sentiment = dfx.sentiment.apply(
        lambda x: 1 if x == "positive" else 0)

    # split data into train and validation
    # stratify : when the splits the train an validations
    # put the same ratio between negative and positive
    df_train, df_valid = model_selection.train_test_split(dfx,
                                                          test_size=0.1,
                                                          random_state=42,
                                                          stratify=dfx.sentiment.values)

    # define train/test dataframe
    df_train = df_train.reset_index(drop=True)
    df_test = df_valid.reset_indx(drop=True)

    # provide train/test dataframe to dataloader
    # to transform data in an appropraited formated
    # to BERT model
    train_dataset = dataset.BERTDataset(review= df_train.review.values,
                                        target= df_train.sentiment.values)

    # transfer data to dataloader
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=config.TRAIN_BATCH_SIZE,
                                                    num_workers=4)

    # do the same for validation
    valid_dataset = dataset.BERTDataset(review= df_test.review.values,
                                        target= df_test.sentiment.values)

    # transfer data to dataloader
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=config.VALID_BATCH_SIZE,
                                                    num_workers=1)

    # specify where the train will be done : cuda
    device = torch.device(config.DEVICE)
    model = BERTBaseUncased()
    model.to(device)

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

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=num_train_steps)

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()


