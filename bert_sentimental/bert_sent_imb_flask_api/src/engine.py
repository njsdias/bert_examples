
# Train and evaluation functions
import torch
import torch.nn as nn
from tqdm import tqdm


def loss_fn(ouputs,targets):
    """
    Evaluate loss function among output and targets
    :param ouputs: model output
    :param targets: target of out dataset
    :return: loss function evaluted by binary cross entropy
    """
    return nn.BCEWithLogitsLoss()(ouputs, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler):

    # put the model in train mode
    model.train()

    # loop for each batch
    # bi:batch_index ; d:dataset
    # len(datatloader) = total of batches that we have
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):

        # grap it from datset.py
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_typed_ids"]
        targets = d["targets"]

        # put the tensors to device (cpu/gpu)
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs,targets)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):

    # put model in evaluation mode
    model.eval()

    # to store final targets
    fin_targets = []
    # to store final outputs
    fin_outputs = []

    with torch.no_grad():
        # loop for each batch
        # bi:batch_index ; d:dataset
        # len(datatloader) = total of batches that we have
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            # grap it from datset.py
            ids = d["ids"]
            mask = d["mask"]
            token_type_ids = d["token_typed_ids"]
            targets = d["targets"]

            # put the tensors to device (cpu/gpu)
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            # convert output to sigmoid function
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return fin_outputs, fin_targets






