import math
import argparse
import torch
import torch.nn as nn
import pickle
import torchtext

import warnings

def neuralBalance(inl, oul, order):
    incoming = torch.linalg.norm(inl.weight, dim=1, ord=order)
    outgoing = torch.linalg.norm(oul.weight, dim=0, ord=order)
    optimal_l = torch.sqrt(outgoing/incoming)
    inl.weight.data *= optimal_l.unsqueeze(1)
    oul.weight.data /= optimal_l

class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
    
class Net(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """

    def __init__(
        self,
        embeddings,
        nhead=8,
        dim_feedforward=2048,
        num_layers=6,
        dropout=0.0,
        activation="relu",
        classifier_dropout=0.3,
    ):

        super().__init__()

        vocab_size, d_model = embeddings.size()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(d_model, 2)
        self.d_model = d_model

    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x


def main():
    parser = argparse.ArgumentParser(description='IMDB')
    parser.add_argument("--epochs", required=False, default = 100, type=int,help="Number Of Epochs The Model Is Trained For")
    parser.add_argument("--lr", required=False, default=1e-4, type=float, help="constant learning rate for model")
    parser.add_argument("--model", required=False, default = 'transformer', type=str, choices = ['transformer'], help="choose dataset")
    parser.add_argument("--dataset", required=False, default = 'IMDB', type=str, choices = ['IMDB'],help="choose dataset")
    parser.add_argument("--gpu", required=False, default = '0', type=str,help="Choose GPU to use")
    parser.add_argument("--batchsize", required=False, default = 30, type=int,help="Choose batch_size for the dataset")
    parser.add_argument("--l2_weight", required=False, default = 0, type=float, help="Multiplier for L2 Regularizer")
    parser.add_argument("--seed", required=False, default = 42, type=int,help="Choose seed")
    parser.add_argument("--neural_balance", required=False, default = 0, type=int,help="Whether we train with neural balance or not")
    parser.add_argument("--neural_balance_epoch", required=False, default = 1, type=int,help="Every how many epochs we are doing neural balance.")
    parser.add_argument("--order", required=False, default = 2, type=int,help="Order of norm when doing neural balance.")
    parser.add_argument("--neuralFullBalanceAtStart", required = False, default = 0, type = int, help="Whether neural balance is fully performed before the model's training begins")
    parser.add_argument("--trainDataFrac", required = False, default = 1, type = float, help = "What fraction of the training dataset is used in training")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    warnings.filterwarnings("ignore", category=UserWarning)

    batch_size = args.batchsize
    max_length = 256

    TEXT = torchtext.data.Field(
        lower=True, include_lengths=False, batch_first=True
    )
    LABEL = torchtext.data.Field(sequential=False)
    train_txt, test_txt = torchtext.datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(
        train_txt,
        vectors=torchtext.vocab.GloVe(name="6B", dim=50, max_vectors=50_000),
        max_size=50_000,
    )

    LABEL.build_vocab(train_txt)

    train_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train_txt, test_txt),
        batch_size=batch_size,
    )

    epochs = args.epochs
    model = Net(
        TEXT.vocab.vectors,
        nhead=5,
        dim_feedforward=2048,  # reduced feedforward network size
        num_layers=6,  # reduced number of layers
        dropout=0.3,
        classifier_dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    if args.l2_weight != 0:
        print("using l2 =", args.l2_weight)


    lr = args.lr
    optimizer = torch.optim.sgd(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=args.l2_weight
    )

    hist = {}

    hist['train_loss'] = []
    hist['train_acc'] = []
    hist['test_loss'] = []
    hist['test_acc'] = []

    print("starting")

    lay = model.transformer_encoder.layers

    if args.neuralFullBalanceAtStart == 1:
        print('full balancing at start')
        while(True):
            restart=False
            for i in lay:
                lay1, lay2 = i.linear1, i.linear2
                incoming = torch.linalg.norm(lay1.weight, dim=1, ord=args.order)
                outgoing = torch.linalg.norm(lay2.weight, dim=0, ord=args.order)
                optimal_l = torch.sqrt(outgoing/incoming).sum()/incoming.shape[0]
                print(optimal_l)
                if optimal_l > 1.001 or optimal_l < .999:
                    restart=True
                neuralBalance(lay1, lay2, order = args.order)
            if not restart:
                break

    for epoch in range(epochs):
        if args.neural_balance == 1 and epoch % args.neural_balance_epoch == 0:
            print("performing neural balance")
            for i in model.transformer_encoder.layers:
                neuralBalance(i.linear1, i.linear2, 2)
        print(f"{epoch=}")
        epoch_loss = 0
        epoch_correct = 0
        epoch_count = 0
        for idx, batch in enumerate(iter(train_iter)):
            predictions = model(batch.text.to(device))
            labels = batch.label.to(device) - 1

            loss = criterion(predictions, labels)

            correct = predictions.argmax(axis=1) == labels
            acc = correct.sum().item() / correct.size(0)

            epoch_correct += correct.sum().item()
            epoch_count += correct.size(0)

            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

        with torch.no_grad():
            test_epoch_loss = 0
            test_epoch_correct = 0
            test_epoch_count = 0

            for idx, batch in enumerate(iter(test_iter)):
                predictions = model(batch.text.to(device))
                labels = batch.label.to(device) - 1
                test_loss = criterion(predictions, labels)

                correct = predictions.argmax(axis=1) == labels
                acc = correct.sum().item() / correct.size(0)

                test_epoch_correct += correct.sum().item()
                test_epoch_count += correct.size(0)
                test_epoch_loss += loss.item()

        print(f"{epoch_loss=}")
        hist['train_loss'].append(epoch_loss)
        print(f"epoch accuracy: {epoch_correct / epoch_count}")
        hist['train_acc'].append(epoch_correct / epoch_count)
        print(f"{test_epoch_loss=}")
        hist['test_loss'].append(test_epoch_loss)
        print(f"test epoch accuracy: {test_epoch_correct / test_epoch_count}")
        hist['test_acc'].append(test_epoch_correct / test_epoch_count)

if __name__ == "__main__":
    main()