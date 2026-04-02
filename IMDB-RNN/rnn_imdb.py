import torch
from torchtext import data, datasets
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
import time
import numpy as np
import random
import pickle
import argparse
from tqdm import tqdm
import csv

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        # Concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        return self.fc(hidden)

def binary_accuracy(preds, y):
    """Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8"""
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def neuralBalance(lstm_layer, order):
    num_layers = lstm_layer.num_layers
    
    for layer in range(num_layers):
        # Define the suffix for identifying parameters for each layer
        suffix = f'_l{layer}'
        
        # Access input-to-hidden and hidden-to-hidden weights
        weight_ih = getattr(lstm_layer, f'weight_ih{suffix}')
        weight_hh = getattr(lstm_layer, f'weight_hh{suffix}')

        inl = weight_ih.data
        oul = weight_hh.data.T

        incoming = torch.linalg.norm(inl, dim=1, ord=order)
        outgoing = torch.linalg.norm(oul, dim=0, ord=order)
        optimal_l = torch.sqrt(outgoing/incoming)
        
        inl *= optimal_l.unsqueeze(1)
        oul /= optimal_l

        oul = oul.T

def train(model, iterator, optimizer, criterion, l2_weight, nb, order, device):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        text, text_lengths = batch.text
        
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = criterion(predictions, batch.label)

        if l2_weight > 0 and order == 1:
            params = torch.cat([x.view(-1) for x in model.parameters()])
            l1_regularization = l2_weight * torch.norm(params, 1)
            loss+= l1_regularization

        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    if nb:
        neuralBalance(model.rnn, order = order)
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in tqdm(iterator):

            text, text_lengths = batch.text
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def main():
    parser = argparse.ArgumentParser(description='IMDB')
    parser.add_argument("--epochs", required=False, default = 100, type=int,help="Number Of Epochs The Model Is Trained For")
    parser.add_argument("--lr", required=False, default=1e-3, type=float, help="constant learning rate for model")
    parser.add_argument("--n_layers", required=False, default = 2, type=int, choices = [2, 3, 4, 5], help="choose number of layers in RNN")
    parser.add_argument("--dataset", required=False, default = 'imdb', type=str, choices = ['imdb'],help="choose dataset")
    parser.add_argument("--gpu", required=False, default = '0', type=str,help="Choose GPU to use")
    parser.add_argument("--batchsize", required=False, default = 256, type=int,help="Choose batch_size for the dataset")
    parser.add_argument("--l2_weight", required=False, default = 0, type=float, help="Multiplier for L2 Regularizer")
    parser.add_argument("--seed", required=False, default = 42, type=int,help="Choose seed")
    parser.add_argument("--neural_balance", required=False, default = 0, type=int,help="Whether we train with neural balance or not")
    parser.add_argument("--neural_balance_epoch", required=False, default = 1, type=int,help="Every how many epochs we are doing neural balance.")
    parser.add_argument("--order", required=False, default = 2, type=int,help="Order of norm when doing neural balance.")
    parser.add_argument("--neuralFullBalanceAtStart", required = False, default = 0, type = int, help="Whether neural balance is fully performed before the model's training begins")
    parser.add_argument("--trainDataFrac", required = False, default = 1, type = float, help = "What fraction of the training dataset is used in training")
    parser.add_argument("--foldername", required = True, default = "default", type=str, help = "folder name")
    parser.add_argument("--filename", required = True, default = "default", type=str, help = "file name")
    args = parser.parse_args()

    set_seed(args.seed)

    headers = ['epoch', 'test_accuracy', 'train_loss', 'test_loss']
    try:
        csv_file = f'IMDB-RNN/hist/{args.foldername}/{args.filename}.csv'
        with open(csv_file, mode='w', newline='') as file:
            file.write(','.join(headers) + '\n')
    except IOError as e:
        print(f"Unable to create file {csv_file}: {e}")
        exit(1)

    file.close()

    # Define the Fields for processing the dataset
    tokenizer = get_tokenizer('basic_english')

    TEXT = data.Field(tokenize=tokenizer, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float)

    # Load the IMDb dataset
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    # Split the training data to create a validation set
    # train_data, valid_data = train_data.split(random_state = torch.manual_seed(SEED))
    train_data, valid_data = train_data.split(split_ratio=0.7)

    # Build the vocabulary and load pre-trained word embeddings (GloVe)
    MAX_VOCAB_SIZE = 25000

    TEXT.build_vocab(train_data, 
                    max_size = MAX_VOCAB_SIZE, 
                    vectors = "glove.6B.100d", 
                    unk_init = torch.Tensor.normal_)

    LABEL.build_vocab(train_data)

    # Create iterators for the data
    BATCH_SIZE = args.batchsize

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE,
        sort_within_batch = True,
        device = device)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = args.n_layers
    BIDIRECTIONAL = False
    DROPOUT = 0.0
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = RNN(INPUT_DIM, 
                EMBEDDING_DIM, 
                HIDDEN_DIM, 
                OUTPUT_DIM, 
                N_LAYERS, 
                BIDIRECTIONAL, 
                DROPOUT, 
                PAD_IDX)

    # Load the pre-trained embeddings
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # Zero the initial weights of the unknown and padding tokens
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    if args.l2_weight > 0:
        print(f"using l2 = {args.l2_weight}")
    if args.order == 2:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)


    N_EPOCHS = args.epochs

    best_valid_loss = float('inf')

    hist = {}
    hist['train_loss'] = []
    hist['train_acc'] = []
    hist['test_loss'] = []
    hist['test_acc'] = []

    if args.neuralFullBalanceAtStart == 1:
        num_layers = model.rnn.num_layers
        print('balancing fully at start')

        while True:  
            restart = False
            for layer in range(num_layers):
                # Define the suffix for identifying parameters for each layer
                suffix = f'_l{layer}'
                
                # Access input-to-hidden and hidden-to-hidden weights
                weight_ih = getattr(model.rnn, f'weight_ih{suffix}')
                weight_hh = getattr(model.rnn, f'weight_hh{suffix}')

                inl = weight_ih.data
                oul = weight_hh.data.T

                incoming = torch.linalg.norm(inl, dim=1, ord=2)
                outgoing = torch.linalg.norm(oul, dim=0, ord=2)
                optimal_l = torch.sqrt(outgoing/incoming)
                temp = optimal_l.sum()/incoming.shape[0]

                if temp > 1.01 or temp < .99:
                    restart = True
                
                print(temp)

                inl *= optimal_l.unsqueeze(1)
                oul /= optimal_l

                oul = oul.T

            if not restart:
                break

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        if args.neural_balance == 1 and epoch % args.neural_balance_epoch == 0:
            train_loss, train_acc = train(model, train_iterator, optimizer, criterion, args.l2_weight, True, args.order, device)
        else:
            train_loss, train_acc = train(model, train_iterator, optimizer, criterion, args.l2_weight, False, args.order, device)
        
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()

        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        hist['train_loss'].append(train_loss)
        hist['train_acc'].append(train_acc)
        hist['test_loss'].append(valid_loss)
        hist['test_acc'].append(valid_acc)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, valid_acc, train_loss, valid_loss])

if __name__ == "__main__":
    main()