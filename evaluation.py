###############################################################################
#
# This file calculates perplexitis of recieved sentences
#
###############################################################################

import argparse
import torch
import pickle

from typing import List
import logzero
import uvicorn
from fastapi import FastAPI
from logzero import logger
from pydantic import BaseModel
import torch.nn as nn
import models
import data
import math

def options():

    parser = argparse.ArgumentParser(description='PyTorch Language Model')

    # Model parameters.
    parser.add_argument('--dict', type=str, default='./models/dict.pkl',
                    help='location of the dictionary')
    parser.add_argument('--load', type=str, default='./models/model',
                    help='prefix to model files')
    parser.add_argument('--outf', type=str, default='./output/ppl.txt',
                    help='output file of preplexities for each sentence')
    parser.add_argument('--input_text', type=str, default='./dataset/test.txt',
                    help='path to the input file')
    parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
    parser.add_argument('--server', action='store_true',
                    help='run as a server')
    parser.add_argument('--host', type=str, default='localhost',
                    help='host name')
    parser.add_argument('--port', type=int, default='8888',
                    help='access port number')
    parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
    parser.add_argument('--batch_size', type=int, default=20,
                    help='batch size')
    parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
    opts = parser.parse_args()
    return opts

class InSentences(BaseModel):
    sentences: List[str]

class Score(BaseModel):
    lm_score: float

###############################################################################
# Evaluator
###############################################################################

def evaluate(opts, corpus, input_texts, model, criterion, device):
    """
    Parameter
    ---------
        corpus: Corpus
    Return
    ------
        total_loss: float
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = corpus.dictionary.tok_len()
    results = []
    # Do not propagate gradients
    with torch.no_grad():
        for batch_id, batch in enumerate(data.data2batch(input_texts, corpus.dictionary, opts.batch_size, flag_shuf=False)):
            hidden = model.init_hidden(batch)
            # Cut the computation graph (Initialize)
            hidden = models.repackage_hidden(hidden)
            # LongTensor of token_ids [seq_len, batch_size]
            input = model.batch2input(batch, device)
            seq_len = input["word"].shape[0]
            batch_size = input["word"].shape[1]
            # target_flat: LongTensor of token_ids [seq_len*batch_size]
            target_flat = model.batch2flat(batch, device)
            # clear previous gradients
            model.zero_grad()
            # output: [seq_len, nbatch, ntoken], hidden: [nlayer, nbatch, nhid]
            output, hidden = model(input, hidden)
            # output_flat: LongTensor of token_ids [seq_len*batch_size, ntoken]
            output_flat = output.view(-1, output.shape[2])
            # batch_loss: LongTensor of token_ids [seq_len*batch_size]
            batch_loss = criterion(output_flat, target_flat)
            # batch_loss: LongTensor of token_ids [seq_len, batch_size]
            batch_loss = batch_loss.reshape(seq_len, batch_size)
            # batch_loss: LongTensor of token_ids [batch_size]
            batch_loss = torch.mean(batch_loss, 0)
            for sent_loss in batch_loss:
                ppl = math.exp(sent_loss)
                results.append(ppl)
    return results


def main():

    ###############################################################################
    # Load command line options.
    ###############################################################################

    opts = options()
    # Set the random seed manually for reproducibility.
    torch.manual_seed(opts.seed)

    ###############################################################################
    # Build a model
    ###############################################################################
    
    with open(opts.load + ".params", 'rb') as f:
        params = pickle.load(f)
    model = models.RNNModel(params)
    model.load_state_dict(torch.load(opts.load + ".pt"))
    if torch.cuda.is_available():
        if not opts.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if opts.cuda else "cpu")
    model.to(device)
    model.eval()
    
    ###############################################################################
    # Load dictionary
    ###############################################################################
    
    corpus = data.Corpus(opts)
    corpus.load_dict()
   
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=corpus.dictionary.pad_id())
    
    ###############################################################################
    # Run as a server 
    ###############################################################################
    
    if opts.server:
        
        app = FastAPI()

        @app.post('/lm', response_model=List[Score], description="get several scores with POST method")
        def predict(req: InSentences):
            print(req)
            stream = []
            for sent in req.sentences:
                seq = ["<s>"] + sent.split(" ") + ["</s>"]
                stream.append(corpus.sent2ids(seq))
            return [ Score(lm_score=(ppl)) for ppl in evaluate(opts, corpus, stream, model, criterion, device) ]
        
        logzero.loglevel(10)  # log_level = DEBUG
        uvicorn.run(app, host=opts.host, port=opts.port, workers=1, logger=logger, debug=True)

    ###############################################################################
    # Calculates perplexities for sentences in the input file
    ###############################################################################
    
    else:
        input_texts = corpus.tokenize(opts.input_text)
        with open(opts.outf, 'w') as f_out:
            for ppl in evaluate(opts, corpus, input_texts, model, criterion, device):
                f_out.write(str(ppl) + "\n")

if __name__ == "__main__":
    main()

