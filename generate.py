###############################################################################
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import torch
import pickle
import models
import data

def options():

    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model Generator')

    # Model parameters.
    parser.add_argument('--dict', type=str, default='./models/dict.pkl',
                    help='location of the dictionary')
    parser.add_argument('--load', type=str, default='./models/model',
                    help='prefix to model files')
    parser.add_argument('--outf', type=str, default='./output/generated.txt',
                    help='output file for generated text')
    parser.add_argument('--sents', type=int, default='1000',
                    help='maximum number of sentencess to generate')
    parser.add_argument('--words', type=int, default='100',
                    help='maximum number of words to generate')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
    parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
    parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
    parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
    parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
    opts = parser.parse_args()
    return opts

def pred2batch(dictionary, model, word_ids, bsz, device):
    input_data = [[[word_ids[sid]], [[dictionary.char_conv2id(char) for char in dictionary.conv2word(word_ids[sid])]]] for sid in range(bsz)]
    for input_batch in data.data2batch(input_data, dictionary, bsz):
        hidden = model.init_hidden(input_batch)
        hidden = models.repackage_hidden(hidden)
        input_dict = model.word2input(input_batch, device)
        return input_dict, hidden

def init2batch(prms, dictionary, model, bsz, device):
    ntokens = dictionary.tok_len()
    if prms["direction"] == "left2right":
        word = "<s>"
    else:
        word = "</s>"
    word_id = dictionary.conv2id(word)
    word_ids = [word_id for i in range(bsz)]
    return pred2batch(dictionary, model, word_ids, bsz, device)

def generate(opts, prms, dictionary, model, device):
    with open(opts.outf, 'w') as outf:
        with torch.no_grad():  # no tracking history
            num_sents = 0
            for sid in range(0, opts.sents, opts.batch_size):
                if num_sents > opts.sents:
                    bsz = opts.sents - num_sents
                else:
                    bsz = opts.batch_size
                out_list = []
                input_batch, hidden = init2batch(prms, dictionary, model, bsz, device)
                for seq_id in range(0, opts.words):
                    output, hidden = model(input_batch, hidden)
                    word_weights = output.squeeze().div(opts.temperature).exp().cpu()
                    word_ids = torch.multinomial(word_weights, 1)
                    out_list.append(word_ids)
                    input_batch, hidden = pred2batch(dictionary, model, word_ids, bsz, device)
                out_list = torch.stack(out_list, 1).squeeze()
                if prms["direction"] == "right2left":
                    out_list = out_list.flip(1)
                for sentence in out_list:
                    line = ""
                    for tid in sentence:
                        token = dictionary.conv2word(tid)
                        if token in ["<s>", "</s>", "<pad>"]:
                            continue
                        if line != "":
                            line += " "
                        line += token
                    outf.write(line + "\n")
                num_sents += bsz
                if num_sents % opts.log_interval == 0:
                    print('| Generated {}/{} sentencess'.format(num_sents, opts.sents))

def main():

    ###############################################################################
    # Load command line options.
    ###############################################################################

    opts = options()
    # Set the random seed manually for reproducibility.
    torch.manual_seed(opts.seed)

    if opts.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3")
    
    ###############################################################################
    # Load the dictionary
    ###############################################################################

    with open(opts.dict, "rb") as f:
        dictionary = pickle.load(f)
    
    ###############################################################################
    # Build a model
    ###############################################################################
    
    with open(opts.load + ".params", 'rb') as f:
        params = pickle.load(f)

    # Model check
    if params["direction"] == "both":
        print("WARNING: Bidirectional language model is not supproted by this generator.")
        assert(False)
    model = models.RNNModel(params)
    model.load_state_dict(torch.load(opts.load + ".pt"))
    if torch.cuda.is_available():
        if not opts.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if opts.cuda else "cpu")
    model.to(device)
    model.eval()
    generate(opts, params, dictionary, model, device)

if __name__ == "__main__":
    main()
