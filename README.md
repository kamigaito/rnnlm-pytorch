# RNN-based language models in pytorch

This is an implementation of bidirectional language models[[1]](#cite1) based on multi-layer RNN (Elman[[2]](#cite2), GRU[[3]](#cite3), or LSTM[[4]](#cite4)) with residual connections[[5]](#cite5) and character embeddings[[6]](#cite6).
After you train a language model, you can calculate perplexities for each input sentence based on the trained model.
You can also generate sentences from the trained model.
When calculating perplexities, this code can receive input sentences through http requests.
Note that some codes of this implementation are borrowed from [word-level language model example in pytorch](https://github.com/pytorch/examples).

## Setup

Run `pip install -r requirements.txt`

## Usage
### Training

Use `train.py` for training.
The `train.py` script accepts the following arguments:

```
optional arguments:
    --data', type=str, default='./dataset', help='location of the data corpus'
    --glove', type=str, default='', help='path to the glove embedding'
    --rnn_type', type=str, default='ResLSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, ResRNN_TANH, ResRNN_RELU, ResLSTM, ResGRU)'
    --direction', type=str, default='left2right', help='type of language model direction (left2right, right2left, both)'
    --wo_tok', action='store_true',  help='without token embeddings'
    --wo_char', action='store_true', help='without character embeddings'
    --tok_emb', type=int, default=200, help='The dimension size of word embeddings'
    --char_emb', type=int, default=50, help='The dimension size of character embeddings'
    --char_kmin', type=int, default=1, help='minimum size of the kernel in the character encoder'
    --char_kmax', type=int, default=5, help='maximum size of the kernel in the character encoder'
    --tok_hid', type=int, default=250, help='number of hidden units of the token level rnn layer'
    --char_hid', type=int, default=50, help='number of hidden units of the character level rnn layer'
    --nlayers', type=int, default=2, help='number of layers'
    --optim_type', type=str, default='SGD', help='type of the optimizer'
    --lr', type=float, default=20, help='initial learning rate'
    --clip', type=float, default=0.25, help='gradient clipping'
    --epochs', type=int, default=40, help='upper epoch limit'
    --batch_size', type=int, default=20, metavar='N', help='batch size'
    --cut_freq', type=int, default=10, help='cut off tokens in a corpus less than this value'
    --max_vocab_size', type=int, default=100000, help='cut off low-frequencey tokens in a corpus if the vocabulary size exceeds this value'
    --max_length', type=int, default=300, help='skip sentences longer than this value (-1: Infinite)'
    --dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)'
    --init_range', type=float, default=0.1, help='initialization range of the weights'
    --tied', action='store_true', help='tie the word embedding and softmax weights'
    --seed', type=int, default=1111, help='random seed'
    --cuda', action='store_true', help='use CUDA'
    --log-interval', type=int, default=10, metavar='N', help='report interval'
    --pretrain', type=str, default='', help='prefix to pretrained model'
    --save', type=str, default='./models/model', help='prefix to save the final model'
    --dict', type=str, default='./models/dict.pkl', help='path to save the dictionary'
```
### Evaluation (CLI)

Use `evaluation.py` for the evaluation.
The `evaluation.py` script accepts the following arguments:

```
optional arguments:
    --dict', type=str, default='./models/dict.pkl', help='location of the dictionary'
    --load', type=str, default='./models/model', help='prefix to model files'
    --outf', type=str, default='./output/ppl.txt', help='output file of preplexities for each sentence'
    --input_text', type=str, default='./dataset/test.txt', help='path to the input file'
    --seed', type=int, default=1234, help='random seed'
    --server', action='store_true', help='run as a server'
    --host', type=str, default='localhost', help='host name'
    --port', type=int, default='8888', help='access port number'
    --cuda', action='store_true', help='use CUDA'
    --batch_size', type=int, default=20, help='batch size'
    --log-interval', type=int, default=100, help='reporting interval'
```

### Evaluation (HTTP Server)

Use `evaluation.py` with `--server` option for the http post based evaluation.
If you run `evaluation.py --server`, a http server waits to receive sentences via http post requests.
How to post sentences to the server is wrtten in the example code `client.py`.

If you encounter this error
`"TypeError: __call__() missing 2 required positional arguments: 'receive' and 'send'"`
Run this command `pip install git+https://github.com/kennethreitz/responder`.

### Generation

If you want to generate sentences from the trained model, use `generate.py` for the generation.
By default, generated sentences are written in the `output/generated.txt`
Note that `generate.py` is not working on the model trained as the bidirectional language model.
The `generate.py` script accepts the following arguments:
```
optional arguments:
    --dict', type=str, default='./models/dict.pkl', help='location of the dictionary'
    --load', type=str, default='./models/model', help='prefix to model files'
    --outf', type=str, default='./output/generated.txt', help='output file for generated text'
    --sents', type=int, default='1000', help='maximum number of sentencess to generate'
    --words', type=int, default='100', help='maximum number of words to generate'
    --batch_size', type=int, default=20, help='batch size'
    --seed', type=int, default=1234, help='random seed'
    --cuda', action='store_true', help='use CUDA'
    --temperature', type=float, default=1.0, help='temperature - higher will increase diversity'
    --log-interval', type=int, default=100, help='reporting interval'
```

## LICENSE

See the file `LICENSE`.

## References

- <a name="cite1">[1]</a> Matthew E. Peters, Waleed Ammar, Chandra Bhagavatula and Russell Power. "Semi-supervised sequence tagging with bidirectional language models." arXiv preprint arXiv:1705.00108 (2017).
- <a name="cite2">[2]</a> Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Černocký and Sanjeev Khudanpur. "Recurrent neural network based language model." Eleventh annual conference of the international speech communication association. 2010.
- <a name="cite3">[3]</a> Junyoung Chung, Caglar Gulcehre, KyungHyun Cho and Yoshua Bengio. "Empirical evaluation of gated recurrent neural networks on sequence modeling." arXiv preprint arXiv:1412.3555 (2014). 
- <a name="cite4">[4]</a> Sundermeyer, Martin, Ralf Schlüter, and Hermann Ney. "LSTM neural networks for language modeling." Thirteenth annual conference of the international speech communication association. 2012.
- <a name="cite5">[5]</a> Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- <a name="cite6">[6]</a> Yoon Kim, Yacine Jernite, David Sontag and Alexander M. Rush. "Character-aware neural language models." Thirtieth AAAI Conference on Artificial Intelligence. 2016.
