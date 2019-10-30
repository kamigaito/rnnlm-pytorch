# RNN-based language models in pytorch

This is an implementation of bidirectional language models[[1]](#cite1) based on multi-layer RNN (Elman[[2]](#cite2), GRU[[3]](#cite3), or LSTM[[4]](#cite4)) with residual connections[[5]](#cite5) and character embeddings[[6]](#cite6).
After you train a language model, you can calculate perplexities for each input **sentence** based on the trained model.
You can also generate sentences from the trained model.
When calculating perplexities, this code can receive input sentences through http requests.
Note that some codes of this implementation are borrowed from [word-level language model example in pytorch](https://github.com/pytorch/examples).

## Setup

Run `pip install -r requirements.txt`

## Usage
### Training

Use `train.py` for training.
The `train.py` script accepts the following arguments:

- `--data`: Location of the data corpus. `type=str`, `default='./dataset'`.
- `--glove`: Path to the glove embedding. `type=str`, `default=''`.
- `--rnn_type`: Type of the recurrent net. You can select one from [RNN_TANH, RNN_RELU, LSTM, GRU, ResRNN_TANH, ResRNN_RELU, ResLSTM, ResGRU], `type=str`, `default='ResLSTM'`.
- `--direction`: Type of the language model direction. You can select one from [left2right, right2left, both]. `type=str`, `default='left2right'`.
- `--wo_tok`: If = 1, the model ignores token embeddings. `type=bool`, `default=False`
- `--wo_char`: If = 1, the model ignores character embeddings. `type=bool`, `default=False`
- `--tok_emb`: The dimension size of word embeddings, `type=int`, `default=200`.
- `--char_emb`: The dimension size of character embeddings, `type=int`, `default=50`.
- `--char_kmin`: Minimum size of the kernel in the character encoder, `type=int`, `default=1`.
- `--char_kmax`: Maximum size of the kernel in the character encoder, `type=int`, `default=5`.
- `--tok_hid`: Number of hidden units of the token level rnn layer, `type=int`, `default=250`.
- `--char_hid`: Number of hidden units of the character level rnn layer, `type=int`, `default=50`.
- `--nlayers`: Number of layers. `type=int`, `default=2`.
- `--optim_type`: Type of the optimizer. `type=str`, `default='SGD'`.
- `--lr`: Initial learning rate. `type=float`, `default=20`.
- `--clip`: Gradient clipping. `type=float`, `default=0.25`.
- `--epochs`: Upper epoch limit. `type=int`, `default=40`.
- `--batch_size`: Batch size. `type=int`, `default=20`.
- `--cut_freq`: Cut off tokens in a corpus less than this value. `type=int`, `default=10`.
- `--max_vocab_size`: Cut off low-frequencey tokens in a corpus if the vocabulary size exceeds this value. `type=int`, `default=100000`.
- `--max_length`: Skip sentences longer than this value (-1: Infinite). `type=int`, `default=300`.
- `--dropout`: Dropout applied to layers (0 = no dropout). `type=float`, `default=0.2`.
- `--init_range`: Initialization range of the weights. `type=float`, `default=0.1`.
- `--tied`: If = 1, tie the word embedding and softmax weights. `type=bool`, `default=False`.
- `--seed`: Random seed. `type=int`, `default=1111`.
- `--cuda`: Use CUDA. `type=bool`, `default=False`.
- `--log-interval`: Report interval. `type=int`, `default=10`.
- `--pretrain`: Prefix to pretrained model. `type=str`, `default=''`.
- `--save`: Prefix to save the final model. `type=str`, `default='./models/model'`.
- `--dict`: Path to save the dictionary. `type=str`, `default='./models/dict.pkl'`.

### Evaluation (CLI)

Use `evaluation.py` for the evaluation.
The `evaluation.py` script accepts the following arguments:


- `--dict`: Location of the dictionary. `type=str`, `default='./models/dict.pkl'`.
- `--load`: Prefix to model files. `type=str`, `default='./models/model'`.
- `--outf`: Output file of preplexities for each sentence. `type=str`, `default='./output/ppl.txt'`.
- `--input_text`: Path to the input file. `type=str`, `default='./dataset/test.txt'`.
- `--seed`: Random seed. `type=int`, `default=1234`.
- `--server`: Run as a server. `type=bool`, `default=False`.
- `--host`: Host name. `type=str`, `default='localhost'`.
- `--port`: Access port number. `type=int`, `default='8888'`.
- `--cuda`: Use CUDA. `type=bool`, `default=False`.
- `--batch_size`: Batch size. `type=int`, `default=20`.
- `--log-interval`: Report interval. `type=int`, `default=100`.

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

- `--dict`: Location of the dictionary. `type=str`, `default='./models/dict.pkl'`.
- `--load`: Prefix to model files. `type=str`, `default='./models/model'`.
- `--outf`: Output file for generated text. `type=str`, `default='./output/generated.txt'`.
- `--sents`: Maximum number of sentencess to generate. `type=int`, `default='1000'`.
- `--words`: Maximum number of words to generate. `type=int`, `default='100'`.
- `--batch_size`, Batch size. `type=int`, `default=20`.
- `--seed`: Random seed. `type=int`, `default=1234`.
- `--cuda`: Use CUDA. `type=bool`, `default=False`.
- `--temperature`: Temperature - higher will increase diversity. type=float, default=1.0, help=''
- `--log-interval`: Report interval. `type=int`, `default=100`.

## LICENSE

See the file `LICENSE`.

## References

- <a name="cite1">[1]</a> Matthew E. Peters, Waleed Ammar, Chandra Bhagavatula and Russell Power. "**Semi-supervised sequence tagging with bidirectional language models.**" arXiv preprint arXiv:1705.00108 (2017).
- <a name="cite2">[2]</a> Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Černocký and Sanjeev Khudanpur. "**Recurrent neural network based language model.**" Eleventh annual conference of the international speech communication association. 2010.
- <a name="cite3">[3]</a> Junyoung Chung, Caglar Gulcehre, KyungHyun Cho and Yoshua Bengio. "**Empirical evaluation of gated recurrent neural networks on sequence modeling.**" arXiv preprint arXiv:1412.3555 (2014).
- <a name="cite4">[4]</a> Sundermeyer, Martin, Ralf Schlüter, and Hermann Ney. "**LSTM neural networks for language modeling.**" Thirteenth annual conference of the international speech communication association. 2012.
- <a name="cite5">[5]</a> Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. "**Deep residual learning for image recognition.**" Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- <a name="cite6">[6]</a> Yoon Kim, Yacine Jernite, David Sontag and Alexander M. Rush. "**Character-aware neural language models.**" Thirtieth AAAI Conference on Artificial Intelligence. 2016.
