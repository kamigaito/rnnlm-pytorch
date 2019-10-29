import argparse
import requests
import urllib.parse
import json

parser = argparse.ArgumentParser(description='PyTorch Language Model')

parser.add_argument('--host', type=str, default='http://localhost',
                    help='host name')
parser.add_argument('--port', type=int, default='8888',
                    help='access port number')
parser.add_argument('--path', type=str, default='lm',
                    help='path')

if __name__ == '__main__':
    args = parser.parse_args()
    url = args.host + ":" + str(args.port) + "/" + args.path
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "sentences" : [
            "This is a test post .",
            "Can you calculate a perplexity of this text ?",
            "If you encounter an error , please report it to us .",
            "This is a test post .",
            "Can you calculate a perplexity of this text ?",
            "If you encounter an error , please report it to us .",
            "This is a test post .",
            "Can you calculate a perplexity of this text ?",
            "If you encounter an error , please report it to us .",
            "This is a test post .",
            "Can you calculate a perplexity of this text ?",
            "If you encounter an error , please report it to us ."
        ]
    }
    r = urllib.request.Request(url, json.dumps(data).encode(), headers)
    with urllib.request.urlopen(r) as res:
        body = json.load(res)
        print(body)
