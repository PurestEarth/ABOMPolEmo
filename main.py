import argparse
import os
import codecs


def main(args):
    print('xD')
    paths = codecs.open(args.input, "r", "utf8").readlines()
    for path in enumerate(paths):
        print(path)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert set of IOB files into a single json file in PolEval 2018 NER format')
    parser.add_argument('--input', required=True, metavar='PATH', help='path to input')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except ValueError as er:
        print("[ERROR] %s" % er)