import sys
import errno
import json
import os
from argparse import ArgumentParser
sys.path.insert(1, 'py-bindings')
from squad import SQUADConverter

def get_samples(test_file, vocab_file, output_dir):
    print("Test file:", test_file)
    print("Vocab file:", vocab_file)
    print("Output dir:", output_dir)
    max_seq_length = 384
    max_query_length = 64
    doc_stride = 128
    lower_case = False
    sqd = SQUADConverter(test_file, vocab_file, max_seq_length, max_query_length, doc_stride, lower_case)
    samples = sqd.convert()
    # Dump samples to json
    print("--Dumping examples to json--")
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir + "/squad_examples.json"
    c = 0 
    with open(output_file, 'w', encoding='utf-8') as fid:
        json.dump({'samples':samples}, fid, ensure_ascii=False, indent=4)
    return c

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str, help="Path to squad test json file", required=True)
    parser.add_argument("--vocab_file", type=str, help="Path to vocab.txt file", required=True)
    parser.add_argument("--max_seq_length", type=int, help="Max sequence length", default=384)
    parser.add_argument("--max_query_length", type=int, help="Max query length", default=64)
    parser.add_argument("--doc_stride", type=int, help="Document stride", default=128)
    parser.add_argument("--lower_case", type=bool, help="Lower case", default=1)
    parser.add_argument("--output_dir", type=str, help="Output directory for saved json", default="samples_cache")
    
    return parser.parse_args()


def main():

    args = get_arguments()
    if not os.path.isfile(args.test_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.test_file)

    if not os.path.isfile(args.vocab_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.vocab_file)

    sqd = SQUADConverter(args.test_file, args.vocab_file, args.max_seq_length, args.max_query_length, args.doc_stride, args.lower_case)

    # Convert examples
    print("--Reading samples--")
    samples = sqd.convert()

    # Dump samples ot json
    print("--Dumping examples to json--")
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = args.output_dir + "/squad_examples.json"

    with open(output_file, 'w', encoding='utf-8') as fid:
        json.dump({'samples':samples}, fid, ensure_ascii=False, indent=4)


if __name__=="__main__":
    main()
