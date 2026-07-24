import cs336_basics.util as util
import logging
import cs336_basics.bpe_tokenizer_trainer as bpe_tokenizer_trainer
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    util.config_logging()

    parser = argparse.ArgumentParser("BPE Token Trainer")
    parser.add_argument(
        "--input-path", help="Path to the training corpus", default=util.get_data_dir(1) + "verysmall_tiny.txt"
    )
    parser.add_argument("--vocab-size", help="Size of the finished vocabulary", type=int, default=300)
    parser.add_argument(
        "--special-tokens", help="List of special token strings", type=list[bytes], default=["<|endoftext|>"]
    )
    parser.add_argument(
        "--outfile", help="Output file, excluding extension; creates map and merge files", type=str, default=None
    )
    args = parser.parse_args()

    if args.outfile is None:
        args.outfile = args.input_path

    bpe_tokenizer_trainer.train_tokenizer(args.input_path, args.vocab_size, args.special_tokens, args.outfile)


if __name__ == "__main__":
    main()
