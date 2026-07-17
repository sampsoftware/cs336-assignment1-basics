import cs336_basics.config as config
import logging
import cs336_basics.bpe_tokenizer_trainer as bpe_tokenizer_trainer
import argparse

logger = logging.getLogger(__name__)


def main():
    config.config_logging()

    parser = argparse.ArgumentParser("BPE Token Trainer")
    parser.add_argument(
        "--input-path", help="Path to the training corpus", default=config.get_data_dir(1) + "verysmall_tiny.txt"
    )
    parser.add_argument("--vocab-size", help="Size of the finished vocabulary", type=int, default=300)
    parser.add_argument(
        "--special-tokens", help="List of special token strings", type=list[bytes], default=["<|endoftext|>"]
    )
    args = parser.parse_args()

    bpe_tokenizer_trainer.train_tokenizer(args.input_path, args.vocab_size, args.special_tokens)


if __name__ == "__main__":
    main()
