import cs336_basics.config as config
import logging
import cs336_basics.bpe_tokenizer_trainer as bpe_tokenizer_trainer

logger = logging.getLogger(__name__)

def main():
    config.config_logging()
    input_path = config.get_data_dir(1) + "verysmall_tiny.txt"
    special_tokens = []
    special_tokens.append('<|endoftext|>')
    vocab_size = 256 + len(special_tokens) + 20

    bpe_tokenizer_trainer.train_tokenizer(input_path,vocab_size,special_tokens)
    #train_tokenizer(input_file,vocab_size,)
    logger.info("COMPLETE")


if __name__ == "__main__":
    main()