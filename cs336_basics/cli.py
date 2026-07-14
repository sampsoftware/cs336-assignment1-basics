import cs336_basics.config as config
import logging
import cs336_basics.bpe_tokenizer_trainer as bpe_tokenizer_trainer

logger = logging.getLogger(__name__)

def main():
    config.config_logging()
    bpe_tokenizer_trainer.train_tokenizer("verysmall_tiny.txt",4,{'<|endoftext|>'})
    #train_tokenizer(input_file,vocab_size,)
    logger.info("COMPLETE")


if __name__ == "__main__":
    main()