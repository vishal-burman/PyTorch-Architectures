###############################################################
# Sample implementation to train BPE (like GPT2) from scratch
###############################################################

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path(".").glob("*.txt")] # list of paths for dataset
tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, vocab_size=50, min_frequency=1, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

# Saving the model
tokenizer.save_model(".", "scratch_tokenizer")
