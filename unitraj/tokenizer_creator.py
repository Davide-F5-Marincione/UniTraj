import os
import sentencepiece as spm
import json
from tqdm import tqdm
from transformers import T5Tokenizer
import argparse

tokenizer_basedir = "./checkpoint/tokenizer/"
os.makedirs(tokenizer_basedir, exist_ok=True)

modeltype = 'bpe'

filename = f"trajectoryspiece-{modeltype}"

spm.SentencePieceTrainer.train(
    input="tokenizer_input.txt",
    model_prefix=tokenizer_basedir + filename,
    vocab_size=1224,
    unk_id=3,
    bos_id=1,
    eos_id=2,
    pad_id=0,
    control_symbols="[PAD],[UNK],[CLS],[SEP],[MASK]",
    model_type=modeltype,  
    train_extremely_large_corpus=True,
    # use_all_vocab=True,
    character_coverage=1.0,  # 0.99995
)

vocab_file = tokenizer_basedir + filename + ".model"
sp_model = spm.SentencePieceProcessor()
sp_model.Load(vocab_file)

print("vocab size:", sp_model.vocab_size())

from sentencepiece import sentencepiece_model_pb2
m = sentencepiece_model_pb2.ModelProto()
with open(vocab_file, 'rb') as f:
    m.ParseFromString(f.read())

with open(tokenizer_basedir + filename + ".txt", 'w', encoding='utf-8') as f:
    f.write("# trainer_spec\n")
    f.write(m.trainer_spec.__repr__())
    m.normalizer_spec.precompiled_charsmap = b''
    f.write("# normalizer_spec\n")
    f.write(m.normalizer_spec.__repr__())
    f.write("# pieces\n")
    for piece in m.pieces:
        f.write(piece.piece + '\n')

vocab_file = tokenizer_basedir + filename + ".model"
tokenizer = T5Tokenizer(vocab_file=vocab_file)
tokenizer.save_pretrained(tokenizer_basedir + filename + '/')