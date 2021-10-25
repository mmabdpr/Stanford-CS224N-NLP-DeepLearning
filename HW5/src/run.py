import argparse
import random
from typing import Union

import torch
from tqdm import tqdm

from dataset import CharCorruptionDataset, NameDataset
from model import GPT, GPTConfig, GPT1Config, Block, CustomLayerNorm
from trainer import Trainer, TrainerConfig
import utils

random.seed(0)

arg_p = argparse.ArgumentParser()
arg_p.add_argument('function',
                   help="Whether to pretrain, finetune or evaluate a model",
                   choices=["pretrain", "finetune", "evaluate"])
arg_p.add_argument('variant',
                   help="Which variant of the model to run ('vanilla' or 'synthesizer')",
                   choices=["vanilla", "synthesizer"])
arg_p.add_argument('pretrain_corpus_path',
                   help="Path of the corpus to pretrain on", default=None)
arg_p.add_argument('--reading_params_path',
                   help="If specified, path of the model to load before finetuning/evaluation",
                   default=None)
arg_p.add_argument('--writing_params_path',
                   help="Path to save the model after pretraining/finetuning", default=None)
arg_p.add_argument('--finetune_corpus_path',
                   help="Path of the corpus to finetune on", default=None)
arg_p.add_argument('--eval_corpus_path',
                   help="Path of the corpus to evaluate on", default=None)
arg_p.add_argument('--outputs_path', default=None)
args = arg_p.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# Keep the block size 128
# Why is the pretraining corpus always required (even if we're not pretraining?)
# It's because we're using it as a hack to always have the same vocabulary
# (that is, the same mapping from character to integer, and we build the 
# vocab from the pretraining corpus.)
block_size = 128
text = open(args.pretrain_corpus_path).read()
pretrain_dataset = CharCorruptionDataset(text, block_size)

# We don't suggest you change these hyperparameters, as they're known to work.
# use them for both the vanilla and the synthesizer models
m_conf = GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
                   n_layer=4, n_head=8, n_embd=256)

"""
Don't change above here; write your code below
"""

if args.variant == 'vanilla':
    # TODO [part c]: Make some model here
    pass
elif args.variant == 'synthesizer':
    # TODO [part g]: Make some other model here
    m_conf.additive = True
else:
    raise Exception("Invalid variant")

model = GPT(m_conf).to(device)

# From here on, your code should be identical independent of which
# variant (vanilla or synthesizer) has been chosen.

if args.function == 'pretrain':
    assert args.pretrain_corpus_path is not None
    assert args.writing_params_path is not None
    # TODO [part f]:
    # - Given:
    #     1. A corpus specified in args.pretrain_corpus_path
    #     2. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. Pretrain the model on this corpus
    #     2. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters for pretraining:
    #     max_epochs=650
    #     batch_size=128
    #     learning_rate=6e-3
    #     lr_decay=True
    #     warmup_tokens=512*20
    #     final_tokens=200*len(pretrain_dataset)*block_size
    #     num_workers=4
    t_conf = TrainerConfig(max_epochs=650, batch_size=128, learning_rate=6e-3,
                           lr_decay=True, warmup_tokens=512 * 20,
                           final_tokens=200*len(pretrain_dataset)*block_size,
                           num_workers=4, checkpoint_path=args.writing_params_path)
    trainer = Trainer(model, pretrain_dataset, None, t_conf)
    trainer.train()
elif args.function == 'finetune':
    assert args.writing_params_path is not None
    assert args.finetune_corpus_path is not None
    # TODO [part c] [part f]:
    # - Given:
    #     1. A finetuning corpus specified in args.finetune_corpus_path
    #     2. A path args.reading_params_path containing pretrained model
    #         parameters, or None if finetuning without a pretrained model
    #     3. An output path args.writing_params_path for the model parameters
    # - Goals:
    #     1. If args.reading_params_path is specified, load these parameters
    #         into the model
    #     2. Finetune the model on this corpus
    #     3. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters:
    #     Hyperparameters for finetuning WITHOUT a pretrained model:
    #         max_epochs=75
    #         batch_size=256
    #         learning_rate=6e-4
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #     Hyperparameters for finetuning WITH a pretrained model:
    #         max_epochs=10
    #         batch_size=256
    #         learning_rate=6e-4
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    if args.reading_params_path is None:
        t_conf = TrainerConfig(max_epochs=75, batch_size=256, learning_rate=6e-4,
                               lr_decay=True, warmup_tokens=512 * 20,
                               final_tokens=200*len(pretrain_dataset)*block_size,
                               num_workers=4, checkpoint_path=args.writing_params_path)
    else:
        model.load_state_dict(torch.load(args.reading_params_path))
        t_conf = TrainerConfig(max_epochs=10, batch_size=256, learning_rate=6e-4,
                               lr_decay=True, warmup_tokens=512 * 20,
                               final_tokens=200 * len(pretrain_dataset) * block_size,
                               num_workers=4, checkpoint_path=args.writing_params_path)
    text = open(args.finetune_corpus_path).read()
    finetune_dataset = NameDataset(pretraining_dataset=pretrain_dataset, data=text)
    trainer = Trainer(model, finetune_dataset, None, t_conf)
    trainer.train()
elif args.function == 'evaluate':
    assert args.outputs_path is not None
    assert args.reading_params_path is not None
    assert args.eval_corpus_path is not None
    model.load_state_dict(torch.load(args.reading_params_path))
    correct = 0
    total = 0
    with open(args.outputs_path, 'w') as f_out:
        predictions = []
        for line in tqdm(open(args.eval_corpus_path)):
            x = line.split('\t')[0]
            x = x + '⁇'
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None, ...].to(device)
            pred = utils.sample(model, x, 32, sample_from_dist=False)[0]
            completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
            pred = completion.split('⁇')[1]
            predictions.append(pred)
            f_out.write(pred + '\n')
        total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
    if total > 0:
        print('Correct: {} out of {}: {}%'.format(correct, total, correct / total * 100))
    else:
        print('Predictions written to {}; no targets provided'
              .format(args.outputs_path))
