# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
import random

import torch
from tqdm import tqdm

import utils
from dataset import CharCorruptionDataset
from model import GPT, GPTConfig

random.seed(0)

pretrain_corpus_path = "wiki.txt"
outputs_path = "vanilla.nopretrain.test.predictions"
reading_params_path = "vanilla.model.params"
eval_corpus_path = "birth_test_inputs.tsv"

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

block_size = 128
text = open(pretrain_corpus_path).read()
pretrain_dataset = CharCorruptionDataset(text, block_size)

m_conf = GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
                   n_layer=4, n_head=8, n_embd=256)

model = GPT(m_conf).to(device)

model.load_state_dict(torch.load(reading_params_path))
correct = 0
total = 0
with open(outputs_path, 'w') as f_out:
    predictions = []
    for line in tqdm(open(eval_corpus_path)):
        x = line.split('\t')[0]
        x = x + '⁇'
        x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None, ...].to(device)
        pred = utils.sample(model, x, 32, sample_from_dist=False)[0]
        completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
        pred = completion.split('⁇')[1]
        predictions.append(pred)
        f_out.write(pred + '\n')
    true_places = ['London' for _ in range(len(predictions))]
    total = len(true_places)
    assert total == len(predictions)
    correct = len(list(filter(lambda xx: xx[0] == xx[1],
                              zip(true_places, predictions))))
if total > 0:
    print('Correct: {} out of {}: {}%'.format(correct, total, correct / total * 100))
else:
    print('Predictions written to {}; no targets provided'
          .format(outputs_path))
