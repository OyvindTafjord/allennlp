#! /usr/bin/env python

"""
Basic commands for various processing associated with semantic retrieval
"""

import json
import gzip
import logging
import re
import os
import subprocess
import sys
import argparse

from allennlp.data import DatasetReader, Instance
from allennlp.data.batch import Batch
from allennlp.models.archival import load_archive
from allennlp.nn import util

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def get_context_embedding(context, model, dataset_reader, cuda_device):
    context_instance = Instance({"context": dataset_reader.make_text_field(context)})
    batch = Batch([context_instance])
    batch.index_instances(model.vocab)
    model_input = util.move_to_device(batch.as_tensor_dict(), cuda_device)
    res = model.context_projection(**model_input)
    return res[0].detach().cpu().tolist()


def add_embeddings(args, max_num=-1):
    model_file = args.model
    in_file = args.input
    out_file = args.output
    cuda_device = args.cuda_device

    logger.info(f"Loading model {model_file}...")
    archive = load_archive(model_file, cuda_device=cuda_device)
    model = archive.model
    model.eval()
    dataset_reader = DatasetReader.from_params(archive.config['dataset_reader'])
    if in_file.endswith(".gz"):
        file_open = gzip.open
    else:
        file_open = open
    logger.info(f"Iterating through {in_file}...")
    with file_open(in_file, 'r') as input, open(out_file, 'w') as output:
        counter = 0
        for line in input:
            if counter == max_num:
                break
            if counter % 1000 == 0:
                logger.info(f"Item num: {counter}")
            counter += 1
            block = json.loads(line)
            embedding = get_context_embedding(" ".join(block['sentences']), model, dataset_reader, cuda_device)
            block['embedding'] = embedding
            output.write(json.dumps(block))
            output.write("\n")
    logger.info(f"Done processing {counter} items!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Semantic Retrieval utils")
    parser.add_argument('--max-num', type=int, default=-1, required=False, help='Max number of lines processed')
    subparsers = parser.add_subparsers()
    parser_embed = subparsers.add_parser('embed')
    parser_embed.add_argument('--model', type=str, required=True, help='Model name or dataset id.')
    parser_embed.add_argument('--input', type=str, required=True, help='Input file')
    parser_embed.add_argument('--output', type=str, required=True, help='Output file')
    parser_embed.add_argument('--cuda-device', type=int, default=-1, required=False, help='CUDA device')
    parser_embed.set_defaults(func=add_embeddings)

    args = parser.parse_args()
    try:
        args.func(args, args.max_num)
    except:
        raise ValueError("Missing command!")
