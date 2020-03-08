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


ES_MAPPING_BLOCKS = '''
    {
      "mappings": {
          "properties": {
            "type": {
              "type": "keyword"
            },
            "docId": {
              "type": "keyword"
            },
            "secId": {
               "type": "integer"
            },
            "paraId": {
               "type": "integer"
            }, 
            "title": {
              "analyzer": "snowball",
              "type": "text"
            },
            "header": {
              "analyzer": "snowball",
              "type": "text"
            },
            "embedding": {
              "type": "dense_vector",
              "dims": 128
            },
            "text": {
              "analyzer": "snowball",
              "type": "text",
              "fields": {
                "raw": {
                  "type": "keyword"
                }
              }
            },
            "tags": {
              "type": "keyword"
            }
          }
      }
    }'''


def make_es_doc(block, index_name):
    sentences = block['sentences']
    headers = sentences[0].strip("% ").split(" % ")
    text = " ".join(sentences[1:])
    doc = {
        '_op_type': 'create',
        '_index': index_name,
        '_id': block['block_id'],
        '_source': {
            'type': "paragraph",
            'docId': block["doc_id"],
            'secId': block["sec_id"],
            'paraId': block["para_id"],
            'title': headers[0],
            'subsection': " :: ".join(headers),
            'text': text,
            'embedding': block['embedding']
        }
    }
    return doc


def add_index(args, max_num=-1):
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    batch_size = 1000

    index_name = args.index
    in_file = args.input

    es = Elasticsearch()
    info = es.info()
    if not isinstance(info, dict):
        logger.error(f"Not able to connect to Elasticsearch on locahost:9200!")
        return None
    logger.info(f"Connected to Elasticsearch: {info}")
    res = es.indices.create(index=index_name, ignore=400, body=ES_MAPPING_BLOCKS)
    logger.info(f"Attempted creation of index {index_name}: {res}")

    if in_file.endswith(".gz"):
        file_open = gzip.open
    else:
        file_open = open
    with file_open(in_file, 'r') as input:
        batch = []
        counter = 0
        for line in input:
            if counter == max_num:
                break
            if counter % 1000 == 0:
                logger.info(f"Item num: {counter}")
            if len(batch) == batch_size:
                bulk(es, batch)
                batch = []
            counter += 1
            block = json.loads(line)
            batch.append(make_es_doc(block, index_name))
        if len(batch) > 0:
            logger.info(f"Indexing final batch")
            bulk(es, batch)
    logger.info(f"Done indexing {counter} items!")


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

    parser_index = subparsers.add_parser('add_index')
    parser_index.add_argument('--index', type=str, required=True, help='Which index to create.')
    parser_index.add_argument('--input', type=str, required=True, help='Input file with embeddings')
    parser_index.set_defaults(func=add_index)

    args = parser.parse_args()
    try:
        func = args.func
    except:
        raise ValueError("Missing command!")

    args.func(args, args.max_num)
