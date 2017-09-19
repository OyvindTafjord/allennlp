"""
The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ python -m allennlp.run evaluate --help
    usage: run [command] evaluate [-h] --archive_file ARCHIVE_FILE
                                --evaluation_data_file EVALUATION_DATA_FILE
                                [--cuda_device CUDA_DEVICE]

    Evaluate the specified model + dataset

    optional arguments:
    -h, --help            show this help message and exit
    --archive_file ARCHIVE_FILE
                            path to an archived trained model
    --evaluation_data_file EVALUATION_DATA_FILE
                            path to the file containing the evaluation data
    --cuda_device CUDA_DEVICE
                            id of GPU to use (if any)
"""
from typing import Dict, Any
import argparse
import json
import logging

import torch
import tqdm

from allennlp.common.util import prepare_environment
from allennlp.data import Dataset, Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn.util import arrays_to_variables

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Evaluate the specified model + dataset'''
    subparser = parser.add_parser(
            'evaluate', description=description, help='Evaluate the specified model + dataset')
    subparser.add_argument('--archive_file',
                           type=str,
                           required=True,
                           help='path to an archived trained model')
    subparser.add_argument('--evaluation_data_file',
                           type=str,
                           required=True,
                           help='path to the file containing the evaluation data')
    subparser.add_argument('--cuda_device',
                           type=int,
                           default=-1,
                           help='id of GPU to use (if any)')
    subparser.add_argument('--output_file',
                           type=str,
                           required=False,
                           help='output file for raw evaluation results')
    subparser.add_argument('--expand_vocabulary',
                           action='store_true',
                           help='expand vocabulary to include new words in evaluation data')

    subparser.set_defaults(func=evaluate_from_args)

    return subparser


def evaluate(model: Model,
             dataset: Dataset,
             iterator: DataIterator,
             cuda_device: int,
             output_file: str = None) -> Dict[str, Any]:
    model.eval()

    generator = iterator(dataset, num_epochs=1)
    logger.info("Iterating over dataset")
    generator_tqdm = tqdm.tqdm(generator, total=iterator.get_num_batches(dataset))
    file_handle = None
    if output_file:
        file_handle =  open(output_file, 'w')
    for batch in generator_tqdm:
        tensor_batch = arrays_to_variables(batch, cuda_device, for_training=False)
        model_output = model.forward(**tensor_batch)
        metrics = model.get_metrics()
        if file_handle:
            metadata = tensor_batch.get("metadata")
            if metadata:
                batch_size = len(metadata)
                for index, meta in enumerate(metadata):
                    res = {}
                    res["question_id"] = meta["question_id"]
                    for key, value in model_output.items():
                        if len(value) == batch_size and not isinstance(value, torch.autograd.Variable):
                            val = value[index]
                            res[key] = val
                    file_handle.write(json.dumps(res))
                    file_handle.write("\n")

        description = ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
        generator_tqdm.set_description(description)
    if file_handle:
        file_handle.close()

    return model.get_metrics()


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    prepare_environment(config)

    # Load the evaluation data
    dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.evaluation_data_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    dataset = dataset_reader.read(evaluation_data_path)
    dataset.index_instances(model.vocab)

    if args.expand_vocabulary:
        vocab_new = Vocabulary.from_params(config.get("vocabulary", {}), dataset)
        vocab_size_old = model.vocab.get_vocab_size("tokens")
        for i in range(0, vocab_new.get_vocab_size("tokens")):
            model.vocab.add_token_to_namespace(vocab_new.get_token_from_index(i), "tokens")
        vocab_size_new = model.vocab.get_vocab_size("tokens")
        if vocab_size_new > vocab_size_old:
            logger.info("Adding %d new tokens to vocabulary", vocab_size_new - vocab_size_old)
            token_embedder = model._text_field_embedder._token_embedders['tokens']
            params_tfe = config.get("model").get("text_field_embedder").get("tokens")
            token_embedder.extend_vocab(model.vocab, params_tfe)

    iterator = DataIterator.from_params(config.pop("iterator"))

    metrics = evaluate(model, dataset, iterator, args.cuda_device, args.output_file)

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    return metrics
