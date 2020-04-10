"""
The ``evaluate_custom`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ python -m allennlp.run evaluate_custom --help
    usage: run [command] evaluate_custom [-h] --archive_file ARCHIVE_FILE
                                --evaluation_data_file EVALUATION_DATA_FILE
                                [--cuda_device CUDA_DEVICE]

    Evaluate the specified model + dataset

    optional arguments:
    -h, --help            show this help message and exit
    --archive_file ARCHIVE_FILE
                            path to an archived trained model
    --evaluation_data_file EVALUATION_DATA_FILE
                            path to the file containing the evaluation data
    --output_file OUTPUT_FILE
                            path to optional output file with detailed predictions
    --cuda_device CUDA_DEVICE
                            id of GPU to use (if any)
"""
from typing import Dict, Any, Iterable
import argparse
from contextlib import ExitStack
import json
import logging
from overrides import overrides
import torch

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import prepare_environment, sanitize
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

Tqdm.set_slower_interval(True)

@Subcommand.register("evaluate_custom")
class EvaluateCustom(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset with optional output'''
        subparser = parser.add_parser('evaluate_custom',
                                      description=description,
                                      help='Evaluate the specified model + dataset with optional output')
        subparser.add_argument('archive_file',
                               type=str,
                               help='path to an archived trained model')
        subparser.add_argument('--evaluation-data-file',
                               type=str,
                               required=True,
                               help='path to the file containing the evaluation data')
        subparser.add_argument('--output-file',
                               type=str,
                               required=False,
                               help='output file for raw evaluation results')
        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')
        subparser.add_argument('--output-metrics-file',
                               type=str,
                               help='output file for metrics')
        subparser.add_argument('--metadata-fields',
                               type=str,
                               required=False,
                               help='metadata fields to output, separated by commas (no spaces)')
        subparser.add_argument('--cuda-device',
                               type=int,
                               default=-1,
                               help='id of GPU to use (if any)')
        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate_custom(model: Model,
                    instances: Iterable[Instance],
                    iterator: DataIterator,
                    cuda_device: int,
                    output_file: str = None,
                    metadata_fields: str = None) -> Dict[str, Any]:

    check_for_gpu(cuda_device)
    Tqdm.set_slower_interval(True)
    with torch.no_grad():
        model.eval()
        model.get_metrics(reset=True)
        generator = iterator(instances, num_epochs=1, shuffle=False)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(generator, total=iterator.get_num_batches(instances))
        metadata_fields_list = metadata_fields.split(",") if metadata_fields is not None else []
        with ExitStack() as stack:
            if output_file is None:
                file_handle = None
            else:
                file_handle = stack.enter_context(open(output_file, 'w'))
            for batch in generator_tqdm:
                batch = util.move_to_device(batch, cuda_device)
                model_output = model.forward(**batch)
                metrics = model.get_metrics()
                model_output = model.decode(model_output)
                if file_handle:
                    _persist_data(file_handle, batch.get("metadata"), model_output, metadata_fields_list)
                description = ', '.join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
                generator_tqdm.set_description(description)

        return model.get_metrics()


def _persist_data(file_handle, metadata, model_output, metadata_fields_list) -> None:
    if metadata:
        batch_size = len(metadata)
        for index, meta in enumerate(metadata):
            res = {}
            # res["id"] = meta.get("id", "NA")
            # res["question"] = meta.get("question", "NA")
            for key, value in meta.items():
                if key in metadata_fields_list:
                    res[key] = sanitize(value)
            # res["slot_values_text_scrambled"] = meta.get("slot_values_text_scrambled", [])
            # We persist model output which matches batch_size in length and is not a Variable
            for key, value in model_output.items():
                key_out = key
                if key in res:
                    key_out = key+"-model"
                if key in metadata_fields_list:
                    try:
                        value_len = len(value)
                    except:
                        value_len = -1
                    if value_len == batch_size:
                        val = value[index]
                        res[key_out] = sanitize(val)
                    else:
                        res[key_out] = sanitize(value)
                        res["batch_index"] = index
            file_handle.write(json.dumps(res))
            file_handle.write("\n")


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.evaluation_data_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    iterator_params = config.pop("validation_iterator", None)
    if iterator_params is None:
        iterator_params = config.pop("iterator")
    iterator = DataIterator.from_params(iterator_params)
    iterator.index_with(model.vocab)

    metrics = evaluate_custom(model, instances, iterator, args.cuda_device,
                              args.output_file, args.metadata_fields)

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    output_file = args.output_metrics_file
    if output_file:
        with open(output_file, "w") as file:
            json.dump(metrics, file, indent=4)
    return metrics
