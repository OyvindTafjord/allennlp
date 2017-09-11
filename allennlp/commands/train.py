"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ python -m allennlp.run train --help
   usage: run [command] train [-h] -s SERIALIZATION_DIR param_path

   Train the specified model on the specified dataset.

   positional arguments:
   param_path            path to parameter file describing the model to be trained

   optional arguments:
    -h, --help            show this help message and exit
    -s SERIALIZATION_DIR, --serialization_dir SERIALIZATION_DIR
                            directory in which to save the model and its logs
"""

import argparse
import json
import logging
import os
import random
import sys
from copy import deepcopy
from typing import Any, Dict, Union

import numpy
import torch

from allennlp.common.checks import log_pytorch_version_info, ensure_pythonhashseed_set
from allennlp.common.params import Params
from allennlp.common.tee_logger import TeeLogger
from allennlp.data import Dataset, Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model
from allennlp.models.model import Model
from allennlp.training.trainer import Trainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Train the specified model on the specified dataset.'''
    subparser = parser.add_parser(
            'train', description=description, help='Train a model')
    subparser.add_argument('param_path',
                           type=str,
                           help='path to parameter file describing the model to be trained')
    subparser.add_argument('-s', '--serialization_dir',
                           type=str,
                           required=True,
                           help='directory in which to save the model and its logs')
    subparser.add_argument('--restore',
                           action='store_true',
                           help='restore from checkpoint')
    subparser.add_argument('--expand_vocabulary',
                           action='store_true',
                           help='expand vocabulary when restoring from checkpoint')
    subparser.set_defaults(func=_train_model_from_args)

    return subparser

def prepare_environment(params: Union[Params, Dict[str, Any]]):
    """
    Sets random seeds for reproducible experiments. This may not work as expected
    if you use this from within a python project in which you have already imported Pytorch.
    If you use the scripts/run_model.py entry point to training models with this library,
    your experiments should be reasonably reproducible. If you are using this from your own
    project, you will want to call this function before importing Pytorch. Complete determinism
    is very difficult to achieve with libraries doing optimized linear algebra due to massively
    parallel execution, which is exacerbated by using GPUs.

    Parameters
    ----------
    params: Params object or dict, required.
        A ``Params`` object or dict holding the json parameters.
    """
    seed = params.pop("random_seed", 13370)
    numpy_seed = params.pop("numpy_seed", 1337)
    torch_seed = params.pop("pytorch_seed", 133)

    if seed is not None:
        random.seed(seed)
    if numpy_seed is not None:
        numpy.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        # Seed all GPUs with the same seed if available.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

    log_pytorch_version_info()


def _train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(args.param_path, args.serialization_dir, args.restore, args.expand_vocabulary)


def train_model_from_file(parameter_filename: str, serialization_dir: str, restore=False, expand_vocabulary=False) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    param_path: str, required.
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """
    # We need the python hashseed to be set if we're training a model
    ensure_pythonhashseed_set()

    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename)
    return train_model(params, serialization_dir, restore, expand_vocabulary)


def train_model(params: Params, serialization_dir: str, restore=False, expand_vocabulary=False) -> Model:
    """
    This function can be used as an entry point to running models in AllenNLP
    directly from a JSON specification using a :class:`Driver`. Note that if
    you care about reproducibility, you should avoid running code using Pytorch
    or numpy which affect the reproducibility of your experiment before you
    import and use this function, these libraries rely on random seeds which
    can be set in this function via a JSON specification file. Note that this
    function performs training and will also evaluate the trained model on
    development and test sets if provided in the parameter json.

    Parameters
    ----------
    params: Params, required.
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """
    prepare_environment(params)

    os.makedirs(serialization_dir, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(serialization_dir, "stdout.log"), sys.stdout)  # type: ignore
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), sys.stderr)  # type: ignore
    handler = logging.FileHandler(os.path.join(serialization_dir, "python_logging.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    serialization_params = deepcopy(params).as_dict(quiet=True)
    with open(os.path.join(serialization_dir, "model_params.json"), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

    # Now we begin assembling the required parts for the Trainer.
    dataset_reader = DatasetReader.from_params(params.pop('dataset_reader'))

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = dataset_reader.read(validation_data_path)
        combined_data = Dataset(train_data.instances + validation_data.instances)
    else:
        validation_data = None
        combined_data = train_data

    params_tfe = None

    if restore:
        logger.info("Restoring vocabulary from saved model")
        vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
        if expand_vocabulary:
            vocab_new = Vocabulary.from_params(params.pop("vocabulary", {}), combined_data)
            vocab_size_old = vocab.get_vocab_size("tokens")
            for i in range(0, vocab_new.get_vocab_size("tokens")):
                vocab.add_token_to_namespace(vocab_new.get_token_from_index(i), "tokens")
            vocab_size_new = vocab.get_vocab_size("tokens")
            if (vocab_size_new > vocab_size_old):
                logger.info("Adding %d new tokens to vocabulary", vocab_size_new - vocab_size_old)
                params_tfe = params.get("model").get("text_field_embedder").get("tokens")
                vocab.save_to_files(os.path.join(serialization_dir, "vocabulary_new"))

    else:
        vocab = Vocabulary.from_params(params.pop("vocabulary", {}), combined_data)
        vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    model = Model.from_params(vocab, params.pop('model'))
    iterator = DataIterator.from_params(params.pop("iterator"))

    train_data.index_instances(vocab)
    if validation_data:
        validation_data.index_instances(vocab)

    trainer_params = params.pop("trainer")
    trainer = Trainer.from_params(model,
                                  serialization_dir,
                                  iterator,
                                  train_data,
                                  validation_data,
                                  trainer_params)
    params.assert_empty('base train command')
    trainer._params_tfe = params_tfe
    trainer.train()

    # Now tar up results
    archive_model(serialization_dir)

    return model
