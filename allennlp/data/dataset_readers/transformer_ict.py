
from typing import Dict, List, Any
import logging

import json
import os
import random
import re
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MultiLabelField, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("transformer_ict")
class TransformerICTDatasetReader(DatasetReader):
    """
    Dataset reader for Inverse Cloze Task. Includes generation of masked examples
    """
    def __init__(self,
                 pretrained_model: str,
                 lazy: bool = False,
                 skip_id_regex: str = None,
                 syntax: str = "raw",
                 model_type: str = None,
                 dataset_dir_out: str = None,
                 min_sentences: int = 4,
                 keep_sentence_probability: float = 0.1,
                 max_pieces: int = 512,
                 sample: int = -1
                 ) -> None:
        super().__init__(lazy)
        self._tokenizer = PretrainedTransformerTokenizer(pretrained_model, max_length=max_pieces, add_special_tokens=True)
        self._tokenizer_internal = self._tokenizer.tokenizer
        token_indexer = PretrainedTransformerIndexer(pretrained_model, max_length=max_pieces)
        self._token_indexers = {'tokens': token_indexer}

        self._max_pieces = max_pieces
        self._sample = sample
        self._syntax = syntax
        self._min_sentences = min_sentences
        self._keep_sentence_probability = keep_sentence_probability
        self._skip_id_regex = skip_id_regex
        self._dataset_dir_out = dataset_dir_out
        self._model_type = model_type
        if model_type is None:
            for model in ["roberta", "bert", "openai-gpt", "gpt2", "transfo-xl", "xlnet", "xlm"]:
                if model in pretrained_model:
                    self._model_type = model
                    break

    @overrides
    def _read(self, file_path: str):
        self._dataset_cache = None
        if self._dataset_dir_out is not None:
            self._dataset_cache = []
        instances = self._read_internal(file_path)
        if self._dataset_cache is not None:
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not os.path.exists(self._dataset_dir_out):
                os.mkdir(self._dataset_dir_out)
            output_file = os.path.join(self._dataset_dir_out, os.path.basename(file_path))
            logger.info(f"Saving contextualized dataset to {output_file}.")
            with open(output_file, 'w') as file:
                for d in self._dataset_cache:
                    file.write(json.dumps(d))
                    file.write("\n")
        return instances

    def _read_internal(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample + 1
        debug = 5

        with open(file_path, 'r') as data_file:
            logger.info("Reading instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                item_json = json.loads(line.strip())

                item_id = item_json.get("block_id", item_json.get("id", "NA"))
                if self._skip_id_regex and re.match(self._skip_id_regex, item_id):
                    continue

                counter -= 1
                debug -= 1
                if counter == 0:
                    break

                if debug > 0:
                    logger.info(item_json)

                if self._syntax == "raw":
                    sentences = item_json.get("sentences")
                    if len(sentences) < self._min_sentences:
                        continue
                    for idx, dropped_sentence in enumerate(sentences):
                        if dropped_sentence.startswith("%%"):
                            continue
                        if random.random() > self._keep_sentence_probability:
                            context = sentences[:idx] + sentences[idx+1:]
                        else:
                            context = sentences
                        context = " ".join(context)
                        sub_id = f"{item_id}-{idx}"

                        if self._dataset_cache is not None:
                            new_item = {'id': sub_id, 'context': context, 'sentence': dropped_sentence}
                            self._dataset_cache.append(new_item)

                        yield self.text_to_instance(
                            item_id=sub_id,
                            context=context,
                            sentence=dropped_sentence,
                            debug=debug)
                elif self._syntax == "proc":
                    context = item_json['context']
                    sentence = item_json['sentence']
                    yield self.text_to_instance(
                        item_id=item_id,
                        context=context,
                        sentence=sentence,
                        debug=debug)
                else:
                    raise ValueError(f"Unknown syntax {self._syntax}")

    def make_text_field(self, text: str):
        tokens = self._tokenizer.tokenize(text)
        return TextField(tokens, self._token_indexers)

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         context: str,
                         sentence: str,
                         additional_metadata: Dict[str, Any] = {},
                         debug: int = -1) -> Instance:
        # pylint: disable=arguments-differ
        context_field = self.make_text_field(context)
        sentence_field = self.make_text_field(sentence)
        fields = {
            'context': context_field,
            'sentence': sentence_field,
            'label': LabelField(-1, skip_indexing=True)   # Signals per-batch negative sampling
        }

        metadata = {
            "id": item_id,
            "context": context,
            "sentence": sentence
        }
        metadata.update(additional_metadata)

        if debug > 0:
            logger.info(f"context_tokens = {context_field.tokens}")
            logger.info(f"sentence_tokens = {sentence_field.tokens}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)