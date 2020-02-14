
from typing import Dict, List, Any
import logging

import json
import re
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MultiLabelField, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("transformer_ddi")
class TransformerDDIDatasetReader(DatasetReader):
    """
    """
    def __init__(self,
                 pretrained_model: str,
                 lazy: bool = False,
                 do_lowercase: bool = None,
                 skip_id_regex: str = None,
                 model_type: str = None,
                 strip_args: bool = False,
                 ddi_labels: List[str] = None,  # Detailed labels in fixed order
                 max_pieces: int = 512,
                 sample: int = -1
                 ) -> None:
        super().__init__(lazy)
        if do_lowercase is None:
            do_lowercase = '-uncased' in pretrained_model
        self._tokenizer = PretrainedTransformerTokenizer(pretrained_model,
                                                         do_lowercase=do_lowercase,
                                                         start_tokens = [],
                                                         end_tokens = [])
        self._tokenizer_internal = self._tokenizer._tokenizer
        token_indexer = PretrainedTransformerIndexer(pretrained_model, do_lowercase=do_lowercase)
        self._token_indexers = {'tokens': token_indexer}
        self._sample = sample
        self._strip_args = strip_args
        self._skip_id_regex = skip_id_regex
        self._model_type = model_type
        self._max_pieces = max_pieces
        self._ddi_labels = ddi_labels
        if model_type is None:
            for model in ["roberta", "bert", "openai-gpt", "gpt2", "transfo-xl", "xlnet", "xlm"]:
                if model in pretrained_model:
                    self._model_type = model
                    break
        self._span_tokens = None
        # For each model pick 4 valid (rare) subword tokens
        if self._model_type == "roberta":
            self._span_tokens = ['Ġ[[', 'Ġ<<', 'Ġ{{', 'Ġ||', 'Ġ>>']
        elif self._model_type == "bert":
            self._span_tokens = ["[unused1]", "[unused2]", "[unused3]", "[unused4]"]
        else:
            raise ValueError(f"Model type {self._model_type} not yet support, needs span_tokens!")


    @overrides
    def _read(self, file_path):
        counter = self._sample + 1
        debug = 5
        with open(file_path, 'r') as data_file:
            for line in data_file:
                json_object = json.loads(line.strip())
                item_id = json_object.get('id')
                if self._skip_id_regex and item_id and re.match(self._skip_id_regex, item_id):
                    continue
                counter -= 1
                debug -= 1
                if counter == 0:
                    break
                if debug > 0:
                    logger.info(f"json_object = {json_object}")
                metadata = json_object.get('metadata', {})
                metadata['id'] = json_object.get('id')
                metadata['arg_ids'] = [json_object['arg1'].get('id', "NA"), json_object['arg2'].get('id', "NA")]
                label=json_object.get('ddi_label')
                if self._ddi_labels is not None and label:
                    ddi_type = json_object.get('ddi_type')
                    if not ddi_type in self._ddi_labels:
                        logger.warning(f"Skipping instance because ddi_type {ddi_type} not in ddi_labels")
                        continue
                    label = self._ddi_labels.index(ddi_type) + 1

                yield self.text_to_instance(
                    text=json_object.get('sentence'),
                    arg1_span=json_object.get('arg1')['span'][0],
                    arg2_span=json_object.get('arg2')['span'][0],
                    label=label,
                    metadata=metadata,
                    debug=debug
                )

    @overrides
    def text_to_instance(self,
                         text: str,
                         arg1_span: List[int],
                         arg2_span: List[int],
                         label: int = None,
                         metadata: Any = None,
                         debug: int = -1) -> Instance:

        metadata = metadata or {}
        metadata['sentence'] = text
        metadata['label'] = label

        text_tokens = self._tokenize_with_tags(text, arg1_span, arg2_span)
        fields = {
            'tokens': TextField(text_tokens, self._token_indexers),
        }
        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=True)

        if debug > 0:
            logger.info(f"text_tokens = {text_tokens}")
            logger.info(f"label= {label}")

        if metadata:
            fields['metadata'] = MetadataField(metadata)
        return Instance(fields)

    def _tokenize_with_tags(self, text, arg1_span, arg2_span):
        cls_token = Token(self._tokenizer_internal.cls_token)
        sep_token = Token(self._tokenizer_internal.sep_token)
        cls_token_at_end = bool(self._model_type in ['xlnet'])
        tokens = []
        tokens_at_end = 1
        if not cls_token_at_end:
            tokens.append(cls_token)
        else:
            tokens_at_end += 1
        if self._strip_args:
            tags = self._span_tokens[:2]
            loc_tags = list(zip([arg1_span, arg2_span], tags))
            # Assume non-overlapping arg spans
            loc_tags.sort(key = lambda x: x[0][0])
            last_index = 0
            for loc, tag in loc_tags:
                tokens += self._tokenizer.tokenize(text[last_index:loc[0]])
                tokens.append(Token(tag))
                last_index = loc[1]
            tokens += self._tokenizer.tokenize(text[last_index:])
        else:
            tags = self._span_tokens[:4]
            loc_tags = list(zip([*arg1_span, *arg2_span], tags))
            loc_tags.sort(key = lambda x: x[0])
            last_index = 0
            for loc, tag in loc_tags:
                tokens += self._tokenizer.tokenize(text[last_index:loc])
                tokens.append(Token(tag))
                last_index = loc
            tokens += self._tokenizer.tokenize(text[last_index:])
        tokens = tokens[:self._max_pieces - tokens_at_end - 1]
        tokens.append(sep_token)
        if cls_token_at_end:
            tokens.append(cls_token)
        return tokens
