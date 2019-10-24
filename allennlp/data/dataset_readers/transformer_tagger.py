from typing import Dict, List, Any
import itertools
import json
import logging
import numpy
import os
import re

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.document_retriever import combine_sentences, list_sentences, DocumentRetriever
from allennlp.data.fields import ArrayField, Field, TextField, LabelField
from allennlp.data.fields import ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# TagSpanType = ((int, int), str)


NONE_TAG = 0

@DatasetReader.register("transformer_tagger")
class TransformerTaggerReader(DatasetReader):
    """

    Parameters
    ----------
    """

    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512,
                 syntax: str = "esst",
                 skip_id_regex: str = None,
                 tag_score_cutoff: float = 0.5,
                 context_strip_sep: str = None,
                 answer_only: bool = False,
                 context_type: str = None,
                 add_prefix: Dict[str, str] = None,
                 dataset_dir_out: str = None,
                 model_type: str = None,
                 do_lowercase: bool = None,
                 word_tokenizer: Tokenizer = None,
                 sample: int = -1) -> None:
        super().__init__()
        if do_lowercase is None:
            do_lowercase = '-uncased' in pretrained_model

        self._tokenizer = PretrainedTransformerTokenizer(pretrained_model,
                                                         do_lowercase=do_lowercase,
                                                         start_tokens = [],
                                                         end_tokens = [])
        self._tokenizer_internal = self._tokenizer._tokenizer
        self._word_tokenizer = word_tokenizer or WordTokenizer()
        token_indexer = PretrainedTransformerIndexer(pretrained_model, do_lowercase=do_lowercase)
        self._token_indexers = {'tokens': token_indexer}

        self._max_pieces = max_pieces
        self._tag_score_cutoff = tag_score_cutoff
        self._sample = sample
        self._syntax = syntax
        self._answer_only = answer_only
        self._skip_id_regex = skip_id_regex
        self._context_type = context_type
        self._context_strip_sep = context_strip_sep
        self._dataset_dir_out = dataset_dir_out
        self._model_type = model_type
        self._add_prefix = add_prefix or {}
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

                item_id = item_json["id"]
                if self._skip_id_regex and re.match(self._skip_id_regex, item_id):
                    continue

                counter -= 1
                debug -= 1
                if counter == 0:
                    break

                if debug > 0:
                    logger.info(item_json)

                if self._syntax == "arc":
                    item_json["text"] = item_json["question"]["stem"]

                text = item_json["text"]
                tags = item_json.get("tags")
                context = None
                if self._context_type is not None:
                    raw_text = item_json["raw_text"]
                    context = re.findall("\\([A1]\\).*", raw_text)
                    context = context[0] if context else None

                yield self.text_to_instance(
                    item_id=item_id,
                    text=text,
                    context=context,
                    tags=tags,
                    debug=debug)

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         text: str,
                         context: str = None,
                         tags: List[Dict[str, Any]] = None,
                         additional_metadata: Dict[str, Any] = {},
                         debug: int = -1) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        word_tokens = self._word_tokenizer.tokenize(text)

        text_tokens, word_offsets = self.expand_subword_tokens(word_tokens, context)
        tokens_field = TextField(text_tokens, self._token_indexers)
        word_offsets_field = ArrayField(numpy.array(word_offsets, dtype=numpy.long), padding_value=-1, dtype=numpy.long)

        tag_per_word_token = None
        if tags is not None:
            tag_per_word_token = self._align_token_tags(word_tokens, tags)
            tags_field = ListField([LabelField(tag, skip_indexing=True) for tag in tag_per_word_token])
            fields['tags'] = tags_field

        if debug > 0:
            logger.info(f"text_tokens = {text_tokens}")
            logger.info(f"word_offsets = {word_offsets}")
            logger.info(f"tags = {tag_per_word_token}")

        fields['tokens'] = tokens_field
        fields['word_offsets'] = word_offsets_field

        metadata = {
            "id": item_id,
            "text": text,
            "word_tokens": word_tokens,
            "tags": tag_per_word_token,
            "word_offsets": word_offsets,
            "tokens": [x.text for x in text_tokens]
        }
        metadata.update(additional_metadata)

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def _align_token_tags(self, word_tokens, tags):
        tag_per_word_token = []
        tags_filtered = [t for t in tags if t['score']/t['max_score'] >= self._tag_score_cutoff]
        for token in word_tokens:
            token_pos = token.idx
            current_tags = []
            for tag in tags_filtered:
                if tag['span'][0] <= token_pos < tag['span'][1]:
                    current_tags.append(tag['label'])
            if current_tags:
                found = 1  # Change this if need to support multiple tags
            else:
                found = 0
            tag_per_word_token.append(found)
        return tag_per_word_token

    def expand_subword_tokens(self, word_tokens: List[Token], context: str = None):
        cls_token = Token(self._tokenizer_internal.cls_token)
        sep_token = Token(self._tokenizer_internal.sep_token)
        #pad_token = self._tokenizer_internal.pad_token
        #sep_token_extra = bool(self._model_type in ['roberta'])
        cls_token_at_end = bool(self._model_type in ['xlnet'])
        # cls_token_segment_id = 2 if self._model_type in ['xlnet'] else 0
        #pad_on_left = bool(self._model_type in ['xlnet'])
        #pad_token_segment_id = 4 if self._model_type in ['xlnet'] else 0
        #pad_token_val=self._tokenizer.encoder[pad_token] if self._model_type in ['roberta'] else self._tokenizer.vocab[pad_token]
        offsets = []
        tokens = []
        tokens_at_end = 1
        if not cls_token_at_end:
            tokens.append(cls_token)
        else:
            tokens_at_end += 1

        if context is not None:
            context_tokens = self._tokenizer.tokenize(context)
            tokens += context_tokens
            tokens = tokens[:self._max_pieces - tokens_at_end - 1]  # just in case it's too long
            tokens.append(sep_token)
        for word_token in word_tokens:
            # NB: Watch out for downstream changes requiring adding leading space here
            subword_tokens = self._tokenizer.tokenize(word_token.text)
            if len(tokens) + len(subword_tokens) + tokens_at_end > self._max_pieces:
                break
            # Assume we want token for first subword in each word
            offsets.append(len(tokens))
            tokens += subword_tokens
        tokens.append(sep_token)
        if cls_token_at_end:
            tokens.append(cls_token)

        return tokens, offsets
