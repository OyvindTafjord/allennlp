from typing import Dict, Optional, List, Any

import logging
from overrides import overrides
from transformers.modeling_roberta import RobertaClassificationHead, RobertaConfig, RobertaModel
from transformers.tokenization_gpt2 import bytes_to_unicode
import re
import torch
from torch.nn.modules.linear import Linear
from torch.nn.functional import binary_cross_entropy_with_logits

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, F1Measure, SquadEmAndF1

logger = logging.getLogger(__name__)

@Model.register("roberta_mc_qa")
class RobertaMCQAModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 top_layer_only: bool = True,
                 transformer_weights_model: str = None,
                 reset_classifier: bool = False,
                 per_choice_loss: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 mc_strategy: str = None,
                 on_load: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if on_load:
            logging.info(f"Skipping loading of initial Transformer weights")
            transformer_config = RobertaConfig.from_pretrained(pretrained_model)
            self._transformer_model = RobertaModel(transformer_config)

        elif transformer_weights_model:
            logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
            transformer_model_loaded = load_archive(transformer_weights_model)
            self._transformer_model = transformer_model_loaded.model._transformer_model
        else:
            self._transformer_model = RobertaModel.from_pretrained(pretrained_model)

        for name, param in self._transformer_model.named_parameters():
            grad = requires_grad
            if layer_freeze_regexes and grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            param.requires_grad = grad

        transformer_config = self._transformer_model.config

        self._output_dim = transformer_config.hidden_size
        self._per_choice_loss = per_choice_loss
        classifier_input_dim = self._output_dim
        classifier_output_dim = 1
        transformer_config.num_labels = classifier_output_dim
        self._classifier = None
        if not on_load and transformer_weights_model \
                and hasattr(transformer_model_loaded.model, "_classifier") \
                and not reset_classifier:
            self._classifier = transformer_model_loaded.model._classifier
            old_dims = (self._classifier.dense.in_features, self._classifier.out_proj.out_features)
            new_dims = (classifier_input_dim, classifier_output_dim)
            if old_dims != new_dims:
                logging.info(f"NOT copying Transformer classifier weights, incompatible dims: {old_dims} vs {new_dims}")
                self._classifier = None
        if self._classifier is None:
            self._classifier = RobertaClassificationHead(transformer_config)

        if self._per_choice_loss:
            self._accuracy = BooleanAccuracy()
            self._loss = torch.nn.BCEWithLogitsLoss()
        else:
            self._accuracy = CategoricalAccuracy()
            self._loss = torch.nn.CrossEntropyLoss()
        self._debug = 2
        self._padding_value = 1  # The index of the RoBERTa padding token

    def forward(self,
                question: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        tokens_data = question['tokens']
        input_ids = tokens_data['token_ids']

        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)

        question_mask = (input_ids != self._padding_value).long()

        if self._debug > 0:
            logger.info(f"batch_size = {batch_size}")
            logger.info(f"num_choices = {num_choices}")
            logger.info(f"question_mask = {question_mask}")
            logger.info(f"input_ids.size() = {input_ids.size()}")
            logger.info(f"input_ids = {input_ids}")
            logger.info(f"segment_ids = {segment_ids}")
            logger.info(f"label = {label}")

        # Segment ids are not used by RoBERTa

        transformer_outputs = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                      # token_type_ids=util.combine_initial_dims(segment_ids),
                                                      attention_mask=util.combine_initial_dims(question_mask))

        cls_output = transformer_outputs[0]

        if self._debug > 0:
            logger.info(f"cls_output = {cls_output}")

        label_logits = self._classifier(cls_output)
        label_logits_flat = label_logits.squeeze(1)
        label_logits = label_logits.view(-1, num_choices)

        output_dict = {}
        output_dict['label_logits'] = label_logits

        if self._per_choice_loss:
            output_dict['label_probs'] = torch.sigmoid(label_logits_flat).view(-1, num_choices)
            output_dict['answer_index'] = (label_logits_flat > 0).view(-1, num_choices)
        else:
            output_dict['label_probs'] = torch.nn.functional.softmax(label_logits, dim=1)
            output_dict['answer_index'] = label_logits.argmax(1)

        if label is not None:
            if self._per_choice_loss:
                binary_label = label.new_zeros((len(label), num_choices))
                binary_label.scatter_(1, label.unsqueeze(1), 1.0)
                binary_label = binary_label.view(-1,1).squeeze(1)
                loss = self._loss(label_logits_flat, binary_label.float())
                self._accuracy(label_logits_flat > 0, binary_label.byte())
            else:
                loss = self._loss(label_logits, label)
                self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        if self._debug > 0:
            logger.info(output_dict)
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1,
              **kwargs) -> 'Model':
        model_params = config.get('model')
        model_params.update({"on_load": True})
        config.update({'model': model_params})
        return super()._load(config=config,
                             serialization_dir=serialization_dir,
                             weights_file=weights_file,
                             cuda_device=cuda_device,
                             **kwargs)


@Model.register("roberta_classifier")
class RobertaClassifierModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 num_labels: int = None,
                 transformer_weights_model: str = None,
                 label_namespace: str = "labels",
                 reset_classifier: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 on_load: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if on_load:
            logging.info(f"Skipping loading of initial Transformer weights")
            transformer_config = RobertaConfig.from_pretrained(pretrained_model)
            self._transformer_model = RobertaModel(transformer_config)

        elif transformer_weights_model:
            logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
            transformer_model_loaded = load_archive(transformer_weights_model)
            self._transformer_model = transformer_model_loaded.model._transformer_model
        else:
            self._transformer_model = RobertaModel.from_pretrained(pretrained_model)

        for name, param in self._transformer_model.named_parameters():
            grad = requires_grad
            if layer_freeze_regexes and grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            param.requires_grad = grad

        transformer_config = self._transformer_model.config

        self._output_dim = transformer_config.hidden_size
        classifier_input_dim = self._output_dim
        self._num_labels = num_labels
        classifier_output_dim = self._num_labels
        transformer_config.num_labels = classifier_output_dim
        self._classifier = None
        if not on_load and transformer_weights_model \
                and hasattr(transformer_model_loaded.model, "_classifier") \
                and not reset_classifier:
            self._classifier = transformer_model_loaded.model._classifier
            old_dims = (self._classifier.dense.in_features, self._classifier.out_proj.out_features)
            new_dims = (classifier_input_dim, classifier_output_dim)
            if old_dims != new_dims:
                logging.info(f"NOT copying Transformer classifier weights, incompatible dims: {old_dims} vs {new_dims}")
                self._classifier = None
        if self._classifier is None:
            self._classifier = RobertaClassificationHead(transformer_config)

        self._label_namespace = label_namespace
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self._f1measure = F1Measure(positive_label=1)
        self._debug = 2
        self._padding_value = 1  # The index of the RoBERTa padding token

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = tokens['tokens']['token_ids']

        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)

        question_mask = (input_ids != self._padding_value).long()

        if self._debug > 0:
            logger.info(f"batch_size = {batch_size}")
            logger.info(f"num_choices = {num_choices}")
            logger.info(f"question_mask = {question_mask}")
            logger.info(f"input_ids.size() = {input_ids.size()}")
            logger.info(f"input_ids = {input_ids}")
            logger.info(f"segment_ids = {segment_ids}")
            logger.info(f"label = {label}")

        # Segment ids are not used by RoBERTa

        transformer_outputs = self._transformer_model(input_ids=input_ids,
                                                      # token_type_ids=segment_ids,
                                                      attention_mask=question_mask)

        cls_output = transformer_outputs[0]

        if self._debug > 0:
            logger.info(f"cls_output = {cls_output}")

        label_logits = self._classifier(cls_output)

        output_dict = {}
        output_dict['label_logits'] = label_logits

        output_dict['label_probs'] = torch.nn.functional.softmax(label_logits, dim=1)
        label_predicted = label_logits.argmax(1).detach().cpu()
        output_dict['label_predicted'] = label_predicted
        output_dict['label_str'] = []
        for i in range(batch_size):
            label_predicted1 = label_predicted[i].item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace) \
                            .get(label_predicted1, str(label_predicted1))
            output_dict['label_str'].append(label_str)

        if label is not None:
            loss = self._loss(label_logits, label)
            self._accuracy(label_logits, label)
            self._f1measure(label_logits, label)
            output_dict["loss"] = loss

        if self._debug > 0:
            logger.info(output_dict)
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1 = self._f1measure.get_metric(reset)
        return {
            'accuracy': self._accuracy.get_metric(reset),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1,
              **kwargs) -> 'Model':
        model_params = config.get('model')
        model_params.update({"on_load": True})
        config.update({'model': model_params})
        return super()._load(config=config,
                             serialization_dir=serialization_dir,
                             weights_file=weights_file,
                             cuda_device=cuda_device,
                             **kwargs)


@Model.register("roberta_span_prediction")
class RobertaSpanPredictionModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 transformer_weights_model: str = None,
                 layer_freeze_regexes: List[str] = None,
                 on_load: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if on_load:
            logging.info(f"Skipping loading of initial Transformer weights")
            transformer_config = RobertaConfig.from_pretrained(pretrained_model)
            self._transformer_model = RobertaModel(transformer_config)

        elif transformer_weights_model:
            logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
            transformer_model_loaded = load_archive(transformer_weights_model)
            self._transformer_model = transformer_model_loaded.model._transformer_model
        else:
            self._transformer_model = RobertaModel.from_pretrained(pretrained_model)

        for name, param in self._transformer_model.named_parameters():
            grad = requires_grad
            if layer_freeze_regexes and grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            param.requires_grad = grad

        transformer_config = self._transformer_model.config
        num_labels = 2  # For start/end
        self.qa_outputs = Linear(transformer_config.hidden_size, num_labels)

        # Import GTP2 machinery to get from tokens to actual text
        self.byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        self._debug = 2
        self._padding_value = 1  # The index of the RoBERTa padding token


    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                start_positions: torch.LongTensor = None,
                end_positions: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = tokens['tokens']['token_ids']
        batch_size = input_ids.size(0)
        tokens_mask = (input_ids != self._padding_value).long()

        if self._debug > 0:
            logger.info(f"batch_size = {batch_size}")
            logger.info(f"tokens_mask = {tokens_mask}")
            logger.info(f"input_ids.size() = {input_ids.size()}")
            logger.info(f"input_ids = {input_ids}")
            logger.info(f"segment_ids = {segment_ids}")
            logger.info(f"start_positions = {start_positions}")
            logger.info(f"end_positions = {end_positions}")

        # Segment ids are not used by RoBERTa

        transformer_outputs = self._transformer_model(input_ids=input_ids,
                                                      # token_type_ids=segment_ids,
                                                      attention_mask=tokens_mask)
        sequence_output = transformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        span_start_logits = util.replace_masked_values(start_logits, tokens_mask, -1e7)
        span_end_logits = util.replace_masked_values(end_logits, tokens_mask, -1e7)
        best_span = get_best_span(span_start_logits, span_end_logits)
        span_start_probs = util.masked_softmax(span_start_logits, tokens_mask)
        span_end_probs = util.masked_softmax(span_end_logits, tokens_mask)
        output_dict = {"start_logits": start_logits, "end_logits": end_logits, "best_span": best_span}
        output_dict["start_probs"] = span_start_probs
        output_dict["end_probs"] = span_end_probs

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            self._span_start_accuracy(span_start_logits, start_positions)
            self._span_end_accuracy(span_end_logits, end_positions)
            self._span_accuracy(best_span, torch.cat([start_positions.unsqueeze(-1), end_positions.unsqueeze(-1)], -1))

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            # Should we mask out invalid positions here?
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            output_dict["loss"] = total_loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['best_span_str'] = []
            output_dict['exact_match'] = []
            output_dict['f1_score'] = []
            tokens_texts = []
            for i in range(batch_size):
                tokens_text = metadata[i]['tokens']
                tokens_texts.append(tokens_text)
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                predicted_start = predicted_span[0]
                predicted_end = predicted_span[1]
                predicted_tokens = tokens_text[predicted_start:(predicted_end + 1)]
                best_span_string = self.convert_tokens_to_string(predicted_tokens)
                if predicted_start == predicted_end == 0:
                    best_span_string = ""
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                if metadata[i].get('is_impossible'):
                    answer_texts = [""]
                exact_match = 0
                f1_score = 0
                if len(answer_texts) > 0:
                    exact_match, f1_score = self._squad_metrics(best_span_string, answer_texts)

                output_dict['exact_match'].append(exact_match)
                output_dict['f1_score'].append(f1_score)
            output_dict['tokens_texts'] = tokens_texts

        if self._debug > 0:
            logger.info(f"output_dict = {output_dict}")

        return output_dict

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        text = text.strip()
        return text

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
            'start_acc': self._span_start_accuracy.get_metric(reset),
            'end_acc': self._span_end_accuracy.get_metric(reset),
            'span_acc': self._span_accuracy.get_metric(reset),
            'em': exact_match,
            'f1': f1_score,
        }

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1,
              **kwargs) -> 'Model':
        model_params = config.get('model')
        model_params.update({"on_load": True})
        config.update({'model': model_params})
        return super()._load(config=config,
                             serialization_dir=serialization_dir,
                             weights_file=weights_file,
                             cuda_device=cuda_device,
                             **kwargs)


class RobertaTokenClassificationHead(torch.nn.Module):
    """Head for token-level classification tasks."""

    def __init__(self, config, num_labels):
        super(RobertaTokenClassificationHead, self).__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        # x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@Model.register("roberta_tagger")
class RobertaTaggerModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 num_labels: int = None,
                 transformer_weights_model: str = None,
                 classifier_head: str = None,
                 reset_classifier: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 on_load: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if on_load:
            logging.info(f"Skipping loading of initial Transformer weights")
            transformer_config = RobertaConfig.from_pretrained(pretrained_model)
            self._transformer_model = RobertaModel(transformer_config)

        elif transformer_weights_model:
            logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
            transformer_model_loaded = load_archive(transformer_weights_model)
            self._transformer_model = transformer_model_loaded.model._transformer_model
        else:
            self._transformer_model = RobertaModel.from_pretrained(pretrained_model)

        for name, param in self._transformer_model.named_parameters():
            grad = requires_grad
            if layer_freeze_regexes and grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            param.requires_grad = grad

        transformer_config = self._transformer_model.config
        self._dropout = torch.nn.Dropout(transformer_config.hidden_dropout_prob)

        self._output_dim = transformer_config.hidden_size
        classifier_input_dim = self._output_dim
        self._num_labels = num_labels
        classifier_output_dim = self._num_labels
        transformer_config.num_labels = classifier_output_dim
        self._classifier = None
        if not on_load and transformer_weights_model \
                and hasattr(transformer_model_loaded.model, "_classifier") \
                and not reset_classifier:
            self._classifier = transformer_model_loaded.model._classifier
            old_dims = (self._classifier.dense.in_features, self._classifier.out_proj.out_features)
            new_dims = (classifier_input_dim, classifier_output_dim)
            if old_dims != new_dims:
                logging.info(f"NOT copying Transformer classifier weights, incompatible dims: {old_dims} vs {new_dims}")
                self._classifier = None
        if self._classifier is None:
            if classifier_head == "std":
                logging.info(f"Using standard token classification head.")
                self._classifier = RobertaTokenClassificationHead(transformer_config, self._num_labels)
            else:
                self._classifier = Linear(transformer_config.hidden_size, self._num_labels)

        self._accuracy = CategoricalAccuracy()
        self._f1measure = F1Measure(positive_label=1)

        # self._loss = torch.nn.CrossEntropyLoss()
        self._debug = 2
        self._padding_value = 1  # The index of the RoBERTa padding token

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                word_offsets: torch.LongTensor,
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = tokens['tokens']

        batch_size = input_ids.size(0)

        question_mask = (input_ids != self._padding_value).long()
        word_offsets_mask = (word_offsets != -1).long()

        if self._debug > 0:
            logger.info(f"batch_size = {batch_size}")
            logger.info(f"question_mask = {question_mask}")
            logger.info(f"input_ids.size() = {input_ids.size()}")
            logger.info(f"input_ids = {input_ids}")
            logger.info(f"tags = {tags}")
            logger.info(f"word_offsets = {word_offsets}")
            logger.info(f"word_offsets_mask = {word_offsets_mask}")

        # Segment ids are not used by RoBERTa

        transformer_outputs = self._transformer_model(input_ids=input_ids,
                                                      # token_type_ids=segment_ids,
                                                      attention_mask=question_mask)

        sequence_output = transformer_outputs[0]
        sequence_output = self._dropout(sequence_output)
        range_vector = util.get_range_vector(batch_size, device=util.get_device_of(word_offsets)).unsqueeze(1)
        word_output = sequence_output[range_vector, word_offsets.long()]

        label_logits = self._classifier(word_output)

        if self._debug > 0:
            logger.info(f"sequence_output shape = {sequence_output.size()}")
            logger.info(f"word_output shape = {word_output.size()}")
            logger.info(f"label_logits = {label_logits}")
            logger.info(f"word_output = {word_output}")

        output_dict = {}
        output_dict['label_logits'] = label_logits

        output_dict['label_probs'] = util.masked_softmax(label_logits, word_offsets_mask.unsqueeze(-1))
        output_dict['label_predicted'] = label_logits.argmax(2)

        if tags is not None:
            # Replace masked tags by zero
            tags.masked_fill_((1 - word_offsets_mask).to(dtype=torch.bool), 0)
            if self._debug > 0:
                logger.info(f"tags mask filled = {tags}")
            loss = util.sequence_cross_entropy_with_logits(label_logits, tags, word_offsets_mask)
            self._accuracy(label_logits, tags, word_offsets_mask)
            self._f1measure(label_logits, tags, word_offsets_mask)
            output_dict["loss"] = loss

        if self._debug > 0:
            logger.info(output_dict)
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1 = self._f1measure.get_metric(reset)
        return {
            'accuracy': self._accuracy.get_metric(reset),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1,
              **kwargs) -> 'Model':
        model_params = config.get('model')
        model_params.update({"on_load": True})
        config.update({'model': model_params})
        return super()._load(config=config,
                             serialization_dir=serialization_dir,
                             weights_file=weights_file,
                             cuda_device=cuda_device,
                             **kwargs)