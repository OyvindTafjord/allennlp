from typing import Dict, Optional, List, Any

import logging
from overrides import overrides
from pytorch_transformers.modeling_bert import BertConfig, BertModel, gelu
import re
import torch
from torch.nn.modules.linear import Linear
from torch.nn.functional import binary_cross_entropy_with_logits

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1

logger = logging.getLogger(__name__)

@Model.register("bert_mc_qa")
class BertMCQAModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 top_layer_only: bool = True,
                 bert_weights_model: str = None,
                 reset_classifier: bool = False,
                 per_choice_loss: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 mc_strategy: str = None,
                 on_load: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if on_load:
            logging.info(f"Skipping loading of initial BERT weights")
            bert_config = BertConfig.from_pretrained(pretrained_model)
            self._bert_model = BertModel(bert_config)

        elif bert_weights_model:
            logging.info(f"Loading BERT weights model from {bert_weights_model}")
            bert_model_loaded = load_archive(bert_weights_model)
            self._bert_model = bert_model_loaded.model._bert_model
        else:
            self._bert_model = BertModel.from_pretrained(pretrained_model)

        for name, param in self._bert_model.named_parameters():
            grad = requires_grad
            if layer_freeze_regexes and grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            param.requires_grad = grad

        bert_config = self._bert_model.config
        self._output_dim = bert_config.hidden_size
        self._dropout = torch.nn.Dropout(bert_config.hidden_dropout_prob)
        self._per_choice_loss = per_choice_loss
        classifier_input_dim = self._output_dim
        classifier_output_dim = 1
        self._pre_classifier = None
        self._mc_strategy = mc_strategy
        if self._mc_strategy in ["concat", "concat+"]:
            classifier_input_dim *= 2
            if self._mc_strategy == "concat+":
                self._pre_classifier = Linear(classifier_input_dim, classifier_input_dim)
                self._pre_classifier.apply(self._bert_model.init_weights)
        self._classifier = None
        if bert_weights_model and hasattr(bert_model_loaded.model, "_classifier") \
                and not reset_classifier:
            self._classifier = bert_model_loaded.model._classifier
            old_dims = (self._classifier.in_features, self._classifier.out_features)
            new_dims = (classifier_input_dim, classifier_output_dim)
            if old_dims != new_dims:
                logging.info(f"NOT copying BERT classifier weights, incompatible dims: {old_dims} vs {new_dims}")
                self._classifier = None
        if self._classifier is None:
            self._classifier = Linear(classifier_input_dim, classifier_output_dim)
            self._classifier.apply(self._bert_model.init_weights)
        self._all_layers = not top_layer_only
        if self._all_layers:
            if bert_weights_model and hasattr(bert_model_loaded.model, "_scalar_mix") \
                    and bert_model_loaded.model._scalar_mix is not None:
                self._scalar_mix = bert_model_loaded.model._scalar_mix
            else:
                num_layers = bert_config.num_hidden_layers
                initial_scalar_parameters = num_layers * [0.0]
                initial_scalar_parameters[-1] = 5.0  # Starts with most mass on last layer
                self._scalar_mix = ScalarMix(bert_config.num_hidden_layers,
                                             initial_scalar_parameters=initial_scalar_parameters,
                                             do_layer_norm=False)
        else:
            self._scalar_mix = None


        if self._per_choice_loss:
            self._accuracy = BooleanAccuracy()
            self._loss = torch.nn.BCEWithLogitsLoss()
        else:
            self._accuracy = CategoricalAccuracy()
            self._loss = torch.nn.CrossEntropyLoss()
        self._debug = 2


    def forward(self,
                question: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = question['tokens']

        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)

        question_mask = (input_ids != 0).long()

        if self._debug > 0:
            print(f"batch_size = {batch_size}")
            print(f"num_choices = {num_choices}")
            print(f"question_mask = {question_mask}")
            print(f"input_ids.size() = {input_ids.size()}")
            print(f"input_ids = {input_ids}")
            print(f"segment_ids = {segment_ids}")
            print(f"label = {label}")

        last_layer, pooled_output = self._bert_model(input_ids=util.combine_initial_dims(input_ids),
                                                         token_type_ids=util.combine_initial_dims(segment_ids),
                                                         attention_mask=util.combine_initial_dims(question_mask))

        encoded_layers = None # TODO if needed

        if self._all_layers:
            mixed_layer = self._scalar_mix(encoded_layers, question_mask)
            pooled_output = self._bert_model.pooler(mixed_layer)

        if self._mc_strategy in ["concat", "concat+"]:
            # Average pooled output across all answer options and concat to original
            pooled_avg = pooled_output.view(batch_size, num_choices, -1).mean(dim=1)
            pooled_avg = torch.unsqueeze(pooled_avg, 1).repeat(1, num_choices, 1)
            pooled_avg = pooled_avg.view(batch_size*num_choices, -1)
            pooled_output = torch.cat((pooled_output, pooled_avg), dim=1)

        if self._debug > 0:
            print(f"pooled_output = {pooled_output}")

        pooled_output = self._dropout(pooled_output)

        if self._pre_classifier is not None:
            pooled_output = gelu(pooled_output)
            pooled_output = self._pre_classifier(pooled_output)
            pooled_output = self._dropout(pooled_output)

        label_logits = self._classifier(pooled_output)
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
            print(output_dict)
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



@Model.register("bert_span_prediction")
class BertSpanPredictionModel(Model):
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
            transformer_config = BertConfig.from_pretrained(pretrained_model)
            self._transformer_model = BertModel(transformer_config)

        elif transformer_weights_model:
            logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
            transformer_model_loaded = load_archive(transformer_weights_model)
            self._transformer_model = transformer_model_loaded.model._transformer_model
        else:
            self._transformer_model = BertModel.from_pretrained(pretrained_model)

        for name, param in self._transformer_model.named_parameters():
            grad = requires_grad
            if layer_freeze_regexes and grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            param.requires_grad = grad

        transformer_config = self._transformer_model.config
        num_labels = 2  # For start/end
        self.qa_outputs = Linear(transformer_config.hidden_size, num_labels)

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
        input_ids = tokens['tokens']

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

        transformer_outputs = self._transformer_model(input_ids=input_ids,
                                                      token_type_ids=segment_ids,
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
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                exact_match = 0
                f1_score = 0
                if answer_texts:
                    exact_match, f1_score = self._squad_metrics(best_span_string, answer_texts)
                output_dict['exact_match'].append(exact_match)
                output_dict['f1_score'].append(f1_score)
            output_dict['tokens_texts'] = tokens_texts

        if self._debug > 0:
            logger.info(f"output_dict = {output_dict}")

        return output_dict

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = ' '.join(tokens)
        # De-tokenize WordPieces that have been split off.
        text = text.replace(" ##", "")
        text = text.replace("##", "").strip()
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
