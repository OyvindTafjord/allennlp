from typing import Dict, Optional, List, Any

import logging
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel, gelu
import re
import torch
from torch.nn.modules.linear import Linear
from torch.nn.functional import binary_cross_entropy_with_logits

from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.modules import Attention
from allennlp.modules.attention import MultiHeadAttention
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy


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
                 per_choice_loss: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 mc_strategy: str = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if bert_weights_model:
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
                self._pre_classifier.apply(self._bert_model.init_bert_weights)
        self._classifier = None
        if bert_weights_model and hasattr(bert_model_loaded.model, "_classifier"):
            self._classifier = bert_model_loaded.model._classifier
            old_dims = (self._classifier.in_features, self._classifier.out_features)
            new_dims = (classifier_input_dim, classifier_output_dim)
            if old_dims != new_dims:
                logging.info(f"NOT copying BERT classifier weights, incompatible dims: {old_dims} vs {new_dims}")
                self._classifier = None
        if self._classifier is None:
            self._classifier = Linear(classifier_input_dim, classifier_output_dim)
            self._classifier.apply(self._bert_model.init_bert_weights)
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
        self._debug = 0


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


        encoded_layers, pooled_output = self._bert_model(input_ids=util.combine_initial_dims(input_ids),
                                            token_type_ids=util.combine_initial_dims(segment_ids),
                                            attention_mask=util.combine_initial_dims(question_mask),
                                            output_all_encoded_layers=self._all_layers)

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


    def forward_old(self,
                question: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        input_ids = question['tokens']
        question_mask = (input_ids != 0).long()
        _, pooled_output = self._bert_model(util.combine_initial_dims(input_ids),
                                            util.combine_initial_dims(segment_ids),
                                            util.combine_initial_dims(question_mask),
                                            output_all_encoded_layers=False)

        label_logits = self._classifier(pooled_output)
        output_dict = {}
        output_dict['label_logits'] = label_logits

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    """
    Needed to load old models into newer huggingface code:
    """
    @overrides
    def load_state_dict(self, d):
        tweaked_dict = {}
        for k, v in d.items():
            tweaked_k = k.replace('LayerNorm.gamma', 'LayerNorm.weight').replace('LayerNorm.beta', 'LayerNorm.bias')
            tweaked_dict[tweaked_k] = v
        super().load_state_dict(tweaked_dict)


@Model.register("bert_mc_qa_annotations")
class BertMCQAAnnotationsModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 top_layer_only: bool = True,
                 bert_weights_model: str = None,
                 per_choice_loss: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 mc_strategy: str = None,
                 num_annotations: int = None,
                 attention_size: int = None,
                 attention_normalize: bool = True,
                 attention_loss_weight: float = 1.0,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if bert_weights_model:
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
        self._attention = None
        self._attention_loss_weight = attention_loss_weight
        self._num_annotations = num_annotations
        if attention_size is not None and num_annotations is not None:
            self._attention = MultiHeadAttention(num_heads=num_annotations,
                                                 input_dim=self._output_dim,
                                                 attention_dim=attention_size * num_annotations,
                                                 values_dim=attention_size * num_annotations,
                                                 normalize=attention_normalize)
            classifier_input_dim *= 2
        self._mc_strategy = mc_strategy
        if self._mc_strategy in ["concat", "concat+"]:
            classifier_input_dim *= 2
            if self._mc_strategy == "concat+":
                self._pre_classifier = Linear(classifier_input_dim, classifier_input_dim)
                self._pre_classifier.apply(self._bert_model.init_bert_weights)
        self._classifier = None
        if bert_weights_model and hasattr(bert_model_loaded.model, "_classifier"):
            self._classifier = bert_model_loaded.model._classifier
            old_dims = (self._classifier.in_features, self._classifier.out_features)
            new_dims = (classifier_input_dim, classifier_output_dim)
            if old_dims != new_dims:
                logging.info(f"NOT copying BERT classifier weights, incompatible dims: {old_dims} vs {new_dims}")
                self._classifier = None
        if self._classifier is None:
            self._classifier = Linear(classifier_input_dim, classifier_output_dim)
            self._classifier.apply(self._bert_model.init_bert_weights)
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
                annotation_tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = question['tokens']

        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)

        question_mask = (input_ids != 0).long()
        question_mask_flat = util.combine_initial_dims(question_mask)
        annotation_tags_flat = annotation_tags.view(batch_size * num_choices, self._num_annotations, -1)

        if self._debug >= 0:
            print(f"batch_size = {batch_size}")
            print(f"num_choices = {num_choices}")
            print(f"question_mask = {question_mask}")
            print(f"input_ids.size() = {input_ids.size()}")
            print(f"input_ids = {input_ids}")
            print(f"segment_ids = {segment_ids}")
            print(f"label = {label}")


        encoded_layers, pooled_output = self._bert_model(input_ids=util.combine_initial_dims(input_ids),
                                                         token_type_ids=util.combine_initial_dims(segment_ids),
                                                         attention_mask=question_mask_flat,
                                                         output_all_encoded_layers=self._all_layers)

        if self._all_layers:
            mixed_layer = self._scalar_mix(encoded_layers, question_mask)
            pooled_output = self._bert_model.pooler(mixed_layer)

        attention_loss = 0
        attentions = None
        if self._attention is not None:
            query = encoded_layers[:,0]
            attended_layer, attentions = self._attention(query, encoded_layers, question_mask_flat)
            if self._debug >= 0:
                print(f"attentions = {attentions}")
                print(f"attention_tags = {annotation_tags}")
                print(f"attention_tags_flat = {annotation_tags_flat}")
            if annotation_tags is not None:
                """
                if not self._attention._normalize:
                    # For now, if attention is not normalized we'll treat each attention as
                    # prediction logit for 0/1 tags and average cross-entropy loss over unmasked tokens
                    question_mask_float = question_mask.float()
                    attention_loss = binary_cross_entropy_with_logits(
                        attentions,
                        tags.float(),
                        weight=question_mask_float,
                        reduction='sum'
                    )/question_mask_float.sum()
                    # Still need to normalize the attentions
                    attentions = torch.sigmoid(attentions) * question_mask_float
                    attentions = attentions / (attentions.sum(dim=1, keepdim=True) + 1e-13)
                """
                if True:
                    # Take dot products of attentions and 0/1 tags to get combined probability for 1's
                    tag_probs = (attentions * annotation_tags_flat).sum(dim=2) + 1e-10
                    attention_loss = -tag_probs.log().mean()
            if self._debug >= 0:
                print(f"tag_probs = {tag_probs}")
                print(f"attention_loss = {attention_loss}")

            pooled_output = torch.cat((pooled_output, attended_layer), dim=1)


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
            if attention_loss != 0:
                loss += self._attention_loss_weight * attention_loss
            output_dict["loss"] = loss

        if self._debug > 0:
            print(output_dict)
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    """
    Needed to load old models into newer huggingface code:
    """
    @overrides
    def load_state_dict(self, d):
        tweaked_dict = {}
        for k, v in d.items():
            tweaked_k = k.replace('LayerNorm.gamma', 'LayerNorm.weight').replace('LayerNorm.beta', 'LayerNorm.bias')
            tweaked_dict[tweaked_k] = v
        super().load_state_dict(tweaked_dict)


@Model.register("bert_mc_qa_per_choice")
class BertMCQAPerChoiceModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 top_layer_only: bool = True,
                 bert_weights_model: str = None,
                 num_choices: int = None,
                 use_sigmoid: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if bert_weights_model:
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
        self._use_sigmoid = use_sigmoid
        final_output_dim = 1 if self._use_sigmoid else 2
        # TODO: Check model name or dimension compatibility for reusing old weights
        if False and bert_weights_model and hasattr(bert_model_loaded.model, "_classifier"):
            self._classifier = bert_model_loaded.model._classifier
        else:
            self._classifier = Linear(self._output_dim, final_output_dim)
            self._classifier.apply(self._bert_model.init_bert_weights)
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

        class_weights = None
        if num_choices is not None and num_choices != 2:
            weights_tensor = torch.Tensor([1.0/num_choices, (num_choices-1.0)/num_choices])
            class_weights = torch.nn.Parameter(weights_tensor, requires_grad=False)
        if self._use_sigmoid:
            self._accuracy = BooleanAccuracy()
            self._loss = torch.nn.BCEWithLogitsLoss()
        else:
            self._accuracy = CategoricalAccuracy()
            self._loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        self._debug = 5


    def forward(self,
                question: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = question['tokens']

        batch_size = input_ids.size(0)

        question_mask = (input_ids != 0).long()

        if self._debug > 0:
            print(f"batch_size = {batch_size}")
            print(f"question_mask = {question_mask}")
            print(f"input_ids.size() = {input_ids.size()}")
            print(f"segment_ids.size() = {segment_ids.size()}")
            print(f"input_ids = {input_ids}")
            print(f"segment_ids = {segment_ids}")
            print(f"label = {label}")
            print(f"label.float() = {label.float()}")

        encoded_layers, pooled_output = self._bert_model(input_ids=input_ids,
                                                         token_type_ids=segment_ids,
                                                         attention_mask=question_mask,
                                                         output_all_encoded_layers=self._all_layers)

        if self._all_layers:
            mixed_layer = self._scalar_mix(encoded_layers, question_mask)
            pooled_output = self._bert_model.pooler(mixed_layer)

        if self._debug > 0:
            print(f"pooled_output = {pooled_output}")

        pooled_output = self._dropout(pooled_output)
        label_logits = self._classifier(pooled_output)
        if self._use_sigmoid:
            label_logits = label_logits.squeeze(1)
        output_dict = {}
        output_dict['label_logits'] = label_logits
        if self._use_sigmoid:
            output_dict['label_probs'] = torch.sigmoid(label_logits)
            output_dict['answer_index'] = label_logits > 0
        else:
            output_dict['label_probs'] = torch.nn.functional.softmax(label_logits, dim=1)
            output_dict['answer_index'] = label_logits.argmax(1)

        if label is not None:
            if self._use_sigmoid:
                loss = self._loss(label_logits, label.float())
                self._accuracy(label_logits > 0, label.byte())
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


@Model.register("bert_wordnet_links")
class BertWordnetLinksModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 top_layer_only: bool = True,
                 bert_weights_model: str = None,
                 layer_freeze_regexes: List[str] = None,
                 arg_token_mode: str = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if bert_weights_model:
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
        self._pre_classifier = None
        self._arg_token_mode = arg_token_mode
        if self._arg_token_mode in ["max", "mean"]:
            self._output_dim = 3 * self._output_dim

        if True:
            self._pre_classifier_dim = self._output_dim // 2
            self._pre_classifier = Linear(self._output_dim, self._pre_classifier_dim)
            self._pre_classifier.apply(self._bert_model.init_bert_weights)
            self._classifier = Linear(self._pre_classifier_dim, 2)
            self._classifier.apply(self._bert_model.init_bert_weights)
        else:
            if bert_weights_model and hasattr(bert_model_loaded.model, "_classifier"):
                self._classifier = bert_model_loaded.model._classifier
            else:
                self._classifier = Linear(self._output_dim, 2)
                self._classifier.apply(self._bert_model.init_bert_weights)
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

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self._debug = -1


    def forward(self,
                assertion: Dict[str, torch.LongTensor],
                arg_marks: torch.LongTensor = None,
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = assertion['tokens']

        batch_size = input_ids.size(0)
        num_choices = input_ids.size(1)

        question_mask = (input_ids != 0).long()

        if self._debug > 100:
            print(f"batch_size = {batch_size}")
            print(f"num_choices = {num_choices}")
            print(f"question_mask = {question_mask}")
            print(f"input_ids.size() = {input_ids.size()}")

        encoded_layers, pooled_output = self._bert_model(input_ids=util.combine_initial_dims(input_ids),
                                                         token_type_ids=util.combine_initial_dims(segment_ids),
                                                         attention_mask=util.combine_initial_dims(question_mask),
                                                         output_all_encoded_layers=self._all_layers)

        output_layer = encoded_layers
        if self._all_layers:
            output_layer = self._scalar_mix(encoded_layers)
            pooled_output = self._bert_model.pooler(output_layer)

        if self._arg_token_mode in ["max", "mean"]:
            arg1_mask = arg_marks == 1
            arg2_mask = arg_marks == 2

            if self._arg_token_mode == "max":
                arg1_tensor = util.masked_max(output_layer,arg1_mask.unsqueeze(2),1)
                arg2_tensor = util.masked_max(output_layer,arg2_mask.unsqueeze(2),1)
            else:
                arg1_tensor = util.masked_mean(output_layer,arg1_mask.unsqueeze(2),1)
                arg2_tensor = util.masked_mean(output_layer,arg2_mask.unsqueeze(2),1)
            pooled_output = torch.cat([pooled_output, arg1_tensor, arg2_tensor], dim=1)

        if self._pre_classifier is not None:
            pooled_output = gelu(pooled_output)
            pooled_output = self._dropout(pooled_output)
            pooled_output = self._pre_classifier(pooled_output)

        pooled_output = self._dropout(pooled_output)

        label_logits = self._classifier(pooled_output)
        output_dict = {}
        output_dict['label_logits'] = label_logits
        output_dict['label_probs'] = torch.nn.functional.softmax(label_logits, dim=1)
        output_dict['is_correct'] = label_logits.argmax(1)

        if label is not None:
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

@Model.register("bert_classifier_attention")
class BertClassifierAttentionModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 top_layer_only: bool = True,
                 bert_weights_model: str = None,
                 attention: Attention = None,
                 attention_strategy: str = None,
                 attention_loss_weight: float = 0.0,
                 num_labels: int = 2,
                 layer_freeze_regexes: List[str] = None,
                 pre_classifier_mode: int = None,
                 skip_pooling: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._per_choice_loss = False
        if bert_weights_model:
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
        self._num_labels = num_labels
        classifier_input_dim = self._output_dim
        classifier_output_dim = self._num_labels
        if self._per_choice_loss:
            classifier_output_dim = 1
        self._pre_classifier = None
        self._attention = attention
        self._attention_strategy = attention_strategy
        self._attention_loss_weight = attention_loss_weight
        if self._attention_strategy in ["concat", "concat+"]:
            classifier_input_dim *= 2
            if self._attention_strategy == "concat+":
                self._pre_classifier = Linear(classifier_input_dim, classifier_input_dim)
                self._pre_classifier.apply(self._bert_model.init_bert_weights)
        if pre_classifier_mode is not None:
            self._pre_classifier = Linear(classifier_input_dim, classifier_input_dim)
            self._pre_classifier.apply(self._bert_model.init_bert_weights)
        self._classifier = None
        if bert_weights_model and hasattr(bert_model_loaded.model, "_classifier"):
            self._classifier = bert_model_loaded.model._classifier
            old_dims = (self._classifier.in_features, self._classifier.out_features)
            new_dims = (classifier_input_dim, classifier_output_dim)
            if old_dims != new_dims:
                logging.info(f"NOT copying BERT classifier weights, incompatible dims: {old_dims} vs {new_dims}")
                self._classifier = None
        if self._classifier is None:
            self._classifier = Linear(classifier_input_dim, classifier_output_dim)
            self._classifier.apply(self._bert_model.init_bert_weights)
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

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self._debug = 5
        self._skip_pooling = skip_pooling
        if self._per_choice_loss:
            self._accuracy = BooleanAccuracy()
            self._loss = torch.nn.BCEWithLogitsLoss()


    def forward(self,
                question: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        self._debug -= 1
        input_ids = question['tokens']

        batch_size = input_ids.size(0)

        question_mask = (input_ids != 0).long()

        if self._debug > 0:
            print(f"batch_size = {batch_size}")
            print(f"question_mask = {question_mask}")
            print(f"input_ids.size() = {input_ids.size()}")
            print(f"input_ids = {input_ids}")
            print(f"segment_ids = {segment_ids}")
            print(f"tags = {tags}")
            print(f"label = {label}")


        encoded_layers, pooled_output = self._bert_model(input_ids=input_ids,
                                                         token_type_ids=segment_ids,
                                                         attention_mask=question_mask,
                                                         output_all_encoded_layers=self._all_layers)

        if self._all_layers:
            encoded_layers = self._scalar_mix(encoded_layers, question_mask)
            pooled_output = self._bert_model.pooler(encoded_layers)
        if self._skip_pooling:
            pooled_output = encoded_layers[:,0]

        if self._debug > 0:
            print(f"pooled_output = {pooled_output}")

        attention_loss = 0
        attentions = None
        if self._attention_strategy in ["concat", "concat+"]:
            query = encoded_layers[:,0]
            attentions = self._attention(query, encoded_layers, question_mask)
            if tags is not None:
                if not self._attention._normalize:
                    # For now, if attention is not normalized we'll treat each attention as
                    # prediction logit for 0/1 tags and average cross-entropy loss over unmasked tokens
                    question_mask_float = question_mask.float()
                    attention_loss = binary_cross_entropy_with_logits(
                        attentions,
                        tags.float(),
                        weight=question_mask_float,
                        reduction='sum'
                    )/question_mask_float.sum()
                    # Still need to normalize the attentions
                    attentions = torch.sigmoid(attentions) * question_mask_float
                    attentions = attentions / (attentions.sum(dim=1, keepdim=True) + 1e-13)
                else:
                    # Take dot products of attentions and 0/1 tags to get combined probability for 1's
                    tag_probs = attentions.unsqueeze(1).bmm(tags.float().unsqueeze(2)) + 1e-10
                    attention_loss = -tag_probs.log().mean()

            attended_layer = util.weighted_sum(encoded_layers, attentions)
            pooled_output = torch.cat((pooled_output, attended_layer), dim=1)

        pooled_output = self._dropout(pooled_output)

        if self._pre_classifier is not None:
            pooled_output = self._pre_classifier(pooled_output)
            pooled_output = gelu(pooled_output)
            pooled_output = self._dropout(pooled_output)

        label_logits = self._classifier(pooled_output)

        output_dict = {}
        output_dict['label_logits'] = label_logits
        output_dict['label_probs'] = torch.nn.functional.softmax(label_logits, dim=1)
        output_dict['answer_index'] = label_logits.argmax(1)
        if attentions is not None:
            output_dict['attentions'] = attentions

        if label is not None:
            if self._per_choice_loss:
                logits_flat = label_logits.squeeze(1)
                loss = self._loss(logits_flat, label.float())
                self._accuracy(logits_flat > 0, label.byte())
            else:
                loss = self._loss(label_logits, label)
                self._accuracy(label_logits, label)
            if attention_loss != 0:
                loss += self._attention_loss_weight * attention_loss
            output_dict["loss"] = loss
            if self._debug > 0:
                print(f"label_logits = {label_logits}")
                print(f"attention_loss = {attention_loss}")
                print(f"label = {label}")

        if self._debug > 0:
            print(output_dict)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }


@Model.register("bert_classifier_attention_old")
class BertClassifierAttentionModelOld(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 top_layer_only: bool = True,
                 bert_weights_model: str = None,
                 per_choice_loss: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 mc_strategy: str = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if bert_weights_model:
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
                self._pre_classifier.apply(self._bert_model.init_bert_weights)
        self._classifier = None
        if bert_weights_model and hasattr(bert_model_loaded.model, "_classifier"):
            self._classifier = bert_model_loaded.model._classifier
            old_dims = (self._classifier.in_features, self._classifier.out_features)
            new_dims = (classifier_input_dim, classifier_output_dim)
            if old_dims != new_dims:
                logging.info(f"NOT copying BERT classifier weights, incompatible dims: {old_dims} vs {new_dims}")
                self._classifier = None
        if self._classifier is None:
            self._classifier = Linear(classifier_input_dim, classifier_output_dim)
            self._classifier.apply(self._bert_model.init_bert_weights)
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
        self._debug = 5


    def forward(self,
                question: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                tags: torch.LongTensor = None,
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


        encoded_layers, pooled_output = self._bert_model(input_ids=util.combine_initial_dims(input_ids),
                                                         token_type_ids=util.combine_initial_dims(segment_ids),
                                                         attention_mask=util.combine_initial_dims(question_mask),
                                                         output_all_encoded_layers=self._all_layers)

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
                #binary_label = label.new_zeros((len(label), num_choices))
                #binary_label.scatter_(1, label.unsqueeze(1), 1.0)
                #binary_label = binary_label.view(-1,1).squeeze(1)
                binary_label = label
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




@Model.register("bert_tagger")
class BertTaggerModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 top_layer_only: bool = True,
                 bert_weights_model: str = None,
                 per_choice_loss: bool = False,
                 layer_freeze_regexes: List[str] = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        if bert_weights_model:
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
        # TODO: Check dimensions before replacing
        if bert_weights_model and hasattr(bert_model_loaded.model, "_classifier"):
            self._classifier = bert_model_loaded.model._classifier
        else:
            final_output_dim = 1
            self._classifier = Linear(self._output_dim, final_output_dim)
            self._classifier.apply(self._bert_model.init_bert_weights)
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
        self._debug = -1


    def forward(self,
                question: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                tags: torch.LongTensor = None,
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


        encoded_layers, pooled_output = self._bert_model(input_ids=util.combine_initial_dims(input_ids),
                                                         token_type_ids=util.combine_initial_dims(segment_ids),
                                                         attention_mask=util.combine_initial_dims(question_mask),
                                                         output_all_encoded_layers=self._all_layers)

        if self._all_layers:
            mixed_layer = self._scalar_mix(encoded_layers, question_mask)
            pooled_output = self._bert_model.pooler(mixed_layer)

        if self._debug > 0:
            print(f"pooled_output = {pooled_output}")

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




