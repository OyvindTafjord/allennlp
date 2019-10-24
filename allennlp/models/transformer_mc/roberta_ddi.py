from typing import Dict, Optional, List, Any

import logging
from pytorch_transformers.modeling_roberta import RobertaClassificationHead, RobertaConfig, RobertaModel
import re
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, SquadEmAndF1

logger = logging.getLogger(__name__)

# Essentially a copy of roberta_classifier, but keeping separate for flexibility
@Model.register("roberta_ddi_classifier")
class RobertaDdiClassifierModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 num_labels: int = None,
                 transformer_weights_model: str = None,
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

        self._accuracy = CategoricalAccuracy()
        # Only calculate F1 scores wrt label 1 for now
        self._f1measure = F1Measure(positive_label=1)
        self._loss = torch.nn.CrossEntropyLoss()
        self._debug = 2
        self._padding_value = 1  # The index of the RoBERTa padding token

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = tokens['tokens']

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
        output_dict['logits'] = label_logits

        output_dict['probs'] = torch.nn.functional.softmax(label_logits, dim=1)
        output_dict['label'] = label_logits.argmax(1)

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