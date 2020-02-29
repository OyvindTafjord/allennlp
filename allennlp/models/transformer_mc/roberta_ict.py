from typing import Dict, Optional, List, Any

import logging
from transformers.modeling_roberta import RobertaClassificationHead, RobertaConfig, RobertaModel
import re
import torch

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, F1Measure

logger = logging.getLogger(__name__)

# Essentially a copy of roberta_classifier, but keeping separate for flexibility
@Model.register("roberta_ict_classifier")
class RobertaICTClassifierModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 final_dim: int = 128,
                 transformer_weights_model: str = None,
                 reset_classifier: bool = False,
                 per_sentence_loss: bool = True,
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
        projection_input_dim = self._output_dim
        self._final_dim= final_dim
        projection_output_dim = self._final_dim
        transformer_config.num_labels = projection_output_dim
        self._context_projection = RobertaClassificationHead(transformer_config)
        self._sentence_projection = RobertaClassificationHead(transformer_config)

        self._per_sentence_loss = per_sentence_loss
        if self._per_sentence_loss:
            self._classifier_bias = torch.nn.Parameter(torch.zeros(1))
            self._accuracy = BooleanAccuracy()
            self._loss = torch.nn.BCEWithLogitsLoss()
            self._f1measure = F1Measure(positive_label=1)
        else:
            self._classifier_bias = None
            self._accuracy = CategoricalAccuracy()
            self._loss = torch.nn.CrossEntropyLoss()
            self._f1measure = None

        self._debug = 2
        self._padding_value = 1  # The index of the RoBERTa padding token

    def forward(self,
                context: Dict[str, torch.LongTensor],
                sentence: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        context_tokens = context['tokens']['token_ids']
        context_mask = (context_tokens != self._padding_value).long()
        sentence_tokens = sentence['tokens']['token_ids']
        sentence_mask = (sentence_tokens != self._padding_value).long()
        batch_size = context_tokens.size(0)

        if self._debug > 0:
            logger.info(f"batch_size = {batch_size}")
            logger.info(f"context_mask = {context_mask}")
            logger.info(f"context_tokens.size() = {context_tokens.size()}")
            logger.info(f"context_tokens = {context_tokens}")
            logger.info(f"segment_ids = {segment_ids}")
            logger.info(f"label = {label}")

        # Segment ids are not used by RoBERTa
        context_outputs = self._transformer_model(input_ids=context_tokens,
                                                      # token_type_ids=segment_ids,
                                                      attention_mask=context_mask)
        context_projection = self._context_projection(context_outputs[0])
        sentence_outputs = self._transformer_model(input_ids=sentence_tokens,
                                                      # token_type_ids=segment_ids,
                                                      attention_mask=sentence_mask)
        sentence_projection = self._sentence_projection(sentence_outputs[0])

        # Batch vector dot product
        label_logits = (context_projection * sentence_projection).sum(dim=1)

        output_dict = {}
        if label is not None:
            # use negative label to signal auto-negatives per batch
            label_used = label
            if label[0] < 0.0 or not self._per_sentence_loss:
                # outer product of vector dot products
                label_logits = torch.einsum("ik,jk->ijk", context_projection, sentence_projection).sum(dim=2)
                if self._per_sentence_loss:
                    label_logits = label_logits + self._classifier_bias
                    label_used = torch.eye(label_logits.size(0), device=label_logits.device)
                    label_logits = label_logits.view(-1)
                    label_used = label_used.view(-1)
                else:
                    label_used = torch.arange(0, batch_size, device=label_logits.device)

            if self._debug > 0:
                logger.info(f"label_logits = {label_logits}")
                logger.info(f"label_used = {label_used}")

            loss = self._loss(label_logits, label_used)
            if self._per_sentence_loss:
                self._accuracy(label_logits > 0, label_used)
                # somewhat of a hack to use existing f1 measure
                self._f1measure(torch.stack([-label_logits, label_logits]).t(), label_used)
                output_dict['true_label'] = label_used.view(batch_size, -1)
                output_dict['label'] = (label_logits > 0).view(batch_size, -1)
                output_dict['label_logits'] = label_logits.view(batch_size, -1)
                output_dict['label_probs'] = torch.sigmoid(label_logits).view(batch_size, -1)
            else:
                self._accuracy(label_logits, label_used)
                output_dict['true_label'] = label_used
                output_dict['label'] = label_logits.argmax(1)
                output_dict['label_logits'] = label_logits
                output_dict['label_probs'] = torch.nn.functional.softmax(label_logits, dim=1)
            output_dict["loss"] = loss


        if self._debug > 0:
            logger.info(output_dict)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self._f1measure is not None:
            precision, recall, f1 = self._f1measure.get_metric(reset)
            return {
                'accuracy': self._accuracy.get_metric(reset),
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        else:
            return {
                'accuracy': self._accuracy.get_metric(reset)
            }

    def context_projection(self, context: Dict[str, torch.LongTensor]) -> torch.Tensor:
        context_tokens = context['tokens']['token_ids']
        context_mask = (context_tokens != self._padding_value).long()
        context_outputs = self._transformer_model(input_ids=context_tokens,
                                                      # token_type_ids=segment_ids,
                                                      attention_mask=context_mask)
        context_projection = self._context_projection(context_outputs[0])
        return context_projection

    def sentence_projection(self, sentence: Dict[str, torch.LongTensor]) -> torch.Tensor:
        sentence_tokens = sentence['tokens']['token_ids']
        sentence_mask = (sentence_tokens != self._padding_value).long()
        sentence_outputs = self._transformer_model(input_ids=sentence_tokens,
                                                      # token_type_ids=segment_ids,
                                                      attention_mask=sentence_mask)
        sentence_projection = self._sentence_projection(sentence_outputs[0])
        return sentence_projection

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