local train_size = 4957;
local batch_size = 1;
local gradient_accumulation_batch_size = 16;
local num_epochs = 4;
local learning_rate = 1e-5;
local weight_decay = 0.1;
local warmup_ratio = 0.06;
local transformer_model = "roberta-large";
local transformer_weights_model = "/models/model.tar.gz";
local dataset_dir = "/inputs/ARC-OBQA-RegLivEnv/";
local dataset_dir_out = "/output/OBQA-IR10V8c";
local cuda_device = 0;
local es_cache = "/caches/ESCache-OBQA-IR10V8c.jsonl.gz";
local random_seed = 789;

{
  "random_seed": random_seed,
  "numpy_seed": random_seed*7,
  "pytorch_seed": random_seed*23,
  "dataset_reader": {
    "type": "transformer_mc_qa",
    "document_retriever": {
      "type": "elastic_search_qa",
      "host": "tushark-data.ari.ai2",
      "port": 9200,
      "indices": "openbook_noquotes",
      "query_format": "aristo-qa",
      "num_retrievals": 30,
      "max_question_length": 1000,
      "max_document_length": 500,
      "cache_files": es_cache
    },
    "context_format": {"mode": "concat-q-all-a", "num_sentences": 5, "max_sentence_length": 300},
    "dataset_dir_out": dataset_dir_out,
    "sample": -1,
    "skip_id_regex": "(RegLivEnv|ARC).*",
    "pretrained_model": transformer_model,
    "max_pieces": 256
  },
  "datasets_for_vocab_creation": [],
  "train_data_path": dataset_dir + "train.jsonl",
  "validation_data_path": dataset_dir + "dev.jsonl",
  "test_data_path": dataset_dir + "test.jsonl",
  "evaluate_on_test": true,
  "evaluate_custom": {
      "metadata_fields": "id,question_text,choice_text_list,correct_answer_index,answer_index,label_logits,label_probs"
  },
  "model": {
    "type": "roberta_mc_qa",
    "transformer_weights_model": transformer_weights_model,
    "pretrained_model": transformer_model
  },
  "iterator": {
    "type": "basic",
    "batch_size": batch_size
  },
  "trainer": {
    "optimizer": {
      "type": "adam_w",
      "betas": [0.9, 0.98],
      "weight_decay": weight_decay,
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": learning_rate
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": num_epochs,
      "cut_frac": warmup_ratio,
      "num_steps_per_epoch": std.ceil(train_size / gradient_accumulation_batch_size),
    },
    "validation_metric": "+accuracy",
    "num_serialized_models_to_keep": 1,
    "should_log_learning_rate": true,
    "gradient_accumulation_batch_size": gradient_accumulation_batch_size,
    "grad_clipping": 1.0,
    "num_epochs": num_epochs,
    "cuda_device": cuda_device
  }
}
