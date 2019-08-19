local train_size = 8992;
local batch_size = 1;
local gradient_accumulation_batch_size = 32;
local num_epochs = 4;
local learning_rate = 1e-5;
local transformer_model = "xlnet-large-cased";
local transformer_weights_model = "/models/xlnet_mc_race-try1-model.tar.gz";
local dataset_dir = "/inputs/ARC-OBQA-RegLivEnv-IR10V2/";

{
  "dataset_reader": {
    "type": "transformer_mc_qa",
    "sample": -1,
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
    "type": "xlnet_mc_qa",
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
      "weight_decay": 0.01,
      "parameter_groups": [[["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}]],
      "lr": learning_rate
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": num_epochs,
      "cut_frac": 0.1,
      "num_steps_per_epoch": std.ceil(train_size / gradient_accumulation_batch_size),
    },
    "validation_metric": "+accuracy",
    "num_serialized_models_to_keep": 1,
    "should_log_learning_rate": true,
    "gradient_accumulation_batch_size": gradient_accumulation_batch_size,
    "grad_clipping": 1.0,
    "num_epochs": num_epochs,
    "cuda_device": 0
  }
}
