import sys
import logging
import pickle
import random
import torch.nn as nn
from typing import Optional, Dict, Union, Any, List, Tuple
from dataclasses import dataclass, field
import evaluate
import torch
from collections import Counter
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    set_seed,
)
from sparkle.sparql_token_generator import SparqlTokenGenerator
from sparkle.utils import UniqueToken, encode_fn

logger = logging.getLogger(__name__)


class SkipValidationCallback(TrainerCallback):
    def __init__(self, skip_steps=2000):
        self.skip_steps = skip_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        if state.global_step < self.skip_steps:
            control.should_evaluate = False
            control.should_save = False


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/bart-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments training to what data we are going to input our model for training and eval.
    """

    train_file: str = field(
        default="data/preprocessed/lcquad1/train_data.json",
        metadata={"help": "The input training data file (a json)."},
    )
    validation_file: str = field(
        default="data/preprocessed/lcquad1/eval_data.json",
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics on a json file."
        },
    )
    test_file: Optional[str] = field(
        default="data/preprocessed/lcquad1/test_data.json",
        metadata={
            "help": "An optional input test data file to evaluate the metrics on a json file."
        },
    )
    max_length: int = field(
        default=256,
        metadata={"help": "The maximum total input sequence length for tokenization."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a training or validation file.")

        if self.train_file is not None:
            extension = self.train_file.rsplit(".", maxsplit=1)[-1]
            assert extension == "json", "`train_file` should be a json file."
        if self.validation_file is not None:
            extension = self.validation_file.rsplit(".", maxsplit=1)[-1]
            assert extension == "json", "`validation_file` should be a json file."
        if self.test_file is not None:
            extension = self.test_file.rsplit(".", maxsplit=1)[-1]
            assert extension == "json", "`test_file` should be a json file."


@dataclass
class KGArguments:
    """
    Knowledge graph arguments.
    """

    ent_rel_token_dict_path: Optional[str] = field(
        default="data/preprocessed/webqsp/entity_relation_token_dict.pkl",
        metadata={
            "help": "A pickle file containing [entity_token_dict, relation_token_dict]."
        },
    )
    ent_rel_tries_path: Optional[str] = field(
        default="data/preprocessed/webqsp/entity_relation_tries.pkl",
        metadata={"help": "A pickle file containing [entity_trie, relation_trie]."},
    )
    link_dicts_path: Optional[str] = field(
        default="data/preprocessed/webqsp/link_dictionaries.pkl",
        metadata={
            "help": "A pickle file contraining link dictionaries [head_rel, rel_tail]."
        },
    )
    var_tokens: List[str] = field(
        default_factory=lambda: ["var_x ", "var_uri "],
        metadata={"help": "Variable tokens that can appear in sparql"},
    )
    select_token: str = field(
        default="SELECT DISTINCT ",
        metadata={"help": "SELECT token that can appear in sparql"},
    )
    ask_token: str = field(
        default="ASK ",
        metadata={"help": "ASK token that can appear in sparql"},
    )


# pylint:disable = too-many-statements, no-member, unbalanced-tuple-unpacking
def main():
    # see https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py for training arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, KGArguments)
    )
    model_args, data_args, training_args, kg_args = parser.parse_args_into_dataclasses()

    # Setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.warning("device: %s, n_gpu: %s", training_args.device, training_args.n_gpu)

    set_seed(2023)

    # Prepare data
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    # raw_datasets = load_dataset(
    #     extension,
    #     data_files=data_files,
    # )

    import json
    from datasets import Dataset

    raw_datasets = {}
    with open(data_files["train"], "r") as f:
        train_dict = [
            {k: v for k, v in data.items() if k in ["text", "sparql_proc"]}
            for data in json.load(f)
        ]
        train_dict = {k: [dic[k] for dic in train_dict] for k in train_dict[0]}
        raw_datasets["train"] = Dataset.from_dict(train_dict)
    with open(data_files["validation"], "r") as f:
        validation_dict = [
            {k: v for k, v in data.items() if k in ["text", "sparql_proc"]}
            for data in json.load(f)
        ]
        validation_dict = {
            k: [dic[k] for dic in validation_dict] for k in validation_dict[0]
        }
        raw_datasets["validation"] = Dataset.from_dict(validation_dict)
    with open(data_files["test"], "r") as f:
        test_dict = [
            {k: v for k, v in data.items() if k in ["text", "sparql_proc"]}
            for data in json.load(f)
        ]
        test_dict = {k: [dic[k] for dic in test_dict] for k in test_dict[0]}
        raw_datasets["test"] = Dataset.from_dict(test_dict)

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.warning(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return None

    # Load pretrained model
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
    )

    # Add individual tokens
    tokenizer.add_tokens([token.value for token in UniqueToken])
    tokenizer.add_tokens(kg_args.var_tokens + [kg_args.select_token, kg_args.ask_token])
    model.resize_token_embeddings(len(tokenizer))

    # sparql_vocab controls constrained decoding
    sparql_vocab = {
        "SELECT": encode_fn(kg_args.select_token, tokenizer),
        "ASK": encode_fn(kg_args.ask_token, tokenizer),
        "VARS": [encode_fn(token, tokenizer) for token in kg_args.var_tokens],
    }

    for token in UniqueToken:
        sparql_vocab[token.name] = encode_fn(token.value, tokenizer)

    print(sparql_vocab)

    with open(kg_args.ent_rel_tries_path, "rb") as f:
        ent_rel_tries = pickle.load(f)

    entity_trie, relation_trie = ent_rel_tries

    # Generating KG dictionary
    with open(kg_args.link_dicts_path, "rb") as f:
        link_dicts = pickle.load(f)

    head_rel = link_dicts[0]
    relation_tail = link_dicts[1]

    # Sparql token generator
    sparql_token_generator = SparqlTokenGenerator(
        tokenizer=tokenizer,
        entity_trie=entity_trie,
        relation_trie=relation_trie,
        head_rel=head_rel,
        relation_tail=relation_tail,
        sparql_vocab=sparql_vocab,
        config=AutoConfig.from_pretrained(model_args.model_name_or_path),
    )

    def preprocess_function(examples, tokenizer):
        texts = examples["text"]
        sparqls = examples["sparql_proc"]

        model_inputs = tokenizer(
            texts,
            max_length=data_args.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        labels = tokenizer(
            sparqls,
            max_length=data_args.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        # T5 / BART
        if model_args.model_name_or_path != "facebook/genre-kilt":
            model_inputs["labels"] = labels["input_ids"]
        # Pre-trained GENRE
        else:
            model_inputs["labels"] = labels["input_ids"][:, 1:]

        return model_inputs

    # *_dataset consists of keys (`input_id`, `attention_mask`, and `labels`)
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                lambda x: preprocess_function(x, tokenizer),
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on train dataset",
            )
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                lambda x: preprocess_function(x, tokenizer),
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                lambda x: preprocess_function(x, tokenizer),
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on prediction dataset",
            )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
    )

    metric = evaluate.load("exact_match")

    def postprocess_sentence(sentences):
        return sentences

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # https://github.com/huggingface/transformers/pull/10046/files
        preds[preds == -100] = tokenizer.pad_token_id
        labels[labels == -100] = tokenizer.pad_token_id

        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        final_preds = postprocess_sentence(decoded_preds)
        final_labels = postprocess_sentence(decoded_labels)

        for i in random.sample(range(len(final_preds)), k=min(2, len(final_preds))):
            try:
                print("====")
                print(raw_datasets["validation"]["text"][i])
                print("---")
                print(final_preds[i])
                print("---")
                print(final_labels[i], flush=True)
            except IndexError:
                continue

        result = metric.compute(predictions=final_preds, references=final_labels)

        precisions = []
        recalls = []
        f1s = []

        for pred, label in zip(final_preds, final_labels):
            pred_tokens = pred.split()
            label_tokens = label.split()
            common = Counter(pred_tokens) & Counter(label_tokens)
            num_same = sum(common.values())

            if len(pred_tokens) == 0:
                precision = 0
            else:
                precision = num_same / len(pred_tokens)
            precisions.append(precision)
            recall = num_same / len(label_tokens)
            recalls.append(recall)
            if precision == 0 and recall == 0:
                f1s.append(0)
            else:
                f1s.append((2 * precision * recall) / (precision + recall))

        result = {
            "exact_match": result["exact_match"],
            "precision": sum(precisions) / len(final_preds),
            "recall": sum(recalls) / len(final_preds),
            "f1": sum(f1s) / len(final_preds),
        }

        return result

    # pylint: disable=unused-argument
    # https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13?u=zzaebok
    def preprocess_logits_for_metrics(preds, labels):
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = torch.argmax(preds, axis=-1)

        return preds

    trainer = SparkleTrainer(
        sparql_token_generator=sparql_token_generator,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[SkipValidationCallback(skip_steps=2000)],
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # TRAINING
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # EVALUATION
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.max_length, metric_key_prefix="eval"
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # PREDICT
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=data_args.max_length,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    return results


class SparkleTrainer(Seq2SeqTrainer):
    def __init__(self, sparql_token_generator, *args, **kwargs):
        self.sparql_token_generator = sparql_token_generator
        super().__init__(*args, **kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()
        gen_kwargs = self._gen_kwargs.copy()
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
        ):
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.model.config.num_beams
        )
        default_synced_gpus = False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"]
            if gen_kwargs.get("synced_gpus") is not None
            else default_synced_gpus
        )

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}

        generated_tokens = self.model.generate(
            prefix_allowed_tokens_fn=lambda batch_id, input_ids: self.sparql_token_generator.generate(
                batch_id, input_ids
            ),
            num_beams=gen_kwargs["num_beams"],
            num_return_sequences=1,
            max_length=gen_kwargs["max_length"],
            no_repeat_ngram_size=None,
            **inputs,
        )

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_length
            )
        elif (
            gen_config.max_new_tokens is not None
            and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_new_tokens + 1
            )

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = (
                        self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    )
                else:
                    loss = (
                        (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
                        .mean()
                        .detach()
                    )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif (
                gen_config.max_new_tokens is not None
                and labels.shape[-1] < gen_config.max_new_tokens + 1
            ):
                labels = self._pad_tensors_to_max_len(
                    labels, gen_config.max_new_tokens + 1
                )
        else:
            labels = None
        return loss, generated_tokens, labels


if __name__ == "__main__":
    main()
