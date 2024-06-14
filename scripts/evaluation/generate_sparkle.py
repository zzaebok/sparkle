import logging
import pickle
import os
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, Dataset
from typing import List
from dataclasses import dataclass, field
import torch
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    set_seed,
)
from sparkle.dataset.webqsp import WebQSP
from sparkle.dataset.lcquad1 import LCQuAD1
from sparkle.dataset.simplequestions import SimpleQuestions
from sparkle.utils import UniqueToken, encode_fn
from sparkle.sparql_token_generator import (
    SparqlTokenGenerator, AblationSparqlTokenGenerator
)
import sys
sys.path.append(os.path.abspath("."))
from scripts.preprocessing.utils.util import load_multi_pickles

logger = logging.getLogger(__name__)

# arguments
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="/path/to/model/checkpoint",
        metadata={"help": "Path to pretrained model"},
    )


@dataclass
class DataEvaluateArguments:
    """
    Arguments training to what data we are going to input our model for training and eval.
    """

    data_file: str = field(
        default="/path/to/json/file",
        metadata={"help": "A test data file to evaluate the metrics on a json file."},
    )
    batch_size: int = field(
        default=1,
    )
    num_beams: int = field(
        default=1,
    )
    max_length: int = field(
        default=256,
        metadata={"help": "The maximum total input sequence length for tokenization."},
    )
    generation_max_length: int = field(
        default=128,
        metadata={"help": "The maximum generation sequence length."},
    )
    ablation_mode: str = field(
        default="full",
        metadata={
            "help": (
                "Ablation mode"
            )
        },
    )
    output_path: str = field(
        default="/path/to/output",
        metadata={
            "help": (
                "Output json path"
            )
        },
    )

    def __post_init__(self):
        assert self.ablation_mode in ["full", "retrieval", "wocd"]


@dataclass
class KGArguments:
    """
    Knowledge graph arguments.
    """

    kg_preprocess_dir: str = field(
        default="/path/to/preprocessed/dataset",
        metadata={
            "help": "Directory where entity_relation_token_dict.pkl, entity_relation_tries.pkl, and link_dictionaries.pkl exist"
        },
    )
    var_tokens: List[str] = field(
        default_factory=lambda: [
            "var_uri ",
            "var_x ",
        ],
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

def main():
    set_seed(2023)
    parser = HfArgumentParser((ModelArguments, DataEvaluateArguments, KGArguments))
    model_args, data_args, kg_args = parser.parse_args_into_dataclasses()

    input_path_list = [os.path.join(kg_args.kg_preprocess_dir, filename) for filename in ["entity_relation_tries.pkl", "link_dictionaries.pkl"]]
    tries, link_dictionaries = load_multi_pickles(input_path_list)
    entity_trie, relation_trie = tries
    head_rel, rel_tail = link_dictionaries

    with open(os.path.join(data_args.data_file), "r") as f:
        test_data = json.load(f)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # sparql_vocab controls constrained decoding
    sparql_vocab = {
        "SELECT": encode_fn(kg_args.select_token, tokenizer),
        "ASK": encode_fn(kg_args.ask_token, tokenizer),
        "VARS": [encode_fn(token, tokenizer) for token in kg_args.var_tokens],
    }

    for token in UniqueToken:
        sparql_vocab[token.name] = encode_fn(token.value, tokenizer)

    print(sparql_vocab)

    if data_args.ablation_mode == "retrieval":
        sparql_token_generator = AblationSparqlTokenGenerator(
            tokenizer=tokenizer,
            entity_trie=entity_trie,
            relation_trie=relation_trie,
            head_rel=None,
            relation_tail=None,
            sparql_vocab=sparql_vocab,
            config=AutoConfig.from_pretrained(model_args.model_name_or_path)
        )
    elif data_args.ablation_mode == "wocd":
        sparql_token_generator = None
    else:
        sparql_token_generator = SparqlTokenGenerator(
            tokenizer=tokenizer,
            entity_trie=entity_trie,
            relation_trie=relation_trie,
            head_rel=head_rel,
            relation_tail=rel_tail,
            sparql_vocab=sparql_vocab,
            config=AutoConfig.from_pretrained(model_args.model_name_or_path)
        )

    data_loader = DataLoader(
        SPARKLEDataset(test_data),
        batch_size=data_args.batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )  # Time evaluation

    times_taken = []
    result = {
        "inference_time": 0,
        "result": []
    }

    with torch.no_grad():
        for num_b, (texts, answers, gt_sparqls) in tqdm(enumerate(data_loader)):
            starter.record()
            inputs = tokenizer(
                texts,
                max_length=data_args.max_length,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            if sparql_token_generator is not None:
                generated_tokens = model.generate(
                    prefix_allowed_tokens_fn=lambda batch_id, input_ids: sparql_token_generator.generate(
                        batch_id,
                        input_ids,
                        verbose=False,
                    ),
                    num_beams=data_args.num_beams,
                    num_return_sequences=data_args.num_beams,
                    max_length=data_args.generation_max_length,
                    no_repeat_ngram_size=None,
                    **inputs,
                )
            else:
                generated_tokens = model.generate(
                    num_beams=data_args.num_beams,
                    num_return_sequences=data_args.num_beams,
                    max_length=data_args.generation_max_length,
                    no_repeat_ngram_size=None,
                    **inputs,
                )
            ender.record()
            torch.cuda.synchronize()
            inference_time = starter.elapsed_time(ender)
            times_taken.append(inference_time)

            draft_sparqls = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            num_batches = min(data_args.batch_size, len(texts))
            for n in range(num_batches):
                result["result"].append(
                    {
                        "text": texts[n],
                        "answers": answers[n],
                        "gt_sparql": gt_sparqls[n],
                        "candidates": draft_sparqls[
                            data_args.num_beams * n : data_args.num_beams * (n + 1)
                        ]
                    }
                )
    result["inference_time"] = sum(times_taken) / len(times_taken)

    if not os.path.exists(os.path.join(model_args.model_name_or_path, "generated")):
        os.makedirs(os.path.join(model_args.model_name_or_path, "generated"), exist_ok=True)
    with open(
        data_args.output_path, "w"
    ) as f:
        json.dump(result, f, indent=4)
    

class SPARKLEDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]["text"]
        y = self.data[index]["answers"]
        sparql = self.data[index]["sparql_proc"]
        return x, y, sparql

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    main()