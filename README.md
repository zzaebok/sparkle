# SPARKLE: Enhancing SPARQL Generation with Direct KG Integration in Decoding

SPARKLE is a fast and simple KBQA framework that uses one single sequence-to-sequence model.
The structural information in knowledge base is directly utilized within constrained decoding.

<p align="center">
  <img src="./figures/method_trie.png" height="100" width="45.9%" />
  <img src="./figures/method_link.png" height="100" width="44.1%" /> 
</p>

## Installation

We used python 3.8 for the experiments.

```bash
$ pip install -e .
# You may see: WARNING like "The scripts inv and invoke are installed in '/home/user/.local/bin' which is not on PATH.", Add the specified directory to the PATH.
$ export PATH=$PATH:/home/user/.local/bin
```

Since preprocessing KG and training KBQA models require many arguments, we use [`invoke`](https://www.pyinvoke.org/) to orgnaize python code.
Please refer to `tasks.py` for details of executable commands.

You should set proper `DATA_PATH` in `.env` file. Default value is set to `data`.

## Quickstart

We provide an inference example using pre-trained weights from LCQuAD 1.0.
Loading preprocessed data can take some time.

You can Download pretrained weights and preprocessed datasets for each dataset in the below links.
There are two `output` (checkpoint) and `preprocessed` subdirectories.
Place them in `data/output/{dataset}`and `data/preprocessed/{dataset}`.

- [LCQuAD 1.0](https://storage.googleapis.com/sparkle-code/lcquad1.zip)
- [SimpleQuestions](https://storage.googleapis.com/sparkle-code/simplequestions.zip)
- [WebQSP](https://storage.googleapis.com/sparkle-code/webqsp.zip)

```python
import pickle
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from sparkle.sparql_token_generator import SparqlTokenGenerator
from sparkle.utils import UniqueToken, encode_fn

# Load checkpoints
tokenizer = AutoTokenizer.from_pretrained("data/output/lcquad1")
model = AutoModelForSeq2SeqLM.from_pretrained("data/output/lcquad1")
config = AutoConfig.from_pretrained("data/output/lcquad1")

# Load KG related preprocessed files
with open("data/preprocessed/lcquad1/entity_relation_tries.pkl", "rb") as f:
    entity_trie, relation_trie = pickle.load(f)
with open("data/preprocessed/lcquad1/link_dictionaries.pkl", "rb") as f:
    head_rel, rel_tail = pickle.load(f)

# Refer to tasks.py to define `sparql_vocab` for each dataset
sparql_vocab = {
    "SELECT": encode_fn("SELECT DISTINCT ", tokenizer),
    "ASK": encode_fn("ASK ", tokenizer),
    "VARS": [encode_fn(token, tokenizer) for token in ["var_uri ", "var_x "]],
}
for token in UniqueToken:
    sparql_vocab[token.name] = encode_fn(token.value, tokenizer)

# SparqlTokenGenerator controls constrained decoding
sparql_token_generator = SparqlTokenGenerator(
    tokenizer=tokenizer,
    entity_trie=entity_trie,
    relation_trie=relation_trie,
    head_rel=head_rel,
    relation_tail=rel_tail,
    sparql_vocab=sparql_vocab,
    config=config,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

inference_start = time.time()
inputs = tokenizer(
    ["Name the movies distributed by Warner Bros. and directed by Ridley Scott."],
    max_length=256,
    return_tensors="pt",
    padding=True,
    truncation=True,
).to(device)
generated_tokens = model.generate(
    prefix_allowed_tokens_fn=lambda batch_id, input_ids: sparql_token_generator.generate(
        batch_id,
        input_ids,
        verbose=False,
    ),
    num_beams=7,
    num_return_sequences=1,
    max_length=128,
    no_repeat_ngram_size=None,
    **inputs,
)
draft_sparqls = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# SELECT DISTINCT var_uri WHERE brack_open var_uri [ distributor : property ] [ warner_bros. : resource ] sep_dot var_uri [ director : property ] [ ridley_scott : resource ] sep_dot var_uri [ type : rdf ] [ film : ontology ] brack_close 
print(draft_sparqls)
# Postprocess draft_sparqls using dataset class, see details in `evaluate_sparkle.py`
print("Inference time: ", time.time() - inference_start)
```

## Preprocessing

### Knowledge Graph

- Preprocess knowledge graphs to extract entity(name, type), relation(name), and connectivity information.
- Output files are `entities.pkl`, `relations.pkl`, and `kg.pkl`.
- See details in `scripts/preprocessing/kg/preprocess_{kg}.py`.

```bash
$ python preprocess_wikidata.py --wikidata_path={} --output_dir={}
$ python preprocess_dbpedia.py --dbpedia_path={} --output_dir={}
$ python preprocess_freebase.py --freebase_path={} --output_dir={}
```

Each argument {kg}_path can be provided by

**1. Wikidata**
- We use Wikidata 2017-12-27 dump from [here](https://zenodo.org/records/1211767)
- Also refer to `scripts/preprocessing/utils/split_wikidata.py` for the splitting.

**2. DBPedia**
- We use DBPedia 2016-04 dump from [here](https://downloads.dbpedia.org/2016-04/core-i18n/en/)
- Merge `ttl.bz2` files into one `ttl.bz2`: We specifically use `infobox_properties_en.ttl.bz2`, `instance_types_en.ttl.bz2`, `instance_types_transitive_en.ttl.bz2`, `labels_en.ttl.bz2`, `mappingbased_literals_en.ttl.bz2`, `mappingbased_objects_en.ttl.bz2` and `specific_mappingbased_properties_en.ttl.bz2`.

**3. Freebase**
- We use Freebase dump from [here](https://developers.google.com/freebase?hl=en)
- Also refer to `scripts/preprocessing/utils/split_freebase.py` for the splitting.


### KBQA dataset

- Preprocess KBQA dataset in advance to standardize representations of entities and relations in SPARQL queries.
- Output files are
    - Identifier related files: `entity_relation_token_dict.pkl`, and `entity_relation_tries.pkl`.
    - Linkage file: `link_dictionaries.pkl`
    - Training related files: `train_data.json`, `eval_data.json`, and `test_data.json`
        - examples:
        ```json
        {
            "text": "Name the office of Richard Coke ?",
            "sparql": "SELECT DISTINCT ?uri WHERE {<http://dbpedia.org/resource/Richard_Coke> <http://dbpedia.org/property/office> ?uri }",
            "sparql_proc": "SELECT DISTINCT var_uri WHERE brack_open [ richard_coke : resource ] [ office : property ] var_uri brack_close ",
            "answers": [
                "http://dbpedia.org/resource/United_States_Senate"
            ]
        }
        ```
- See details in `scripts/preprocessing/dataset/preprocess_dataset.py`.
- Available `{kbqa_dataset}`s are `SimpleQuestions`, `LCQuAD1`, and `WebQSP`. (Refer to `scripts/preprocessing/dataset/{kbqa_dataset}.py`)

```bash
$ invoke preprocess-{kbqa_dataset}
```

Argument `dataset_dir` sources are

**1. SimpleQuestions-Wiki**
- Download SimpleQuestions-Wiki answerable dataset from [here](https://github.com/askplatypus/wikidata-simplequestions/tree/master/qald-format)
- Files are located in `data/original/simplequestions`
- We filter out questions that are not supported by Wikidata dump we used (refer to `sparkle/dataset/simplequestions.py`)

**2. LCQuAD 1.0**
- Download LCQuAD 1.0 dataset from [here](https://github.com/AskNowQA/LC-QuAD)
- We attached answers by sending queries to knowledge graph (see `Virtuoso Setup` section below)
- Files are located in `data/original/lcquad1`

**3. WebQSP**
- Download WebQSP dataset following instructions on [here](https://github.com/awslabs/decode-answer-logical-form)
- We use the same split as detailed in [here](https://github.com/awslabs/decode-answer-logical-form/tree/main/DecAF/Datasets/QA/WebQSP)
- Files are located in `data/original/webqsp`

## Training

- We made a training script based on Huggingface official Machine Translation [script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py).
- See details in `scripts/training/train_sparkle.py`.
- `sparkle/sparql_token_generator.py` is responsible for constrained decoding.
- To reproduce the experimental results, please use the arguments set in `tasks.py`. (batch 4 is set with the number of gpu of 8 = total should be 32)

```bash
$ invoke train-{kbqa_dataset}
# example
$ invoke train-simplequestions --model-name-or-path=facebook/bart-large
```

## Evaluation

- See details in `scripts/evaluation/generate_sparkle.py` (for model inference).
- See details in `scripts/evaluation/evaluate_sparkle.py` (for querying and performance measurement).

```bash
$ invoke generate-{kbqa_dataset}
# example
$ invoke generate-simplequestions --model-name-or-path=data/output/simplequestions/lr5e-05_batch4_beam4
```

```bash
$ invoke evaluate-{kbqa_dataset}
# example
$ invoke evaluate-simplequestions --predict-file=data/output/simplequestions/lr5e-05_batch4_beam4/generated/generate_batch8_beam10.json
```

## Virtuoso Setup

**1. Wikidata**
We use Wikidata Query Service https://query.wikidata.org/

**2. DBpedia**
To set up the DBpedia Virtuoso, please refer to the installation guide available at [here](https://github.com/harsh9t/Dockerised-DBpedia-Virtuoso-Endpoint-Setup-Guide).
LC-QuAD 1.0 that we used for the experiment can be answered based on the 2016-04 version.

**3. Freebase**
To set up the Freebase Virtuoso, please refer to the installation guide available at [here](https://github.com/dki-lab/Freebase-Setup).
