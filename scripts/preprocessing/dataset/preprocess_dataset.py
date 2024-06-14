import time
import pickle
import re
import argparse
import os
import random
import json
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath("."))
from scripts.preprocessing.utils.util import load_multi_pickles
from transformers import AutoTokenizer, PreTrainedTokenizer
from sparkle.utils import UniqueToken
from sparkle.trie import Trie
from sparkle.dataset.webqsp import WebQSP
from sparkle.dataset.lcquad1 import LCQuAD1
from sparkle.dataset.simplequestions import SimpleQuestions


# arguments
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="webqsp",
        choices=["webqsp", "lcquad1", "simplequestions"],
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/original/webqsp",
        help="Directory where kbqa dataset are located",
    )
    parser.add_argument(
        "--kg_preprocess_dir",
        type=str,
        default="data/preprocessed/freebase",
        help="Directory where preprocessed kg files are saved. (should have entities.pkl, relations.pkl, and kg.pkl).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/bart-large",
        help="Huggingface model name for building tries and dictionaries",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/preprocessed/webqsp",
        help="Directory where output files are saved",
    )

    args = parser.parse_args()
    return args


def main(args):
    random.seed(2023)
    dataset = None

    load_start_time = time.time()
    input_path_list = [
        os.path.join(args.kg_preprocess_dir, filename)
        for filename in ["entities.pkl", "relations.pkl", "kg.pkl"]
    ]
    entities, relations, kg = load_multi_pickles(input_path_list)
    print("KG data loaded. time - {}".format(time.time() - load_start_time))

    # dataset select
    if args.dataset == "webqsp":
        dataset = WebQSP(args.dataset_dir)
    elif args.dataset == "lcquad1":
        dataset = LCQuAD1(args.dataset_dir)
    elif args.dataset == "simplequestions":
        dataset = SimpleQuestions(args.dataset_dir, kg[0], kg[1])
    else:
        raise Exception

    # entities, relations original format to dataset format
    entities, relations, kg = dataset.change_format(entities, relations, kg)

    sparqls = [x["sparql"] for x in dataset.train_dataset]
    sparqls.extend([x["sparql"] for x in dataset.eval_dataset])
    sparqls.extend([x["sparql"] for x in dataset.test_dataset])
    sparql_ents = set()
    sparql_rels = set()
    for sparql in sparqls:
        for token in sparql.split():
            if dataset.is_entity(token):
                sparql_ents.add(token)
            if dataset.is_relation(token):
                sparql_rels.add(token)

    # check sanity - all entity_iri, relation_iri should be in the entities and relations
    # if not, use iri as their label
    for entity in sparql_ents:
        if entity not in entities:
            print(f"entity {entity} not in entities.pkl")
            entities[entity] = {"label": entity}

    for relation in sparql_rels:
        if relation not in relations:
            print(f"relation {relation} not in relations.pkl")
            relations[relation] = relation

    print("Sanity check finished.")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens([token.value for token in UniqueToken])

    entity_token_dict, entity_trie = build_entity_trie(entities, tokenizer)
    relation_token_dict, relation_trie = build_relation_trie(relations, tokenizer)
    print("Entity, Relation Trie created.")

    with open(
        os.path.join(args.output_dir, "entity_relation_token_dict.pkl"), "wb"
    ) as f:
        pickle.dump([entity_token_dict, relation_token_dict], f, protocol=-1)
    with open(os.path.join(args.output_dir, "entity_relation_tries.pkl"), "wb") as f:
        pickle.dump([entity_trie, relation_trie], f, protocol=-1)

    # add sparql_proc column
    train_dataset = change_entity_relation_repr(
        dataset.train_dataset,
        dataset,
        entity_token_dict,
        relation_token_dict,
    )
    eval_dataset = change_entity_relation_repr(
        dataset.eval_dataset, dataset, entity_token_dict, relation_token_dict
    )
    test_dataset = change_entity_relation_repr(
        dataset.test_dataset, dataset, entity_token_dict, relation_token_dict
    )

    # save all files
    with open(os.path.join(args.output_dir, "train_data.json"), "w") as f:
        json.dump(train_dataset, f, indent=4)
    with open(os.path.join(args.output_dir, "eval_data.json"), "w") as f:
        json.dump(eval_dataset, f, indent=4)
    with open(os.path.join(args.output_dir, "train_eval_data.json"), "w") as f:
        json.dump(train_dataset + eval_dataset, f, indent=4)
    with open(os.path.join(args.output_dir, "test_data.json"), "w") as f:
        json.dump(test_dataset, f, indent=4)

    # link
    link_dicts = build_link_dictionaries(
        kg, entity_token_dict, relation_token_dict, entity_trie, relation_trie
    )
    print("Link dictionaries are created.")

    with open(os.path.join(args.output_dir, "link_dictionaries.pkl"), "wb") as f:
        pickle.dump(link_dicts, f, protocol=-1)


def build_entity_trie(entity_dict, tokenizer: PreTrainedTokenizer):
    entity_token_dict = dict()
    registered_entities = set()

    for i, entity in tqdm(enumerate(entity_dict)):
        # t5 tokenizer encodes such square brackets same
        entity_label = (
            entity_dict[entity]
            .get("label", entity)
            .replace("[", "")
            .replace("]", "")
            .replace("［", "")
            .replace("］", "")
            .replace("﹇", "")
            .lower()
            # marisa-trie cannot handle unicode 0
            .replace("<s>", "s")
            .strip()
        )
        entity_types = entity_dict[entity].get("type", [])

        if len(entity_types) == 0:
            if entity_label in registered_entities:
                entity_repr = "{}{} {}{}".format(
                    UniqueToken.WRAP_OPEN.value,
                    entity_label,
                    entity,
                    UniqueToken.WRAP_CLOSE.value,
                )

            else:
                entity_repr = "{}{}{}".format(
                    UniqueToken.WRAP_OPEN.value,
                    entity_label,
                    UniqueToken.WRAP_CLOSE.value,
                )
                registered_entities.add(entity_label)
        else:
            if len(entity_types) == 1:
                t = list(entity_types)[0]
                entity_types = entity_dict[t].get("label", t) if t in entity_dict else t
            else:
                entity_types = ", ".join(
                    [
                        entity_dict[t].get("label", t) if t in entity_dict else t
                        for t in entity_types
                    ]
                )
            entity_types = (
                entity_types.replace("[", "")
                .replace("]", "")
                .replace("［", "")
                .replace("］", "")
                .replace("﹇", "")
                .strip()
            )
            entity_full = "{} ({})".format(entity_label, entity_types)
            if entity_full in registered_entities:
                entity_repr = "{}{} {}{}".format(
                    UniqueToken.WRAP_OPEN.value,
                    entity_full,
                    entity,
                    UniqueToken.WRAP_CLOSE.value,
                )
            else:
                entity_repr = "{}{}{}".format(
                    UniqueToken.WRAP_OPEN.value,
                    entity_full,
                    UniqueToken.WRAP_CLOSE.value,
                )
                registered_entities.add(entity_full)

        if i < 10:
            print(entity_repr)

        encoded_entity_repr = tokenizer.encode(entity_repr, add_special_tokens=False)
        entity_token_dict[entity] = {
            "encoded": tuple(encoded_entity_repr),
            "decoded": entity_repr.strip(),
        }

    return entity_token_dict, Trie([v["encoded"] for v in entity_token_dict.values()])


def build_relation_trie(relation_dict, tokenizer):
    relation_token_dict = dict()

    for i, relation in tqdm(enumerate(relation_dict)):
        # sanity check
        relation_label = relation_dict.get(relation, relation)
        relation_repr = "{}{}{}".format(
            UniqueToken.WRAP_OPEN.value, relation_label, UniqueToken.WRAP_CLOSE.value
        )

        if i < 10:
            print(relation_repr)

        encoded_relation_repr = tokenizer.encode(
            relation_repr, add_special_tokens=False
        )
        relation_token_dict[relation] = {
            "encoded": tuple(encoded_relation_repr),
            "decoded": relation_repr.strip(),
        }

    return relation_token_dict, Trie(
        [v["encoded"] for v in relation_token_dict.values()]
    )


def build_link_dictionaries(
    kg, entity_token_dict, relation_token_dict, entity_trie, relation_trie
):
    head_rel, rel_tail = kg

    for head, rels in list(head_rel.items()):
        del head_rel[head]
        head_rel[entity_trie.get_id_from_tokens(entity_token_dict[head]["encoded"])] = [
            relation_trie.get_id_from_tokens(relation_token_dict[rel]["encoded"])
            for rel in rels
        ]

    for rel, tails in list(rel_tail.items()):
        del rel_tail[rel]
        rel_tail[
            relation_trie.get_id_from_tokens(relation_token_dict[rel]["encoded"])
        ] = [
            entity_trie.get_id_from_tokens(entity_token_dict[tail]["encoded"])
            for tail in tails
        ]

    return [head_rel, rel_tail]


def change_entity_relation_repr(data, dataset, entity_token_dict, relation_token_dict):
    for i, row in enumerate(data):
        sparql = row["sparql"]
        tokens = []
        for token in sparql.split():
            if dataset.is_entity(token):
                tokens.append(entity_token_dict[token]["decoded"])
            elif dataset.is_relation(token):
                tokens.append(relation_token_dict[token]["decoded"])
            else:
                tokens.append(token)
        data[i]["sparql"] += " "
        data[i]["sparql_proc"] = " ".join(tokens) + " "

        pattern = r"\?(\w+)"

        def replace_var(match):
            return "var_" + match.group(1)

        data[i]["sparql"] = (
            data[i]["sparql"]
            .replace(UniqueToken.ATTR_OPEN.value, "(")
            .replace(UniqueToken.ATTR_CLOSE.value, ")")
            .replace(UniqueToken.COND_OPEN.value, "{")
            .replace(UniqueToken.COND_CLOSE.value, "}")
            .replace(UniqueToken.SEP_DOT.value, ".")
        )
        data[i]["sparql_proc"] = re.sub(pattern, replace_var, data[i]["sparql_proc"])

    return data


if __name__ == "__main__":
    start = time.time()
    args = parse_arguments()
    main(args)
    print(f"Time consumed: {time.time()-start}")
