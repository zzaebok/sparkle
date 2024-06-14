import time
import pickle
import random
import argparse
import os
import csv
import sys
import gzip
import re
import json
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
sys.path.append(os.path.abspath("."))
from scripts.preprocessing.utils.util import save_multi_pickles
from dotenv import load_dotenv

load_dotenv()

# arguments
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wikidata_path",
        type=str,
        default=f"{os.getenv('DATA_PATH')}/original/wikidata/parts",
        help="Path to wikidata directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{os.getenv('DATA_PATH')}/preprocessed/wikidata",
        help="Directory where output files are saved",
    )

    args = parser.parse_args()
    return args

def preprocess(chunk_path):
    # entity information: name and type
    entities = defaultdict(dict)
    # all relations
    relations = dict()
    # triple information
    head_rel = dict()
    rel_tail = dict()

    with gzip.open(chunk_path, "rt") as f:
        chunks = json.load(f)


    for i, component in enumerate(chunks):
        assert component["type"] == "item" or component["type"] == "property"

        # relation
        if component["type"] == "property":
            relation_id = component["id"]
            relation_name = component["labels"]["en"]["value"] if len(component["labels"]) != 0 else relation_id
            relations[relation_id] = relation_name

        # entity
        else:
            head_id = component["id"]
            # label
            head_label = component["labels"]["en"]["value"] if len(component["labels"]) != 0 else head_id
            if head_id not in entities:
                entities[head_id] = {"label": head_label}
            else:
                if entities[head_id]["label"] == head_id:
                    entities[head_id]["label"] = head_label
            
            # connectivity
            for relation_id, claims in component["claims"].items():
                if head_id not in head_rel:
                    head_rel[head_id] = set()
                head_rel[head_id].add(relation_id)

                for claim in claims:
                    if 'datavalue' in claim['mainsnak']:
                        if claim['mainsnak']['datavalue'].get('type') == "wikibase-entityid":
                            tail_id = claim['mainsnak']['datavalue']['value']['id']
                            if tail_id not in entities:
                                entities[tail_id] = {"label": tail_id}
                            
                            if relation_id not in rel_tail:
                                rel_tail[relation_id] = set()

                            rel_tail[relation_id].add(tail_id)

                            if relation_id == "P31":
                                if "type" not in entities[head_id]:
                                    entities[head_id]["type"] = set()
                                entities[head_id]["type"].add(tail_id)

    
    entities = dict(entities)

    return (entities, relations, head_rel, rel_tail)

def main(args):

    random.seed(2023)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # entity information: name and type
    entities = defaultdict(dict)
    # all relations
    relations = dict()
    # triple information
    head_rel = dict()
    rel_tail = dict()
    
    # multi-processing
    with ProcessPoolExecutor(16) as executor:
        futures = []
        for chunk_path in os.listdir(args.wikidata_path):
            if not chunk_path.endswith(".json.gz"):
                continue
            future = executor.submit(preprocess, os.path.join(args.wikidata_path, chunk_path))
            futures.append(future)

        for future in tqdm(futures):

            curr_entities, curr_relations, curr_head_rel, curr_rel_tail = future.result()

            for entity, entity_info in curr_entities.items():
                if entity in entities and entities[entity]["label"] != entity:
                    continue
                entities[entity] = entity_info

            for relation, relation_label in curr_relations.items():
                relations[relation] = relation_label

            for head in curr_head_rel:
                head_rel[head] = curr_head_rel[head]

            for rel in curr_rel_tail:
                if rel in rel_tail:
                    rel_tail[rel] |= curr_rel_tail[rel]
                else:
                    rel_tail[rel] = curr_rel_tail[rel]

    
    # choose max 2 types
    for entity in list(entities.keys()):
        if "type" in entities[entity] and len(entities[entity]["type"]) > 2:
            entities[entity]["type"] = list(
                set(random.sample(entities[entity]["type"], 2))
            )

    for head in list(head_rel.keys()):
        head_rel[head] = list(head_rel[head])

    for rel in rel_tail:
        rel_tail[rel] = list(rel_tail[rel])

    files = [("entities.pkl", dict(entities)), ("relations.pkl", relations), ("kg.pkl", [head_rel, rel_tail])]

    save_multi_pickles(args.output_dir, files)


if __name__ == "__main__":
    start = time.time()
    args = parse_arguments()
    main(args)
    print(f"Time consumed: {time.time()-start}")
