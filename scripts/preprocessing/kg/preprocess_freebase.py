import time
import pickle
import random
import argparse
import os
import csv
import sys
import gzip
import re
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
sys.path.append(os.path.abspath("."))
from scripts.preprocessing.utils.util import save_multi_pickles
from dotenv import load_dotenv

load_dotenv()


entity_pattern = r'<http://rdf\.freebase\.com/ns/([mg]\.\w+)>'
relation_pattern = r'<http://rdf\.freebase\.com/ns/(.+)>'

# arguments
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--freebase_path",
        type=str,
        default=f"{os.getenv('DATA_PATH')}/original/freebase/parts",
        help="Path to freebase directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{os.getenv('DATA_PATH')}/preprocessed/freebase",
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
        for i, line in enumerate(f):
            assert len(line.split("\t")) == 4, line

            head, relation, tail = line.split("\t")[:3]
            head_match = re.findall(entity_pattern, head)
            relation_match = re.findall(relation_pattern, relation)
            tail_match = re.findall(entity_pattern, tail)

            if len(relation_match) == 0:
                continue

            relation = relation_match[0]
            # Freebase use relation name as itself
            relations[relation] = relation
            
            if len(head_match) != 0:
                head_entity = head_match[0]
                if head_entity not in entities:
                    entities[head_entity]["label"] = head_entity
                if head_entity not in head_rel:
                    head_rel[head_entity] = set()
                head_rel[head_entity].add(relation)

                # multiple name overwrite
                if relation == "type.object.name" and tail.endswith("@en"):
                    entities[head_entity]["label"] = tail.split('"')[1]
                if relation == "common.topic.notable_types":
                    if "type" not in entities[head_entity]:
                        entities[head_entity]["type"] = set()
                    entities[head_entity]["type"].add(tail_match[0])

            if len(tail_match) != 0:
                tail_entity = tail_match[0]
                if tail_entity not in entities:
                    entities[tail_entity]["label"] = tail_entity
                if relation not in rel_tail:
                    rel_tail[relation] = set()
                rel_tail[relation].add(tail_entity)
    
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
    with ProcessPoolExecutor(int(os.cpu_count() * 0.8)) as executor:
        futures = []
        for chunk_path in os.listdir(args.freebase_path):
            future = executor.submit(preprocess, os.path.join(args.freebase_path, chunk_path))
            futures.append(future)

        for future in tqdm(futures):
            curr_entities, curr_relations, curr_head_rel, curr_rel_tail = future.result()

            for entity in curr_entities:
                
                if entity in entities:
                    if "type" in curr_entities[entity]:
                        # only type concatenation
                        if "type" not in entities[entity]:
                            entities[entity]["type"] = set()
                        entities[entity]["type"] |= curr_entities[entity]["type"]
                    if "label" in curr_entities[entity]:
                        # label overwrite
                        label = curr_entities[entity]["label"]
                        if label.startswith("m.") or label.startswith("g."):
                            continue
                        elif entities[entity]["label"].startswith("m.") or entities[entity]["label"].startswith("g."):
                            entities[entity]["label"] = label

                else:
                    entities[entity] = curr_entities[entity]

            for relation in curr_relations:
                if relation not in relations:
                    relations[relation] = relation

            for head in curr_head_rel:
                if head not in head_rel:
                    head_rel[head] = curr_head_rel[head]
                else:
                    head_rel[head] |= curr_head_rel[head]

            for rel in curr_rel_tail:
                if rel not in rel_tail:
                    rel_tail[rel] = curr_rel_tail[rel]
                else:
                    rel_tail[rel] |= curr_rel_tail[rel]

    
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
