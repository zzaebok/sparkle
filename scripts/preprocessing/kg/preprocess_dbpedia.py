import time
import pickle
import bz2
import argparse
import os
from tqdm import tqdm
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()


# arguments
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dbpedia_path",
        type=str,
        default=f"{os.getenv('DATA_PATH')}/original/dbpedia/merge.ttl.bz2",
        help="Path to the dbpedia dump file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{os.getenv('DATA_PATH')}/preprocessed/dbpedia",
        help="Directory where output files are saved",
    )

    args = parser.parse_args()
    return args


def is_valid_component(full_iri):
    if "dbpedia.org/resource" in full_iri:
        return True
    elif "dbpedia.org/ontology" in full_iri:
        return True
    elif "dbpedia.org/property" in full_iri:
        return True

    return False


def main(args):
    # entity information: name and type
    entities = defaultdict(dict)
    # all relations
    relations = dict()
    # triple information
    head_rel = dict()
    rel_tail = dict()

    with bz2.open(args.dbpedia_path, "rt") as f:
        for _, line in enumerate(tqdm(f)):
            if line.startswith("#"):
                continue
            line = line.replace(" .", "")
            triple = line.strip().split()

            head_iri = triple[0]
            relation_iri = triple[1]
            tail_iri = " ".join(triple[2:])

            # dbpedia entity label equals to iri
            if head_iri not in entities:
                entities[head_iri] = {"label": head_iri}

            # set of relations
            if relation_iri not in relations:
                relations[relation_iri] = relation_iri

            if head_iri not in head_rel:
                head_rel[head_iri] = set()

            head_rel[head_iri].add(relation_iri)

            # (head, tail) both have dbpedia entity - interest of sparql generation
            if is_valid_component(tail_iri):
                if tail_iri not in entities:
                    entities[tail_iri] = {"label": tail_iri}

                # neighbor information
                if relation_iri not in rel_tail:
                    rel_tail[relation_iri] = set()
                rel_tail[relation_iri].add(tail_iri)

    for head in list(head_rel.keys()):
        head_rel[head] = list(head_rel[head])

    for rel in rel_tail:
        rel_tail[rel] = list(rel_tail[rel])

    with open(os.path.join(args.output_dir, "entities.pkl"), "wb") as f:
        pickle.dump(entities, f)

    with open(os.path.join(args.output_dir, "relations.pkl"), "wb") as f:
        pickle.dump(relations, f)

    with open(os.path.join(args.output_dir, "kg.pkl"), "wb") as f:
        pickle.dump([head_rel, rel_tail], f)


if __name__ == "__main__":
    start = time.time()
    args = parse_arguments()
    main(args)
    print(f"Time consumed: {time.time()-start}")
