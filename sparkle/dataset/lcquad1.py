import os
import json
import re
import random
from SPARQLWrapper import SPARQLWrapper, JSON, GET
from .dataset import Dataset
from ..utils import UniqueToken

class LCQuAD1(Dataset):
    datastore = None
    
    def __init__(self, dataset_dir):
        self.preprocess(dataset_dir)

    def preprocess(self, dataset_dir):
        # Answer attached files
        with open(os.path.join(dataset_dir, "train-data-answer.json"), "r") as f:
            lcquad1_train = json.load(f)
        with open(os.path.join(dataset_dir, "test-data-answer.json"), "r") as f:
            lcquad1_test = json.load(f)

        train_dataset = self.cleanse_questions(lcquad1_train)
        random.shuffle(train_dataset)
        SPLIT_INDEX = int(0.95 * len(train_dataset))
        eval_dataset = train_dataset[SPLIT_INDEX:]
        train_dataset = train_dataset[:SPLIT_INDEX]
        test_dataset = self.cleanse_questions(lcquad1_test)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset

    def cleanse_questions(self, questions):
        """
        Args:
            questions (List[Dict]): lcquad1 question data
            Example:
                [
                    {
                        '_id': '1501',
                        'corrected_question': 'How many movies did Stanley Kubrick direct?',
                        'intermediary_question': 'How many <movies> are there whose <director> is <Stanley Kubrick>?',
                        'sparql_query': 'SELECT DISTINCT COUNT(?uri) WHERE {?uri <http://dbpedia.org/ontology/director> <http://dbpedia.org/resource/Stanley_Kubrick>  . }',
                        'sparql_template_id': 101
                    }
                    ...
                ]
        """
        data = []
        for question in questions:
            raw_question = question["corrected_question"]

            sparql = question["sparql_query"]

            # CLENASE
            sparql = sparql.replace(". }", "}").replace("{", " { ").replace("}", " } ")
            sparql = re.sub(r"\s+", " ", sparql).strip()

            # REPLACE
            replace_token_dict = {
                "?uri.": "?uri .",
                "COUNT(?uri)": "COUNT attr_open ?uri attr_close",
                "{": "brack_open",
                "}": "brack_close",
            }
            for replace_key, replace_value in replace_token_dict.items():
                sparql = sparql.replace(replace_key, replace_value)

            sparql = " ".join(["sep_dot" if token == "." else token for token in sparql.split()])

            row = {"text": raw_question, "sparql": sparql, "answers": question["answers"]}

            data.append(row)
        return data

    @staticmethod
    def is_entity(token):
        if token.startswith("<"):
            if "dbpedia.org/resource" in token:
                return True
            elif "dbpedia.org/ontology" in token and token.split("ontology/")[1][0].isupper():
                return True
        return False

    @staticmethod
    def is_relation(token):
        if token.startswith("<"):
            if "dbpedia.org/resource" in token:
                return False
            elif "dbpedia.org/ontology" in token and token.split("ontology/")[1][0].isupper():
                return False
            return True

        return False

    # LCQuAD1.0 has the same format as DBPedia, so change the label as human readable way
    def change_format(self, entities, relations, kg):
        def make_label(full_iri):
            if "dbpedia.org/resource" in full_iri:
                return f"{full_iri.split('/')[-1][:-1]} : resource"
            elif "dbpedia.org/ontology" in full_iri:
                return f"{full_iri.split('/')[-1][:-1]} : ontology"
            elif "dbpedia.org/property" in full_iri:
                return f"{full_iri.split('/')[-1][:-1]} : property"
            elif full_iri == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                return "type : rdf"
            return full_iri

        for entity in list(entities.keys()):
            entity_info = entities.pop(entity)
            if "label" in entity_info:
                entity_info["label"] = make_label(entity_info["label"])

            entities[entity] = entity_info

        for relation in list(relations.keys()):
            relations[relation] = make_label(relations[relation])

        return entities, relations, kg

    @classmethod
    def postprocess(cls, token_entity_dict, token_relation_dict, draft_sparqls):
        postprocessed_sparqls = []
        for sparql in draft_sparqls:
            # entity, relation restore
            placeholder_pattern = r"\[[^\]]+\]"
            matches = re.findall(placeholder_pattern, sparql)

            invalid_query_generated = False
            for match in matches:
                if match in token_entity_dict:
                    sparql = sparql.replace(match, token_entity_dict[match])
                elif match in token_relation_dict:
                    sparql = sparql.replace(match, token_relation_dict[match])
                else:
                    invalid_query_generated = True
                    break
            if invalid_query_generated:
                postprocessed_sparqls.append("")
                continue

            # var_name
            sparql = sparql.replace("var_", "?")

            # tokens
            replace_token_dict = {
                UniqueToken.ATTR_OPEN.value: "(",
                UniqueToken.ATTR_CLOSE.value: ")",
                UniqueToken.COND_OPEN.value: "{",
                UniqueToken.COND_CLOSE.value: "} ",
                "sep_dot": ".",
            }
            for replace_key, replace_value in replace_token_dict.items():
                sparql = sparql.replace(replace_key, replace_value)

            postprocessed_sparqls.append(sparql.strip())

        return postprocessed_sparqls

    @classmethod
    def query(cls, sparql_query: str, endpoint: str):
        if cls.datastore is None:
            cls.datastore = SPARQLWrapper(endpoint)
            cls.datastore.setTimeout(10)
            cls.datastore.setReturnFormat(JSON)
            cls.datastore.setMethod(GET)
        
        sparql_query = sparql_query.strip()
        cls.datastore.setQuery(sparql_query)
        response = cls.datastore.queryAndConvert()

        if sparql_query.startswith("ASK WHERE"):
            return [response["boolean"]]
        
        elif sparql_query.startswith("SELECT DISTINCT COUNT"):
            return [int(response['results']['bindings'][0]['callret-0']['value'])]
        
        else:
            rtn = []
            for result in response["results"]["bindings"]:
                for var in result:
                    if result[var]["type"] == var:
                        rtn.append(
                            result[var]["value"]
                        )

            return rtn