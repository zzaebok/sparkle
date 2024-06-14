import os
import json
import re
from SPARQLWrapper import SPARQLWrapper, JSON, GET
from .dataset import Dataset
from ..utils import UniqueToken

entity_pattern = r"<http://www.wikidata.org/entity/(Q\d+)>"
relation_pattern = r"<http://www.wikidata.org/prop/direct/(P\d+)>"


class SimpleQuestions(Dataset):
    datastore = None

    def __init__(self, dataset_dir, head_rel, rel_tail):
        # for simplequestions, we filter sparqls that contain triple patterns that doesn't align with wikidata dump used.
        self.head_rel = head_rel
        self.rel_tail = rel_tail
        self.preprocess(dataset_dir)

    def preprocess(self, dataset_dir):
        with open(
            os.path.join(dataset_dir, "annotated_wd_data_train_answerable.json"), "r"
        ) as f:
            sq_train = json.load(f)
        with open(
            os.path.join(dataset_dir, "annotated_wd_data_valid_answerable.json"), "r"
        ) as f:
            sq_eval = json.load(f)
        with open(
            os.path.join(dataset_dir, "annotated_wd_data_test_answerable.json"), "r"
        ) as f:
            sq_test = json.load(f)

        self.train_dataset = self.cleanse_questions(sq_train["questions"])
        self.eval_dataset = self.cleanse_questions(sq_eval["questions"])
        self.test_dataset = self.cleanse_questions(sq_test["questions"])

    def cleanse_questions(self, questions):
        """
        Args:
            questions (List[Dict]): simplequestions question data
            Example:
                [
                    {
                        'id': 2,
                        'knowledgebase': 'wikidata',
                        'question': [
                            {
                                'string': 'what is the language in which mera shikar was filmed in\n',
                                'language': 'en'
                            }
                        ],
                        'query': {
                            'answers': [
                                {
                                    'head': {
                                        'vars': ['o1']
                                    },
                                    'results': {
                                        'bindings': [
                                            {
                                                'o1': {
                                                    'type': 'uri',
                                                    'value': 'http://www.wikidata.org/entity/Q1568'
                                                }
                                            }
                                        ]
                                    }
                                }
                            ],
                            'sparql': 'SELECT ?o1 WHERE { <http://www.wikidata.org/entity/Q6817891> <http://www.wikidata.org/prop/direct/P364> ?o1 }'}
                        }
                ]
        """
        data = []
        for question in questions:
            raw_question = question["question"][0]["string"].replace("\n", "")
            answers = []
            for binding in question["query"]["answers"][0]["results"]["bindings"]:
                for value in binding.values():
                    answers.append(value["value"])
            sparql = question["query"]["sparql"]

            # CLENASE
            sparql = sparql.replace("{", " { ").replace("}", " } ")
            sparql = re.sub(r"\s+", " ", sparql).strip()

            # REPLACE
            replace_token_dict = {
                "?s1": "?x",
                "?o1": "?x",
                "{": UniqueToken.COND_OPEN.value,
                "}": UniqueToken.COND_CLOSE.value,
            }
            for replace_key, replace_value in replace_token_dict.items():
                sparql = sparql.replace(replace_key, replace_value)

            row = {"text": raw_question, "sparql": sparql, "answers": answers}

            entity = re.findall(entity_pattern, sparql)[0]
            relation = re.findall(relation_pattern, sparql)[0]

            try:
                # rel_tail
                if sparql.startswith("SELECT ?x WHERE { ?x"):
                    if entity not in self.rel_tail[relation]:
                        continue
                # head_rel
                else:
                    if relation not in self.head_rel[entity]:
                        continue
            except KeyError:
                continue

            data.append(row)
        return data

    @staticmethod
    def is_entity(token):
        if "wikidata.org/entity" in token:
            return True
        return False

    @staticmethod
    def is_relation(token):
        if "wikidata.org/prop/direct" in token:
            return True
        return False

    @staticmethod
    def change_iri(full_iri):
        if full_iri.startswith("P"):
            return f"<http://www.wikidata.org/prop/direct/{full_iri}>"
        else:
            # "Q"
            return f"<http://www.wikidata.org/entity/{full_iri}>"

    def change_format(self, entities, relations, kg):
        """
        Two functionalities
        1. Change IRI from original wikidata format to simplequestions format
        2. Label format (from original to custom)
        """

        for entity in list(entities.keys()):
            entity_info = entities.pop(entity)
            if "type" in entity_info:
                entity_info["type"] = [self.change_iri(t) for t in entity_info["type"]]
            entities[self.change_iri(entity)] = entity_info

        for relation in list(relations.keys()):
            relations[self.change_iri(relation)] = relations.pop(relation)

        for kg_dict in kg:
            for left, rights in list(kg_dict.items()):
                kg_dict[self.change_iri(left)] = [self.change_iri(r) for r in rights]
                del kg_dict[left]

        return entities, relations, kg

    @classmethod
    def postprocess(cls, token_entity_dict, token_relation_dict, draft_sparqls):
        postprocessed_sparqls = []
        for sparql in draft_sparqls:
            sparql = sparql.split("brack_close")[0] + "brack_close "
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
                UniqueToken.COND_OPEN.value: "{",
                UniqueToken.COND_CLOSE.value: "} ",
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

        rtn = []
        for result in response["results"]["bindings"]:
            for var in result:
                if result[var]["type"] == "uri":
                    rtn.append(result[var]["value"])

        return rtn
