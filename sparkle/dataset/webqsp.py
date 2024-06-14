import os
import json
import re
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON, GET
from .dataset import Dataset
from ..utils import UniqueToken


class WebQSP(Dataset):
    datastore = None

    def __init__(self, dataset_dir):
        self.preprocess(dataset_dir)

    def preprocess(self, dataset_dir):
        with open(os.path.join(dataset_dir, "WebQSP.ptrain.json"), "r") as f:
            webqsp_train = json.load(f)
        with open(os.path.join(dataset_dir, "WebQSP.pdev.json"), "r") as f:
            webqsp_eval = json.load(f)
        with open(os.path.join(dataset_dir, "WebQSP.test.json"), "r") as f:
            webqsp_test = json.load(f)

        self.train_dataset = self.cleanse_questions(webqsp_train["Questions"])
        self.eval_dataset = self.cleanse_questions(webqsp_eval["Questions"])
        self.test_dataset = self.cleanse_questions(webqsp_test["Questions"])

    def cleanse_questions(self, questions):
        """
        Args:
            questions (List[Dict]): WebQSP question data
            Example:
                [
                    {
                        "QuestionId": "WebQTest-0",
                        "RawQuestion": "what does jamaican people speak?",
                        "ProcessedQuestion": "what does jamaican people speak",
                        "Parses": [
                            {
                                "ParseId": "WebQTest-0.P0",
                                "Sparql": "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.03_r3)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.03_r3 ns:location.country.languages_spoken ?x .\n}\n",
                                ...
                            },
                        ]
                    },
                    ...
                ]
        """
        data = []
        for question in tqdm(questions):
            raw_question = question["RawQuestion"]

            # if multiple sparqls, shortest one selected
            sparql = question["Parses"][0]["Sparql"]
            constraints = question["Parses"][0]["Constraints"]
            list_answers = [[a["AnswerArgument"] for a in question["Parses"][0]["Answers"]]]

            for parse in question["Parses"][1:]:
                if not (parse["AnnotatorComment"]["QuestionQuality"] == "Good" and parse["AnnotatorComment"]["ParseQuality"] == "Complete"):
                    continue
                
                list_answers.append([a["AnswerArgument"] for a in parse["Answers"]])
                
                if len(parse["Sparql"]) < len(sparql):
                    sparql = parse["Sparql"]
                    constraints = parse["Constraints"]

            # remove manual mark
            if sparql.startswith("#MANUAL SPARQL"):
                continue

            # duplicate time conditions in a specific query
            if "?sk8" in sparql:
                continue

            # CLEANSE
            sparql = (
                sparql.replace("PREFIX ns: <http://rdf.freebase.com/ns/>\n", "")
                .replace("\n", " ")
                .replace("(", " ( ")
                .replace(")", " ) ")
                .replace("{", " { ")
                .replace("}", " } ")
                .replace("OFFSET 1", "")
            )

            sparql = re.sub(r"\s+", " ", sparql).strip()

            # string match
            string_pattern = r"FILTER \( str \( \?sk0 \) = [ a-zA-Z0-9\"]+ \)"
            matches = re.search(string_pattern, sparql)
            if matches is not None:
                target_string = (
                    matches.group(0).split("FILTER ( str ( ?sk0 ) = ")[1].replace(" )", "")
                )
                sparql = re.sub(string_pattern, f"FILTER {target_string}", sparql)

            # year filter
            year_pattern = r"FILTER \( NOT EXISTS { \?[xy] ns:[a-zA-Z0-9_.]+ \?sk[01] } \|\| EXISTS { \?[xy] ns:[a-zA-Z0-9_.]+ \?sk[12] . FILTER \( xsd:datetime \( \?sk[12] \) [><=]+ \"[0-9\-]+\"\^\^xsd:dateTime \) } \) FILTER \( NOT EXISTS { \?[xy] ns:[a-zA-Z0-9_.]+ \?sk[23] } \|\| EXISTS { \?[xy] ns:[a-zA-Z0-9_.]+ \?sk[34] . FILTER \( xsd:datetime \( \?sk[34] \) [><=]+ \"[0-9\-]+\"\^\^xsd:dateTime \) } \)"
            matches = re.search(year_pattern, sparql)
            if matches is not None:
                greater_date, greater_relation = None, None
                less_relation = None, None
                for constraint in constraints:
                    if constraint["Operator"] == "GreaterOrEqual":
                        if constraint["Argument"] == "2015-08-10":
                            greater_date = "now"
                        else:
                            greater_date = constraint["Argument"]
                        greater_relation = constraint["NodePredicate"]
                    if constraint["Operator"] == "LessOrEqual":
                        less_relation = constraint["NodePredicate"]
                if "?x" in matches.group(0):
                    var_name = "?x"
                else:
                    var_name = "?y"
                sparql = re.sub(
                    year_pattern,
                    f"FILTER {greater_date[:4]} {var_name} ns:{greater_relation} ns:{less_relation}",
                    sparql,
                )

            # order by filter
            if "ORDER BY" in sparql:
                is_datetime = "datetime" in sparql.split("ORDER BY")[1]
                if "ORDER BY DESC" in sparql:
                    sparql = sparql.split("ORDER BY")[0].strip()
                    sparql += "ORDER BY " + "last" if is_datetime else "large"
                else:
                    sparql = sparql.split("ORDER BY")[0].strip()
                    sparql += "ORDER BY " + "first" if is_datetime else "small"

            duplicate_filter = r" FILTER \( \?x != ns:m\.[a-zA-Z0-9_]+ \) FILTER \( !isLiteral \( \?x \) OR lang \( \?x \) = '' OR langMatches \( lang \( \?x \) , 'en' \) \)"
            sparql = re.sub(duplicate_filter, "", sparql)

            # REPLACE
            replace_token_dict = {
                "(": UniqueToken.ATTR_OPEN.value,
                ")": UniqueToken.ATTR_CLOSE.value,
                "{": UniqueToken.COND_OPEN.value,
                "}": UniqueToken.COND_CLOSE.value,
            }
            for replace_key, replace_value in replace_token_dict.items():
                sparql = sparql.replace(replace_key, replace_value)

            sparql = " ".join(
                ["sep_dot" if token == "." else token for token in sparql.split()]
            ).replace("sep_dot brack_close", "brack_close")

            row = {
                "text": raw_question,
                "sparql": sparql,
                "answers": list_answers,
            }

            data.append(row)
        return data

    @staticmethod
    def is_entity(token):
        return token.startswith("ns:m.") or token.startswith("ns:g.")

    @staticmethod
    def is_relation(token):
        return token.startswith("ns:") and not (
            token.startswith("ns:m.") or token.startswith("ns:g.")
        )
    
    @staticmethod
    def change_iri(full_iri):
        return f"ns:{full_iri}"

    def change_format(self, entities, relations, kg):
        '''
        Two functionalities
        1. Change IRI from original freebase format to webqsp format (m.1234 to ns:m.1234)
        2. Label format (relation abc.def to def abc)
        '''

        for entity in list(entities.keys()):
            entity_info = entities.pop(entity)
            if "type" in entity_info:
                entity_info["type"] = [self.change_iri(t) for t in entity_info["type"]]

            entities[self.change_iri(entity)] = entity_info

        for relation in list(relations.keys()):
            relation_label = relations.pop(relation)
            # ns:film.performance.character -> character performance film
            relations[self.change_iri(relation)] = " ".join(
                [c for c in reversed(relation_label.split("."))]
            )

        # kg_dict: head_rel and rel_tail
        for kg_dict in kg:
            for left, rights in list(kg_dict.items()):
                kg_dict[self.change_iri(left)] = [self.change_iri(r) for r in rights]
                del kg_dict[left]

        return entities, relations, kg

    @classmethod
    def postprocess(cls, token_entity_dict, token_relation_dict, sparqls):
        postprocessed_sparqls = []
        for sparql in sparqls:
            # remove invalid tokens after brack_close
            if UniqueToken.COND_CLOSE.value in sparql:
                after_brack_close = sparql[sparql.index(UniqueToken.COND_CLOSE.value) :]
                after_brack_close_pattern = r"brack_close ORDER BY [a-z]+ "
                matches = re.search(after_brack_close_pattern, after_brack_close)
                if matches is None and not sparql.endswith(UniqueToken.COND_CLOSE.value):
                    sparql = (
                        sparql[: sparql.index(UniqueToken.COND_CLOSE.value)]
                        + UniqueToken.COND_CLOSE.value
                    )

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

            # string matching
            string_pattern = r"FILTER \"[0-9a-zA-Z ]+\""
            string_matches = re.findall(string_pattern, sparql)
            for match in string_matches:
                sparql = sparql.replace(match, f"FILTER (str(?sk0) = {' '.join(match.split()[1:])})")

            # year pattern
            if len(string_matches) != 0:
                sk_idx = 1
            else:
                sk_idx = 0
            year_pattern = r"FILTER [a-z0-9]+ \?[xy] ns:[a-zA-Z0-9_.]+ ns:[a-zA-Z0-9_.]+"
            matches = re.findall(year_pattern, sparql)
            for match in matches:
                _, year, var_name, greater, less = match.split()
                if year != "now":
                    year_start = year + "-01-01"
                    year_end = year + "-12-31"
                else:
                    year_start = "2015-08-10"
                    year_end = "2015-08-10"
                filter_year_end = f'FILTER (NOT EXISTS {{ {var_name} {less} ?sk{sk_idx} }} || EXISTS {{ {var_name} {less} ?sk{sk_idx+1} . FILTER(xsd:datetime(?sk{sk_idx+1}) <= "{year_end}"^^xsd:dateTime) }} ) '
                filter_year_start = f'FILTER (NOT EXISTS {{ {var_name} {greater} ?sk{sk_idx+2} }} || EXISTS {{ {var_name} {greater} ?sk{sk_idx+3} . FILTER(xsd:datetime(?sk{sk_idx+3}) >= "{year_start}"^^xsd:dateTime) }} )'

                sparql = sparql.replace(match, filter_year_end + filter_year_start)

            # order by
            if "ORDER BY" in sparql:
                if "?sk" not in sparql:
                    var_name = "?x"
                elif "?sk1" in sparql:
                    var_name = "?sk4"
                else:
                    var_name = "?sk0"

                is_datetime = ("first" in sparql.split("ORDER BY")[1]) or (
                    "last" in sparql.split("ORDER BY")[1]
                )

                if "ORDER BY last" in sparql or "ORDER BY large" in sparql:
                    sparql = sparql.replace(
                        f"ORDER BY DESC {'datetime' if is_datetime else 'float'}",
                        f"ORDER BY DESC(xsd:{'datetime' if is_datetime else 'float'}({var_name})) LIMIT 1",
                    )
                else:
                    sparql = sparql.replace(
                        f"ORDER BY {'datetime' if is_datetime else 'float'}",
                        f"ORDER BY xsd:{'datetime' if is_datetime else 'float'}({var_name}) LIMIT 1",
                    )

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

            # original FILTERs
            sparql = sparql.replace(
                "WHERE {",
                "WHERE { FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en')) ",
            )

            for token in sparql.split():
                if token.startswith("ns:m.") or token.startswith("ns:g."):
                    sparql = sparql.replace(
                        "WHERE {",
                        f"WHERE {{ FILTER (?x != {token}) ",
                    )

            postprocessed_sparqls.append(sparql.strip())

        return postprocessed_sparqls

    @classmethod
    def query(cls, sparql_query: str, endpoint: str):
        if cls.datastore is None:
            cls.datastore = SPARQLWrapper(endpoint)
            cls.datastore.setTimeout(10)
            cls.datastore.setReturnFormat(JSON)
            cls.datastore.setMethod(GET)
        
        SPARQL_PREFIXES = "PREFIX ns: <http://rdf.freebase.com/ns/> "
        sparql_query = SPARQL_PREFIXES + sparql_query
        cls.datastore.setQuery(sparql_query)

        response = cls.datastore.queryAndConvert()

        rtn = []
        for result in response["results"]["bindings"]:
            assert len(result) == 1  # only select one variable
            for var in result:
                rtn.append(
                    result[var]["value"]
                    .replace("http://rdf.freebase.com/ns/", "")
                    .replace("-08:00", "")
                    .replace("T05:12:00", "")
                )

        return rtn