from abc import ABC


class Dataset(ABC):
    @staticmethod
    def is_entity(token):
        raise NotImplementedError

    @staticmethod
    def is_relation(token):
        raise NotImplementedError

    # cleanse raw dataset, split train/eval/test
    def preprocess(self):
        raise NotImplementedError

    def change_format(self, entities, relations, kg):
        raise NotImplementedError

    def postprocess(self, entity_token_dict, relation_token_dict, question):
        raise NotImplementedError
