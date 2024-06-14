from typing import List, Dict
import torch
from enum import Enum
from tqdm import tqdm
from .trie import Trie
from sparkle.utils import UniqueToken, encode_fn, verbose_generator

class Status(Enum):
    FORM = 0
    VARS = 1
    HEAD_ENTITY = 2
    RELATION = 3
    TAIL_ENTITY = 4
    CONTINUE_OR_NOT = 5
    NORMAL = 6


class SparqlTokenGenerator:
    def __init__(
        self,
        tokenizer,
        entity_trie: Trie,
        relation_trie: Trie,
        head_rel: Dict,
        relation_tail: Dict,
        sparql_vocab: Dict,
        config,
    ):
        self.tokenizer = tokenizer
        self.form_start_index = 1 if config.forced_bos_token_id is None else 2

        self.wrap_token_ids = [
            encode_fn(UniqueToken.WRAP_OPEN.value, tokenizer),
            encode_fn(UniqueToken.WRAP_CLOSE.value, tokenizer),
        ]
        self.entity_trie = entity_trie
        self.relation_trie = relation_trie

        self.head_rel_trie = Trie([self.entity_trie.get_tokens_from_id(head_id) for head_id in head_rel.keys()])
        self.rel_tail_trie = Trie(
            [self.relation_trie.get_tokens_from_id(rel_id) for rel_id in relation_tail.keys()]
        )
        self.head_rel = head_rel
        self.relation_tail = relation_tail

        self.head_rel_cache = dict()
        self.relation_tail_cache = dict()

        # pre-build rel_tail_cache to speed up inference.
        for relation in tqdm(self.rel_tail_trie):
            relation = tuple(relation)
            try:
                if relation not in self.relation_tail_cache:
                    self.relation_tail_cache[relation] = Trie(
                        [
                            self.entity_trie.get_tokens_from_id(x)
                            for x in self.relation_tail[self.relation_trie.get_id_from_tokens(relation)]
                        ]
                    )
            except KeyError:
                continue

        assert all(
            [
                True if key in sparql_vocab else False
                for key in [
                    "SELECT",
                    "ASK",
                    "VARS",
                    "WHERE",
                    "COND_OPEN",
                    "COND_CLOSE",
                    "FILTER",
                ]
            ]
        ), "Pre-defined keys are not in the `sparql_vocab`"
        self.sparql_vocab = sparql_vocab
        
        self.normal_tokens = list(range(len(tokenizer)))
        self.normal_tokens.remove(self.sparql_vocab["ORDER_BY"])
        self.select_var_tokens = self.sparql_vocab["VARS"] + \
            [self.sparql_vocab["WHERE"]] + \
            [
                encode_fn(token, self.tokenizer)
                for token in [
                    UniqueToken.COUNT.value,
                    UniqueToken.ATTR_OPEN.value,
                    UniqueToken.ATTR_CLOSE.value,
                ]
            ]
        self.select_var_tokens = set(self.select_var_tokens)
        
    def get_token_indices(self, generated_tokens: List[int]):
        token_index = {
            "COND_OPEN": [],
            "SEP_DOT": [],
            "COND_CLOSE": [],
            "FILTER": [],
            "WRAP_OPEN": [],
            "WRAP_CLOSE": [],
        }
        for i, token in enumerate(generated_tokens):
            if token == self.sparql_vocab["COND_OPEN"]:
                token_index["COND_OPEN"].append(i)
            elif token == self.sparql_vocab["SEP_DOT"]:
                token_index["SEP_DOT"].append(i)
            elif token == self.sparql_vocab["COND_CLOSE"]:
                token_index["COND_CLOSE"].append(i)
            elif token == self.sparql_vocab["FILTER"]:
                token_index["FILTER"].append(i)
            elif token == self.sparql_vocab["WRAP_OPEN"]:
                token_index["WRAP_OPEN"].append(i)
            elif token == self.sparql_vocab["WRAP_CLOSE"]:
                token_index["WRAP_CLOSE"].append(i)
        
        return token_index

    def get_status(self, generated_tokens: List[int], token_index: Dict):
        # Just started, only decoder_start_token_id, forced_bos_token_id
        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L693
        # https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/generation/logits_process.py#L882
        
        if len(generated_tokens) == self.form_start_index:
            return Status.FORM

        # Query form generated, but not opend condition clause yet
        if self.sparql_vocab["WHERE"] not in generated_tokens:
            return Status.VARS

        cond_open_index = token_index["COND_OPEN"][-1] if len(token_index["COND_OPEN"]) != 0 else -10000
        sep_dot_index = token_index["SEP_DOT"][-1] if len(token_index["SEP_DOT"]) != 0 else -10000
        cond_close_index = token_index["COND_CLOSE"][-1] if len(token_index["COND_CLOSE"]) != 0 else -10000
        filter_index = token_index["FILTER"][-1] if len(token_index["FILTER"]) != 0 else -10000

        if max(cond_close_index, filter_index) > max(cond_open_index, sep_dot_index):
            return Status.NORMAL

        last_triple_start_index = max(cond_open_index, sep_dot_index)
        num_wrap_tokens = 0
        for token in generated_tokens[last_triple_start_index:]:
            if token in self.wrap_token_ids:
                num_wrap_tokens += 1
            if token in self.sparql_vocab["VARS"]:
                num_wrap_tokens += 2

        # After one triple created -> decide to continue or not
        if num_wrap_tokens % 6 == 0 and (
            generated_tokens[-1] == self.wrap_token_ids[1]
            or generated_tokens[-1] in self.sparql_vocab["VARS"]
        ):
            return Status.CONTINUE_OR_NOT

        # After condition open or sep dot, triple generation
        if num_wrap_tokens % 6 in [0, 1]:
            return Status.HEAD_ENTITY
        elif num_wrap_tokens % 6 in [2, 3]:
            return Status.RELATION
        else:
            return Status.TAIL_ENTITY

    @verbose_generator
    def generate(self, batch_id: int, input_ids: torch.Tensor, verbose=False):
        """
        A beam-search process function which acts as a `prefix_allowed_tokens_fn` in Huggingface transformers.
        For more details, refer to https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.generate
        Args:
            batch_id (int): batch id
            input_ids (torch.Tensor): Tokenized sentence that is currently generated by auto regressive way at each step.
        Returns:
            List[int]: Possible tokens the model can generate at this moment.
        Examples:
            ...
            >>> print(trie_search_processor.process(torch.Tensor([1,2,3])))
            [1, 4, 5]
        """
        # print(input_ids)
        input_ids = input_ids.tolist()

        # attach where and {
        if input_ids[-1] == self.sparql_vocab["WHERE"]:
            return [self.sparql_vocab["COND_OPEN"]]
        if input_ids[-1] == self.sparql_vocab["COND_CLOSE"]:
            return [self.sparql_vocab["ORDER_BY"], self.tokenizer.eos_token_id]
        
        token_index = self.get_token_indices(input_ids)

        status = self.get_status(input_ids, token_index)
        candidate_tokens = []

        if status == Status.FORM:
            candidate_tokens = [self.sparql_vocab["SELECT"], self.sparql_vocab["ASK"]]

        elif status == Status.VARS:
            # ASK
            if self.sparql_vocab["ASK"] in input_ids:
                return [self.sparql_vocab["WHERE"]]

            # SELECT
            candidate_tokens = list(self.select_var_tokens - set(input_ids[2:]))

        elif status == Status.HEAD_ENTITY:
            entity_start_pos = -10000

            # last position for sep_dot and cond_open to find currently generating entity position
            sep_dot_index = token_index["SEP_DOT"][-1] if len(token_index["SEP_DOT"]) != 0 else -10000
            cond_open_index = token_index["COND_OPEN"][-1] if len(token_index["COND_OPEN"]) != 0 else -10000
            
            for token_last_index in [sep_dot_index, cond_open_index]:
                if token_last_index + 1 > entity_start_pos:
                    entity_start_pos = token_last_index + 1

            prefix = input_ids[entity_start_pos:]

            # ASK
            if self.sparql_vocab["ASK"] in input_ids:
                candidate_tokens = self.entity_trie.get(prefix)

            # SELECT
            else:
                # whether var or entity
                if len(prefix) == 0:
                    candidate_tokens = (
                        self.head_rel_trie.get(prefix)
                        + self.sparql_vocab["VARS"]
                    )
                    if input_ids[-1] == self.sparql_vocab["SEP_DOT"]:
                        candidate_tokens.append(self.sparql_vocab["FILTER"])
                else:
                    candidate_tokens = self.head_rel_trie.get(prefix)

        elif status == Status.RELATION:
            head_start, head_end = self.find_head_entity(input_ids, token_index)

            prefix = input_ids[head_end + 1 :]
            # relation prunning

            # ASK
            if self.sparql_vocab["ASK"] in input_ids:
                candidate_tokens = self.relation_trie.get(prefix)

            # SELECT
            else:
                if head_start == head_end:
                    # head entity is a variable 
                    valid_relation_tokens = self.rel_tail_trie.get(prefix)
                else:
                    head_entity = tuple(input_ids[head_start : head_end + 1])
                    # get_from_subset
                    # valid_relations = self.relation_trie.get(prefix)
                    if head_entity not in self.head_rel_cache:
                        self.head_rel_cache[head_entity] = Trie(
                            [
                                self.relation_trie.get_tokens_from_id(x)
                                for x in self.head_rel[self.entity_trie.get_id_from_tokens(head_entity)]
                            ]
                        )
                    valid_relation_tokens = self.head_rel_cache[head_entity].get(prefix)

                # only concrete relation possible
                candidate_tokens = valid_relation_tokens

        elif status == Status.TAIL_ENTITY:
            head_start, head_end = self.find_head_entity(input_ids, token_index)
            relation_start, relation_end = self.find_relation(head_end + 1, input_ids, token_index)
            head_entity = tuple(input_ids[head_start : head_end + 1])
            relation = tuple(input_ids[relation_start : relation_end + 1])

            prefix = input_ids[relation_end + 1 :]

            # ASK
            if self.sparql_vocab["ASK"] in input_ids:
                candidate_tokens = self.entity_trie.get(prefix)

            # SELECT
            else:
                # Concrete head and relation
                if head_start != head_end:
                    return self.sparql_vocab["VARS"]

                if relation not in self.relation_tail_cache:
                    self.relation_tail_cache[relation] = Trie(
                        [
                            self.entity_trie.get_tokens_from_id(x)
                            for x in self.relation_tail[self.relation_trie.get_id_from_tokens(relation)]
                        ]
                    )
                valid_tail_tokens = self.relation_tail_cache[relation].get(prefix)

                # whether var or entity
                if len(prefix) == 0:
                    candidate_tokens = valid_tail_tokens + self.sparql_vocab["VARS"]
                else:
                    candidate_tokens = valid_tail_tokens

        elif status == Status.CONTINUE_OR_NOT:
            candidate_tokens = [
                self.sparql_vocab["SEP_DOT"],
                self.sparql_vocab["COND_CLOSE"],
            ]
        # Status.NORMAL
        else:
            last_wrap_open_idx = token_index["WRAP_OPEN"][-1] if len(token_index["WRAP_OPEN"]) != 0 else -10000
            last_wrap_close_idx = token_index["WRAP_CLOSE"][-1] if len(token_index["WRAP_CLOSE"]) != 0 else -10000
            if last_wrap_open_idx > last_wrap_close_idx:
                # relation generation during filter
                prefix = input_ids[last_wrap_open_idx:]
                candidate_tokens = self.relation_trie.get(prefix)
            else:
                candidate_tokens = self.normal_tokens

        return candidate_tokens

    def find_head_entity(self, input_ids: List[int], token_index):
        head_start_pos = -10000

        sep_dot_index = token_index["SEP_DOT"][-1] if len(token_index["SEP_DOT"]) != 0 else -10000
        cond_open_index = token_index["COND_OPEN"][-1] if len(token_index["COND_OPEN"]) != 0 else -10000
        for token_last_index in [sep_dot_index, cond_open_index]:
            if token_last_index + 1 > head_start_pos:
                head_start_pos = token_last_index + 1
        # head entity is a variable (?x)
        if input_ids[head_start_pos] in self.sparql_vocab["VARS"]:
            return (head_start_pos, head_start_pos)

        # head entity is a concrete entity
        head_end_pos = -10000
        for pos in token_index["WRAP_CLOSE"]:
            if pos > head_start_pos:
                head_end_pos = pos
                break

        return (head_start_pos, head_end_pos)

    def find_relation(self, relation_start_pos: int, input_ids: List[int], token_index: Dict):
        # relation is a variable
        if input_ids[relation_start_pos] in self.sparql_vocab["VARS"]:
            return (relation_start_pos, relation_start_pos)
        
        relation_end_pos = -10000
        for pos in token_index["WRAP_CLOSE"]:
            if pos > relation_start_pos:
                relation_end_pos = pos
                break

        return (relation_start_pos, relation_end_pos)


class AblationSparqlTokenGenerator:
    def __init__(
        self,
        tokenizer,
        entity_trie: Trie,
        relation_trie: Trie,
        head_rel: Dict,
        relation_tail: Dict,
        sparql_vocab: Dict,
        config,
    ):
        self.tokenizer = tokenizer
        self.form_start_index = 1 if config.forced_bos_token_id is None else 2

        self.wrap_token_ids = [
            encode_fn(UniqueToken.WRAP_OPEN.value, tokenizer),
            encode_fn(UniqueToken.WRAP_CLOSE.value, tokenizer),
        ]
        self.entity_trie = entity_trie
        self.relation_trie = relation_trie
        print(len(entity_trie))

        assert all(
            [
                True if key in sparql_vocab else False
                for key in [
                    "SELECT",
                    "ASK",
                    "VARS",
                    "WHERE",
                    "COND_OPEN",
                    "COND_CLOSE",
                    "FILTER",
                ]
            ]
        ), "Pre-defined keys are not in the `sparql_vocab`"
        self.sparql_vocab = sparql_vocab
        
        self.normal_tokens = list(range(len(tokenizer)))
        self.normal_tokens.remove(self.sparql_vocab["ORDER_BY"])
        self.select_var_tokens = self.sparql_vocab["VARS"] + \
            [self.sparql_vocab["WHERE"]] + \
            [
                encode_fn(token, self.tokenizer)
                for token in [
                    UniqueToken.COUNT.value,
                    UniqueToken.ATTR_OPEN.value,
                    UniqueToken.ATTR_CLOSE.value,
                ]
            ]
        self.select_var_tokens = set(self.select_var_tokens)
        
    def get_token_indices(self, generated_tokens: List[int]):
        token_index = {
            "COND_OPEN": [],
            "SEP_DOT": [],
            "COND_CLOSE": [],
            "FILTER": [],
            "WRAP_OPEN": [],
            "WRAP_CLOSE": [],
        }
        for i, token in enumerate(generated_tokens):
            if token == self.sparql_vocab["COND_OPEN"]:
                token_index["COND_OPEN"].append(i)
            elif token == self.sparql_vocab["SEP_DOT"]:
                token_index["SEP_DOT"].append(i)
            elif token == self.sparql_vocab["COND_CLOSE"]:
                token_index["COND_CLOSE"].append(i)
            elif token == self.sparql_vocab["FILTER"]:
                token_index["FILTER"].append(i)
            elif token == self.sparql_vocab["WRAP_OPEN"]:
                token_index["WRAP_OPEN"].append(i)
            elif token == self.sparql_vocab["WRAP_CLOSE"]:
                token_index["WRAP_CLOSE"].append(i)
        
        return token_index

    def get_status(self, generated_tokens: List[int], token_index: Dict):
        # Just started, only decoder_start_token_id, forced_bos_token_id
        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L693
        # https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/generation/logits_process.py#L882
        
        if len(generated_tokens) == self.form_start_index:
            return Status.FORM

        # Query form generated, but not opend condition clause yet
        if self.sparql_vocab["WHERE"] not in generated_tokens:
            return Status.VARS

        cond_open_index = token_index["COND_OPEN"][-1] if len(token_index["COND_OPEN"]) != 0 else -10000
        sep_dot_index = token_index["SEP_DOT"][-1] if len(token_index["SEP_DOT"]) != 0 else -10000
        cond_close_index = token_index["COND_CLOSE"][-1] if len(token_index["COND_CLOSE"]) != 0 else -10000
        filter_index = token_index["FILTER"][-1] if len(token_index["FILTER"]) != 0 else -10000

        if max(cond_close_index, filter_index) > max(cond_open_index, sep_dot_index):
            return Status.NORMAL

        last_triple_start_index = max(cond_open_index, sep_dot_index)
        num_wrap_tokens = 0
        for token in generated_tokens[last_triple_start_index:]:
            if token in self.wrap_token_ids:
                num_wrap_tokens += 1
            if token in self.sparql_vocab["VARS"]:
                num_wrap_tokens += 2

        # After one triple created -> decide to continue or not
        if num_wrap_tokens % 6 == 0 and (
            generated_tokens[-1] == self.wrap_token_ids[1]
            or generated_tokens[-1] in self.sparql_vocab["VARS"]
        ):
            return Status.CONTINUE_OR_NOT

        # After condition open or sep dot, triple generation
        if num_wrap_tokens % 6 in [0, 1]:
            return Status.HEAD_ENTITY
        elif num_wrap_tokens % 6 in [2, 3]:
            return Status.RELATION
        else:
            return Status.TAIL_ENTITY

    @verbose_generator
    def generate(self, batch_id: int, input_ids: torch.Tensor, verbose=False):
        """
        A beam-search process function which acts as a `prefix_allowed_tokens_fn` in Huggingface transformers.
        For more details, refer to https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.generate
        Args:
            batch_id (int): batch id
            input_ids (torch.Tensor): Tokenized sentence that is currently generated by auto regressive way at each step.
        Returns:
            List[int]: Possible tokens the model can generate at this moment.
        Examples:
            ...
            >>> print(trie_search_processor.process(torch.Tensor([1,2,3])))
            [1, 4, 5]
        """
        # print(input_ids)
        input_ids = input_ids.tolist()

        # attach where and {
        if input_ids[-1] == self.sparql_vocab["WHERE"]:
            return [self.sparql_vocab["COND_OPEN"]]
        if input_ids[-1] == self.sparql_vocab["COND_CLOSE"]:
            return [self.sparql_vocab["ORDER_BY"], self.tokenizer.eos_token_id]
        
        token_index = self.get_token_indices(input_ids)

        status = self.get_status(input_ids, token_index)
        candidate_tokens = []

        if status == Status.FORM:
            candidate_tokens = [self.sparql_vocab["SELECT"], self.sparql_vocab["ASK"]]

        elif status == Status.VARS:
            # ASK
            if self.sparql_vocab["ASK"] in input_ids:
                return [self.sparql_vocab["WHERE"]]

            # SELECT
            candidate_tokens = list(self.select_var_tokens - set(input_ids[2:]))

        elif status == Status.HEAD_ENTITY:
            entity_start_pos = -10000

            # last position for sep_dot and cond_open to find currently generating entity position
            sep_dot_index = token_index["SEP_DOT"][-1] if len(token_index["SEP_DOT"]) != 0 else -10000
            cond_open_index = token_index["COND_OPEN"][-1] if len(token_index["COND_OPEN"]) != 0 else -10000
            
            for token_last_index in [sep_dot_index, cond_open_index]:
                if token_last_index + 1 > entity_start_pos:
                    entity_start_pos = token_last_index + 1

            prefix = input_ids[entity_start_pos:]

            # ASK
            if self.sparql_vocab["ASK"] in input_ids:
                candidate_tokens = self.entity_trie.get(prefix)

            # SELECT
            else:
                # whether var or entity
                if len(prefix) == 0:
                    candidate_tokens = (
                        self.entity_trie.get(prefix)
                        + self.sparql_vocab["VARS"]
                    )
                    if input_ids[-1] == self.sparql_vocab["SEP_DOT"]:
                        candidate_tokens.append(self.sparql_vocab["FILTER"])
                else:
                    candidate_tokens = self.entity_trie.get(prefix)

        elif status == Status.RELATION:
            head_start, head_end = self.find_head_entity(input_ids, token_index)

            prefix = input_ids[head_end + 1 :]
            
            candidate_tokens = self.relation_trie.get(prefix)

        elif status == Status.TAIL_ENTITY:
            head_start, head_end = self.find_head_entity(input_ids, token_index)
            relation_start, relation_end = self.find_relation(head_end + 1, input_ids, token_index)
            head_entity = tuple(input_ids[head_start : head_end + 1])
            relation = tuple(input_ids[relation_start : relation_end + 1])

            prefix = input_ids[relation_end + 1 :]

            # ASK
            if self.sparql_vocab["ASK"] in input_ids:
                candidate_tokens = self.entity_trie.get(prefix)

            # SELECT
            else:
                # Concrete head and relation
                if head_start != head_end:
                    return self.sparql_vocab["VARS"]

                valid_tail_tokens = self.entity_trie.get(prefix)

                # whether var or entity
                if len(prefix) == 0:
                    candidate_tokens = valid_tail_tokens + self.sparql_vocab["VARS"]
                else:
                    candidate_tokens = valid_tail_tokens

        elif status == Status.CONTINUE_OR_NOT:
            candidate_tokens = [
                self.sparql_vocab["SEP_DOT"],
                self.sparql_vocab["COND_CLOSE"],
            ]
        # Status.NORMAL
        else:
            last_wrap_open_idx = token_index["WRAP_OPEN"][-1] if len(token_index["WRAP_OPEN"]) != 0 else -10000
            last_wrap_close_idx = token_index["WRAP_CLOSE"][-1] if len(token_index["WRAP_CLOSE"]) != 0 else -10000
            if last_wrap_open_idx > last_wrap_close_idx:
                # relation generation during filter
                prefix = input_ids[last_wrap_open_idx:]
                candidate_tokens = self.relation_trie.get(prefix)
            else:
                candidate_tokens = self.normal_tokens

        return candidate_tokens

    def find_head_entity(self, input_ids: List[int], token_index):
        head_start_pos = -10000

        sep_dot_index = token_index["SEP_DOT"][-1] if len(token_index["SEP_DOT"]) != 0 else -10000
        cond_open_index = token_index["COND_OPEN"][-1] if len(token_index["COND_OPEN"]) != 0 else -10000
        for token_last_index in [sep_dot_index, cond_open_index]:
            if token_last_index + 1 > head_start_pos:
                head_start_pos = token_last_index + 1
        # head entity is a variable (?x)
        if input_ids[head_start_pos] in self.sparql_vocab["VARS"]:
            return (head_start_pos, head_start_pos)

        # head entity is a concrete entity
        head_end_pos = -10000
        for pos in token_index["WRAP_CLOSE"]:
            if pos > head_start_pos:
                head_end_pos = pos
                break

        return (head_start_pos, head_end_pos)

    def find_relation(self, relation_start_pos: int, input_ids: List[int], token_index: Dict):
        # relation is a variable
        if input_ids[relation_start_pos] in self.sparql_vocab["VARS"]:
            return (relation_start_pos, relation_start_pos)
        
        relation_end_pos = -10000
        for pos in token_index["WRAP_CLOSE"]:
            if pos > relation_start_pos:
                relation_end_pos = pos
                break

        return (relation_start_pos, relation_end_pos)
