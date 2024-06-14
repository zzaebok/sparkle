import sys
from typing import List
from collections import defaultdict
import marisa_trie

sys.setrecursionlimit(10000)


class Trie:
    def __init__(
        self,
        sequences: List[List[int]] = [],
    ):
        zero_depth = set()
        first_depth = set()
        second_depth = dict()
        
        components = []
        for sequence in sequences:
            zero_depth.add(sequence[0])
            first_depth.add(sequence[1])
            if sequence[0] not in second_depth:
                second_depth[sequence[0]] = dict()
            if sequence[1] not in second_depth[sequence[0]]:
                second_depth[sequence[0]][sequence[1]] = set()
            
            if len(sequence) >= 3:
                second_depth[sequence[0]][sequence[1]].add(sequence[2])
            components.append("".join([chr(i) for i in sequence]))

        assert len(zero_depth) == 1
        self.zero_depth = list(zero_depth)
        self.first_depth = list(first_depth)
        self.second_depth = second_depth

        self.trie = marisa_trie.Trie(components)

    def get(self, prefix_sequence: List[int]):
        try:
            if len(prefix_sequence) == 0:
                return self.zero_depth
            if len(prefix_sequence) == 1 and (prefix_sequence[0] == self.zero_depth[0]):
                return self.first_depth
            if len(prefix_sequence) == 2:
                return list(self.second_depth[prefix_sequence[0]][prefix_sequence[1]])

            key = "".join([chr(i) for i in prefix_sequence])
            return list(
                ord(e[len(key)]) for e in self.trie.keys(key) if len(e) > len(key)
            )
        except:
            return []
    
    def get_id_from_tokens(self, tokens: tuple):
        return self.trie.key_id("".join([chr(i) for i in tokens]))

    def get_tokens_from_id(self, id: int):
        return tuple([ord(c) for c in self.trie.restore_key(id)])

    def __iter__(self):
        for sequence in self.trie.iterkeys():
            yield [ord(e) for e in sequence]

    def __len__(self):
        return len(self.trie)

    def __getitem__(self, value):
        return self.get(value)

