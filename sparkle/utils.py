from enum import Enum


class UniqueToken(Enum):
    WRAP_OPEN = "[ "
    WRAP_CLOSE = " ] "
    WHERE = "WHERE "
    COND_OPEN = "brack_open "
    COND_CLOSE = "brack_close "
    SEP_DOT = "sep_dot "
    COUNT = "COUNT "
    ATTR_OPEN = "attr_open "
    ATTR_CLOSE = "attr_close "
    ORDER_BY = "ORDER BY "
    LIMIT = "LIMIT "
    FILTER = "FILTER "
    EXISTS = "EXISTS "
    NOT = "NOT "
    OR = "expr_or "
    DESC = "DESC "
    
class SexprUniqueToken(Enum):
    WRAP_OPEN = "[ "
    WRAP_CLOSE = " ] "
    COUNT = "COUNT "
    ARGMAX = "ARGMAX "
    ARGMIN = "ARGMIN "
    JOIN = "JOIN "
    AND = "AND "
    ATTR_OPEN = "attr_open "
    ATTR_CLOSE = "attr_close "
    R = "R "
    LE = "le "
    LT = "lt "
    GE = "ge "
    GT = "gt "


def encode_fn(x, tokenizer):
    return tokenizer.encode(x, add_special_tokens=False)[0]

def decode_fn(x, tokenizer):
    return tokenizer.decode(x)

def verbose_generator(generate_func):
    def wrapper(self, batch_id, input_ids, verbose=False):
        candidate_tokens = generate_func(self, batch_id, input_ids, verbose)
        if verbose:
            print("---")
            print("batch: ", batch_id)
            print("input_ids: ", input_ids)
            print("candidates: ", candidate_tokens[:10], "Length: ", len(candidate_tokens))
        return candidate_tokens
    return wrapper