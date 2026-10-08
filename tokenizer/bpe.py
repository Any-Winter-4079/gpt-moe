"""Readable byte-level BPE with no external dependencies.

Each document is a separate byte sequence. Merges can cross spaces and word
boundaries within a document; there is no GPT-2 regex splitting or special tokens.
Training recounts all pairs after each merge, so start with small text samples.

Example:
    tokenizer = BytePairEncoding()
    tokenizer.train(["banana banana", "bandana"], vocab_size=264)
    tokens = tokenizer.encode("banana")
    text = tokenizer.decode(tokens)
"""


def merge_pair(tokens: list[int], pair: tuple[int, int], token_id: int) -> list[int]:
    merged = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            merged.append(token_id)
            # consume both tokens so replacements cannot overlap
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


class BytePairEncoding:
    def __init__(self):
        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        self.merges = {}

    def train(self, documents: list[str], vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 to represent every byte")

        # start from the 256 byte values on each training call
        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        self.merges = {}
        sequences = [list(document.encode("utf-8")) for document in documents]

        while len(self.vocab) < vocab_size:
            counts = {}
            for tokens in sequences:
                # adjacent pairs overlap when counting: 'aaa' contains two 'aa' pairs
                for left, right in zip(tokens, tokens[1:]):
                    pair = (left, right)
                    counts[pair] = counts.get(pair, 0) + 1

            if not counts:
                break

            # ties go to the first pair encountered in document order
            pair = max(counts, key=counts.get)
            token_id = len(self.vocab)
            self.merges[pair] = token_id
            self.vocab[token_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            sequences = [merge_pair(tokens, pair, token_id) for tokens in sequences]

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        # dictionaries preserve insertion order, which is the learned merge order
        for pair, token_id in self.merges.items():
            tokens = merge_pair(tokens, pair, token_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        # join bytes before decoding since a token may contain only part of a character
        return b"".join(self.vocab[token_id] for token_id in tokens).decode("utf-8")
