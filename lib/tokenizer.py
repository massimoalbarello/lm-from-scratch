import logging

logger = logging.getLogger(__name__)


class Tokenizer:
    U8_SIZE = 255
    PROHIBITED_TOKEN = b"|"

    def __init__(self, vocabulary_size=300) -> None:
        if vocabulary_size <= self.U8_SIZE:
            raise f"vacabulary size has to be greather than {self.U8_SIZE}"

        self.vocabulary_size = vocabulary_size
        self.vocabulary = {i: bytes([i]) for i in range(self.U8_SIZE)}

    def _get_max_occurring(self, sequence):
        stats: dict[(int, int), int] = {}
        for pair in zip(sequence, sequence[1:]):
            stats[pair] = stats.get(pair, 0) + 1

        occurrences = sorted(((v, k) for k, v in stats.items()), reverse=True)
        _, to_be_replaced = max(occurrences)

        return to_be_replaced

    def _merge(self, sequence, to_be_replaced, new_token):
        new_sequence = []
        i = 0
        while i < len(sequence):
            if (
                i < len(sequence) - 1
                and (sequence[i], sequence[i + 1]) == to_be_replaced
            ):
                new_sequence.append(new_token)
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        return new_sequence

    def find_all(self, text_bytes, token):
        indeces = []
        next = 0
        while True:
            index = text_bytes.find(token, next)
            if index == -1:
                return indeces
            indeces.append(index)
            next = index + len(token)

    def train(self, sequence):
        for i in range(self.vocabulary_size - self.U8_SIZE):
            new_token = self.U8_SIZE + i + 1

            to_be_replaced = self._get_max_occurring(sequence)

            sequence = self._merge(sequence, to_be_replaced, new_token)
            logger.info(f"merged pair {to_be_replaced} into new token {new_token}")

            self.vocabulary[new_token] = (
                self.vocabulary[to_be_replaced[0]] + self.vocabulary[to_be_replaced[1]]
            )

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        map = {}
        for i, token in sorted(self.vocabulary.items(), reverse=True):
            if token == self.PROHIBITED_TOKEN:
                continue
            indeces = self.find_all(text_bytes, token)
            if len(indeces) == 0:
                continue
            for index in indeces:
                map[index] = (i, token)
                logger.info(f"found token {token} at index {index}")
            text_bytes = text_bytes.replace(token, self.PROHIBITED_TOKEN * len(token))

        return [pair[0] for _, pair in sorted(map.items())]

    def decode(self, sequence):
        text = b""
        for token in sequence:
            text += self.vocabulary[token]
        # not all byte sequences are valid utf-8
        # replace invalid ones instead of throwing
        return text.decode("utf-8", errors="replace")
