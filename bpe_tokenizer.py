from collections import defaultdict
import re


class BPETokenizer:
    def __init__(self, vocab_size=50):
        self.vocab_size = vocab_size

    def get_vocab(self, corpus):
        vocab = defaultdict(int)
        for word in corpus:
            vocab[' '.join(list(word)) + ' -'] += 1
        return vocab

    def get_stats(self, vocab):
        stats = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                stats[(symbols[i], symbols[i+1])] += freq
        return stats

    def merge_pair(self, pair, vocab):
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        new_vocab = {}
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

        for word, freq in vocab.items():
            new_word = pattern.sub(replacement, word)
            new_vocab[new_word] = freq

        return new_vocab

    def fit(self, corpus):
        vocab = self.get_vocab(corpus)
        bpe_codes = {}

        for _ in range(self.vocab_size):
            stats = self.get_stats(vocab)
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            bpe_codes[best_pair] = stats[best_pair]

            vocab = self.merge_pair(best_pair, vocab)

        return bpe_codes
