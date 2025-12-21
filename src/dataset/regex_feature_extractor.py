import numpy as np
import re
import os


class RegexFeatureExtractor:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.regexes: list[re.Pattern] = []

    def load_regexes(self,
                     save_path: str,
                     dna_values: list[str],
                     dna_labels: np.ndarray,
                     force_regex_recreation: bool = False
                     ):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if not os.path.exists(save_path) or force_regex_recreation:
            self._fit_and_save(save_path, dna_values, dna_labels)

        with open(save_path, 'r') as f:
            for regex_str in f.readlines():
                clean_regex = regex_str.strip()
                if clean_regex:
                    self.regexes.append(re.compile(clean_regex))

        if self.tokenizer is not None and \
            len(self.regexes) != self.tokenizer.vocab_size:
            raise ValueError("RegEx file exists, but has different size than tokenizer's vocab size")

        return self.regexes

    def transform(self, sequences: list[str]) -> np.ndarray:
        seq_length = len(sequences[0])  # NOTE: we assume all sequences has the same length
        regex_features_extracted = np.zeros(
            shape=(len(sequences), seq_length*len(self.regexes)), dtype=np.int8
        )

        for indx, seq in enumerate(sequences):
            regex_features_extracted[indx, :] = self._extract_features_for_seq(seq)

        return regex_features_extracted

    def _fit_and_save(self, save_path: str, dna_values: list[str], dna_labels: np.ndarray):
        positives = [seq for seq, label in zip(dna_values, dna_labels) if label==1]
        if self.tokenizer is None:
            raise Exception("No default regexes. Add tokenizer to generate them.")

        bpe_codes = self.tokenizer.fit(positives)

        with open(save_path, 'w') as f:
            for pattern in bpe_codes:
                regex = "".join(pattern).replace("-", "")
                f.write(regex + '\n')

    def _extract_features_for_seq(self, seq):
        seq_len = len(seq) # NOTE: we assume all sequences has the same length
        n_regexes = len(self.regexes)
        features = np.zeros(shape=(seq_len * n_regexes), dtype=np.int8)

        for i, rg in enumerate(self.regexes):
            L = len(rg.pattern)
            base_index = i * seq_len

            for off in range(seq_len):
                if off + L > seq_len:
                    continue
                s = seq[off : off+L]
                if rg.fullmatch(s):
                    final_index = base_index + off
                    features[final_index] = 1

        return features
