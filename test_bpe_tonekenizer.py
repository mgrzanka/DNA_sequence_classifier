from bpe_tokenizer import BPETokenizer


def test_get_vocab_basic():
    tokenizer = BPETokenizer()
    corpus = ["cat", "cat", "dog"]

    vocab = tokenizer.get_vocab(corpus)

    assert vocab["c a t -"] == 2
    assert vocab["d o g -"] == 1
    assert len(vocab) == 2


def test_get_stats_basic():
    tokenizer = BPETokenizer()
    vocab = {
        "c a t -": 2,
        "d o g -": 1,
    }

    stats = tokenizer.get_stats(vocab)

    assert stats[("c", "a")] == 2
    assert stats[("a", "t")] == 2
    assert stats[("t", "-")] == 2
    assert stats[("d", "o")] == 1
    assert stats[("o", "g")] == 1
    assert stats[("g", "-")] == 1


def test_merge_pair_replaces_correctly():
    tokenizer = BPETokenizer()
    vocab = {
        "c a t -": 2,
        "c a r -": 1,
    }

    new_vocab = tokenizer.merge_pair(("c", "a"), vocab)

    assert "ca t -" in new_vocab
    assert "ca r -" in new_vocab
    assert new_vocab["ca t -"] == 2
    assert new_vocab["ca r -"] == 1


def test_merge_pair_does_not_replace_partial_matches():
    tokenizer = BPETokenizer()
    vocab = {
        "a a a -": 1,
    }

    new_vocab = tokenizer.merge_pair(("a", "a"), vocab)

    assert "aa a -" in new_vocab
    assert new_vocab["aa a -"] == 1


def test_fit_returns_bpe_codes():
    tokenizer = BPETokenizer(vocab_size=5)
    corpus = ["low", "lower", "newest", "widest"]

    bpe_codes = tokenizer.fit(corpus)

    assert isinstance(bpe_codes, dict)
    assert len(bpe_codes) <= tokenizer.vocab_size
    for key in bpe_codes.keys():
        assert isinstance(key, tuple)
        assert len(key) == 2


def test_fit_stops_when_no_stats():
    tokenizer = BPETokenizer(vocab_size=10)
    corpus = ["a"]

    bpe_codes = tokenizer.fit(corpus)

    assert len(bpe_codes) <= 2


def test_empty_corpus():
    tokenizer = BPETokenizer()

    bpe_codes = tokenizer.fit([])

    assert bpe_codes == {}
