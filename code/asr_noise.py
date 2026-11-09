# -*- coding: utf-8 -*-
"""asr_noise.py - Adds synthetic transcription noise to the benchmark (word error rate ~15%).

This is not a model of a speech recogniser. It perturbs each word independently (substitution
by a word of the same language drawn from the benchmark vocabulary, deletion, or insertion of a
filler), which approximates the word-level error rates reported for speech recognition in noisy
industrial settings, so that the interpretation layer can be stressed with imperfect text.
"""
import json, random, re

WER = 0.15
FILLERS = {"en": ["uh", "um", "the", "a", "and"], "es": ["eh", "este", "o", "la", "y"]}


def noisy(text, lang, vocab, rng):
    out = []
    for w in text.split():
        if rng.random() < WER:
            u = rng.random()
            if u < 0.5: out.append(rng.choice(vocab[lang]))
            elif u < 0.8: continue
            else: out += [w, rng.choice(FILLERS[lang])]
        else:
            out.append(w)
    return " ".join(out) if out else text


if __name__ == "__main__":
    rows = json.load(open("../results/corpus.json", encoding="utf-8"))
    vocab = {l: sorted({w for r in rows if r["lang"] == l for w in r["utterance"].split()}) for l in ("en", "es")}
    rng = random.Random(2026)
    for r in rows: r["utterance_asr"] = noisy(r["utterance"], r["lang"], vocab, rng)
    json.dump(rows, open("../results/corpus_asr.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    n = sum(r["utterance"] != r["utterance_asr"] for r in rows)
    print(n, "of", len(rows), "utterances perturbed")
