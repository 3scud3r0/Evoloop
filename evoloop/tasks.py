"""Família de tarefas sintéticas verificáveis. Split: evolve (5 famílias) / ood (3 famílias nunca vistas no loop)."""
from __future__ import annotations
import random
import re
from dataclasses import dataclass
from typing import Callable

WORDS = ("granite harbor lantern violet copper meadow orbit quartz saddle timber velvet walnut yonder zephyr "
         "anchor bramble cinder dusk ember falcon glacier hollow island juniper kestrel lagoon marble nectar "
         "opal prairie quiver ridge summit tundra umber vortex willow xenon yarrow zenith aurora basalt canyon "
         "delta estuary fjord gorge horizon").split()
VOW = "aeiou"


def _rev(s): return s[::-1]
def _upv(s): return "".join(c.upper() if c in VOW else c for c in s)
def _dsum(s): return str(sum(int(c) for c in s if c.isdigit()))
def _sortw(s): return " ".join(sorted(s.split()))
def _cntv(s): return str(sum(c in VOW for c in s.lower()))
def _caesar(s): return "".join(chr((ord(c) - 97 + 3) % 26 + 97) if "a" <= c <= "z" else c for c in s)
def _dedupe(s):
    out = []
    for w in s.split():
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)
def _longest(s):
    best = ""
    for w in s.split():
        if len(w) > len(best):
            best = w
    return best


def _g_word(r): return r.choice(WORDS)
def _g_words(n): return lambda r: " ".join(r.choice(WORDS) for _ in range(n if isinstance(n, int) else r.randint(*n)))
def _g_alnum(r):
    while True:
        s = "".join(r.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(r.randint(5, 9)))
        if sum(c.isdigit() for c in s) >= 2:
            return s
def _g_dups(r):
    ws = [r.choice(WORDS) for _ in range(r.randint(3, 5))]
    out = []
    for w in ws:
        out += [w] * r.choice((1, 1, 2, 3))
    return " ".join(out)


@dataclass(frozen=True)
class Family:
    name: str
    split: str                         # evolve | ood
    instruction: str
    solve: Callable[[str], str]
    gen: Callable[[random.Random], str]
    shape_ok: Callable[[str, str], bool]   # verificador FRACO (só forma), propositalmente
    base_p: float                      # prob. de acerto do LLM simulado sem ajuda
    keys: tuple                        # regex que "cobrem" a família num prompt/lição/regra (usado pelo SimLLM)


FAMILIES: dict[str, Family] = {f.name: f for f in [
    Family("reverse", "evolve", "Reverse the characters of the input string.", _rev, _g_word,
           lambda i, a: len(a) == len(i), 0.30, (r"revers",)),
    Family("upper_vowels", "evolve", "Uppercase only the vowels of the input string; leave every other character unchanged.",
           _upv, _g_word, lambda i, a: len(a) == len(i), 0.30, (r"vowel", r"upper")),
    Family("digit_sum", "evolve", "Return the sum of all digits that appear in the input string, as a number.",
           _dsum, _g_alnum, lambda i, a: a.isdigit(), 0.35, (r"digit", r"sum")),
    Family("sort_words", "evolve", "Sort the words of the input alphabetically and join them with single spaces.",
           _sortw, _g_words((4, 5)), lambda i, a: len(a.split()) == len(i.split()), 0.40, (r"sort", r"word")),
    Family("count_vowels", "evolve", "Count the vowels (a, e, i, o, u) in the input and return the count as a number.",
           _cntv, _g_words((2, 3)), lambda i, a: a.isdigit(), 0.40, (r"vowel", r"count")),
    Family("caesar3", "ood", "Shift every lowercase letter of the input forward by 3 positions in the alphabet, wrapping z to c; keep other characters.",
           _caesar, _g_words((1, 2)), lambda i, a: len(a) == len(i), 0.25, (r"shift|caesar|cipher", r"letter|alphabet")),
    Family("dedupe_words", "ood", "Remove consecutive duplicate words from the input, keeping the first of each run.",
           _dedupe, _g_dups, lambda i, a: 0 < len(a) <= len(i), 0.30, (r"duplicate|repeated", r"word")),
    Family("longest_word", "ood", "Return the longest word of the input (the first one if there is a tie).",
           _longest, _g_words((3, 5)), lambda i, a: a.isalpha(), 0.35, (r"longest", r"word")),
]}
EVOLVE_FAMILIES = [n for n, f in FAMILIES.items() if f.split == "evolve"]
OOD_FAMILIES = [n for n, f in FAMILIES.items() if f.split == "ood"]
OOD_PROBE = ["caesar3"]                                   # entra no laço (sinal de transferência para a seleção)
OOD_TEST = [f for f in OOD_FAMILIES if f not in OOD_PROBE]  # NUNCA visto pelo laço: só o relatório final
LOOP_FAMILIES = EVOLVE_FAMILIES + OOD_PROBE


# Reformulações SEM as palavras-chave da instrução canônica (mede generalização semântica de memória/roteamento).
PARAPHRASES: dict[str, tuple] = {
    "reverse": ("Flip the string backwards.", "Write the text back to front."),
    "upper_vowels": ("Capitalize just the a, e, i, o and u letters of the input; everything else stays as it is.",
                     "Turn a, e, i, o and u into capitals and leave every other symbol untouched."),
    "digit_sum": ("Add up every numeral found inside the input and give the total.",
                  "Find the numerals in the input and report their total."),
    "sort_words": ("Put the words of the input in alphabetical order, separated by one space.",
                   "Rearrange the input words alphabetically, single space between them."),
    "count_vowels": ("How many of the letters a, e, i, o, u does the input contain? Answer with the number.",
                     "Tally the letters a, e, i, o, u in the input and answer with the number."),
    "caesar3": ("Move each lowercase letter three places along the alphabet (z wraps to c), leaving other characters alone.",
                "Replace every lowercase letter by the one three positions later in the alphabet, wrapping z to c."),
    "dedupe_words": ("Collapse runs of the same word into a single occurrence.",
                     "Whenever a word is immediately followed by itself, keep only one copy."),
    "longest_word": ("Give back the word with the most characters (earliest wins ties).",
                     "Return the input word that has the greatest number of characters; on a tie take the first."),
}


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    instruction: str
    input: str
    expected: str
    variant: int = 0          # 0 = instrução canônica; >0 = paráfrase

    def prompt(self) -> str:
        return f"### TASK {self.id}\n{self.instruction}\nINPUT: {self.input}"

    def check(self, answer: str) -> bool:           # grader estrito (oráculo do avaliador)
        return answer.strip() == self.expected

    def shape_ok(self, answer: str) -> bool:        # verificador fraco (o que o agente vê)
        return FAMILIES[self.family].shape_ok(self.input, answer.strip())


def make_tasks(families: list[str], n_per_family: int, seed: int, prefix: str = "t", para_rate: float = 0.0) -> list[Task]:
    out = []
    for fname in families:
        fam = FAMILIES[fname]
        rng = random.Random(f"{seed}:{fname}")
        prng = random.Random(f"{seed}:{fname}:para")      # fluxo separado: as ENTRADAS não mudam com para_rate
        for i in range(n_per_family):
            inp = fam.gen(rng)
            v = 0
            if para_rate > 0 and prng.random() < para_rate:
                v = 1 + prng.randrange(len(PARAPHRASES[fname]))
            ins = fam.instruction if v == 0 else PARAPHRASES[fname][v - 1]
            out.append(Task(f"{prefix}{seed}-{fname}-{i}", fname, ins, inp, fam.solve(inp), v))
    return out


def family_of_instruction(text: str) -> str | None:
    for name, fam in FAMILIES.items():
        if fam.instruction in text or any(p in text for p in PARAPHRASES[name]):
            return name
    return None


def covers(family: str, text: str) -> bool:
    """O texto contém uma regra/lição/ferramenta que cobre a família? (todas as chaves casam)."""
    t = text.lower()
    return all(re.search(k, t) for k in FAMILIES[family].keys)
