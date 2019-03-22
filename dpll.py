import random
from collections import Counter

from cnf import get_random_kcnf, DPLL, CNF


class RandomVarDPLL(DPLL):
    def suggest(self, cnf: CNF):
        var = random.choice(tuple(cnf.vars))
        var *= random.choice([-1, 1])
        return var


class RandomClauseDPLL(DPLL):
    def suggest(self, cnf: CNF):
        clause = random.choice(cnf.clauses)
        var = random.choice(list(clause))
        # We don't randomize a sign, it's on purpose.
        return var


class MostCommonVarDPLL(DPLL):
    def suggest(self, cnf: CNF):
        counter = Counter()
        for clause in cnf.clauses:
            for svar in clause:
                counter[svar] += 1
        return counter.most_common(1)[0][0]


class JeroslowWangDPLL(DPLL):
    def suggest(self, cnf: CNF):
        counter = Counter()

        for clause in cnf.clauses:
            for l in clause:
                counter[l] += 2. ** (-len(clause))

        return counter.most_common(1)[0][0]


def main():
    s = CNF([
        [1, 2],
        [-2, 3],
        [-3],
        [3],
    ])
    print(s)
    print(DPLL().run(s))

    rcnf = get_random_kcnf(3, 4, 20)
    print(rcnf)
    print(DPLL().run(rcnf))
    print(rcnf.simplified())


if __name__ == "__main__":
    main()
