from collections import deque

import numpy as np
from numba import njit

from index import Index, Matches, Label


class bidict(dict):

    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)


@njit
def index_in_array(array: np.ndarray, item: Label) -> int:
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

    return len(array)


def semantic_join(a: Index, b: Index, resolution: int = 10) -> bidict:
    """Performs a Semantic Join, matching one entry from `a` with 
    one entry from `b`, converging towards a Stable Marriage.
    Assuming the collections can be different in size, classical solution
    doesn't provide 

    :param resolution: Approximate matches per member to consider, defaults to 10
    :type resolution: int, optional
    :return: Bidirectional mapping from men to women and `.inverse`
    :rtype: bidict
    """

    men, women = (a, b) if len(a) > len(b) else (b, a)
    matches = bidict()

    men_vectors = np.vstack(men[i] for i in range(len(men)))
    women_vectors = np.vstack(women[i] for i in range(len(women)))

    man_count_proposed = np.zeros(len(men))
    man_to_women_preferences: Matches = men.search(women_vectors, resolution)
    woman_to_men_preferences: Matches = women.search(men_vectors, resolution)

    # A nice optimization is to prioritize free men by the quality of their
    # best remaining variant. TODO: Replace with `heapq`
    free_men = deque(range(len(men)))

    idle_cycles: int = 0
    hopeless_men_count = len(men) - len(women)
    while len(free_men) > hopeless_men_count:

        # In the worst case scenario we may need to match more candidates for every
        # remaining man. This, however, may drastically increase the runtime.
        if len(free_men) == idle_cycles:
            break

        man: Label = free_men.popleft()
        count_proposals = man_count_proposed[man]
        if count_proposals == man_to_women_preferences.counts[man]:
            free_men.append(man)
            idle_cycles += 1
            continue

        woman: Label = man_to_women_preferences.labels[man, count_proposals]
        man_count_proposed[man] += 1

        if woman not in matches.inverse:
            matches[man] = woman

        else:
            her_preferences = woman_to_men_preferences.labels[woman, :]
            husband: Label = matches.inverse[woman]
            husband_rank = index_in_array(her_preferences, husband)
            challenger_rank = index_in_array(her_preferences, man)
            if challenger_rank < husband_rank:
                del matches[husband]
                free_men.append(husband)
                matches[man] = woman

            else:
                free_men.append(man)

    return matches
