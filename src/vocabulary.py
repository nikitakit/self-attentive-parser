import collections

class Vocabulary(object):
    def __init__(self):
        self.frozen = False
        self.values = []
        self.indices = {}
        self.counts = collections.defaultdict(int)

    @property
    def size(self):
        return len(self.values)

    def value(self, index):
        assert 0 <= index < len(self.values)
        return self.values[index]

    def index(self, value):
        if not self.frozen:
            self.counts[value] += 1

        if value in self.indices:
            return self.indices[value]

        elif not self.frozen:
            self.values.append(value)
            self.indices[value] = len(self.values) - 1
            return self.indices[value]

        else:
            raise ValueError("Unknown value: {}".format(value))

    def index_or_unk(self, value, unk_value):
        assert self.frozen
        if value in self.indices:
            return self.indices[value]
        else:
            return self.indices[unk_value]

    def count(self, value):
        return self.counts[value]

    def freeze(self):
        self.frozen = True
