class HParams():
    _skip_keys = ['populate_arguments', 'set_from_args', 'print', 'to_dict']
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        if not hasattr(self, item):
            raise KeyError("Hyperparameter {} has not been declared yet".format(item))
        setattr(self, item, value)

    def to_dict(self):
        res = {}
        for k in dir(self):
            if k.startswith('_') or k in self._skip_keys:
                continue
            res[k] =  self[k]
        return res

    def populate_arguments(self, parser):
        for k in dir(self):
            if k.startswith('_') or k in self._skip_keys:
                continue
            v = self[k]
            k = k.replace('_', '-')
            if type(v) in (int, float, str):
                parser.add_argument('--{}'.format(k), type=type(v), default=v)
            elif isinstance(v, bool):
                if not v:
                    parser.add_argument('--{}'.format(k), action='store_true')
                else:
                    parser.add_argument('--no-{}'.format(k), action='store_false')

    def set_from_args(self, args):
        for k in dir(self):
            if k.startswith('_') or k in self._skip_keys:
                continue
            if hasattr(args, k):
                self[k] = getattr(args, k)
            elif hasattr(args, 'no_{}'.format(k)):
                self[k] = getattr(args, 'no_{}'.format(k))

    def print(self):
        for k in dir(self):
            if k.startswith('_') or k in self._skip_keys:
                continue
            print(k, repr(self[k]))
