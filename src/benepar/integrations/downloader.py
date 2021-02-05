import os

BENEPAR_SERVER_INDEX = "https://kitaev.com/benepar/index.xml"

_downloader = None
def get_downloader():
    global _downloader
    if _downloader is None:
        import nltk.downloader
        _downloader = nltk.downloader.Downloader(server_index_url=BENEPAR_SERVER_INDEX)
    return _downloader

def download(*args, **kwargs):
    return get_downloader().download(*args, **kwargs)

def locate_model(name):
    if os.path.exists(name):
        return name
    elif "/" not in name and "." not in name:
        import nltk.data
        try:
            nltk_loc = nltk.data.find(f"models/{name}")
            return nltk_loc.path
        except LookupError as e:
            arg = e.args[0].replace("nltk.download", "benepar.download")
        
        raise LookupError(arg)
    
    raise LookupError("Can't find {}".format(name))

def load_trained_model(model_name_or_path):
    model_path = locate_model(model_name_or_path)
    from ..parse_chart import ChartParser
    parser = ChartParser.from_trained(model_path)
    return parser
