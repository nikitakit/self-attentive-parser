BENEPAR_SERVER_INDEX = "https://kitaev.io/benepar_models/index.xml"

_downloader = None
def get_downloader():
    global _downloader
    if _downloader is None:
        import nltk.downloader
        _downloader = nltk.downloader.Downloader(server_index_url=BENEPAR_SERVER_INDEX)
    return _downloader

def download(*args, **kwargs):
    return get_downloader().download(*args, **kwargs)

def load_model(name):
    name = "models/{}.gz".format(name)
    import nltk.data
    try:
        return nltk.data.load(name, format="raw")
    except LookupError as e:
        arg = e.args[0].replace("nltk.download", "benepar.download")
    raise LookupError(arg)
