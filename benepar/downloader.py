import json

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
    import nltk.data

    name_gz = "models/{}.gz".format(name)
    name_zip = "models/{}.zip".format(name)
    use_gz = False
    try:
        nltk.data.find(name_gz)
        use_gz = True
    except LookupError:
        pass

    try:
        if use_gz:
            return nltk.data.load(name_gz, format="raw")
        else:
            return {
                'meta': json.loads(nltk.data.load(name_zip + "/meta.json", format="text")),
                'model': nltk.data.load(name_zip + "/model.pb", format="raw"),
                'vocab': nltk.data.load(name_zip + "/vocab.txt", format="text", encoding="utf-8"),
                }
    except LookupError as e:
        arg = e.args[0].replace("nltk.download", "benepar.download")
    raise LookupError(arg)
