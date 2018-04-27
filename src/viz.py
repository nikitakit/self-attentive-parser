import json
import os
from IPython.display import display
import pandas as pd
import altair as alt

#%%

def viz(template, data):
    with open(os.path.join('viz', template)) as f:
        spec = json.load(f)
    if '$schema$' in spec and 'vega-lite' in spec['$schema']:
        mimetype = 'application/vnd.vegalite.v2+json'
    else:
        mimetype = 'application/vnd.vega.v3+json'

    if isinstance(data, dict):
        newdata = []
        for k, v in data.items():
            chart = alt.Chart(v)
            # chart.max_rows = 50000
            v = json.loads(chart.to_json())['data']['values']
            newdata.append({'name': k, 'values': v})
        data = newdata

    if 'data' not in spec:
        spec['data'] = data
    else:
        provided = {x['name']: x for x in data}
        for i, el in enumerate(spec['data']):
            if el['name'] in provided:
                spec['data'][i] = provided[el['name']]
                del provided[el['name']]
        spec['data'] = list(provided.values()) + spec['data']

    display({mimetype: spec}, raw=True)

#%%

def viz_attention(words, attention):
    def attention_to_elements(attention):
        if len(attention.shape) == 2:
            attention = attention[None,:,:]
        for x in range(attention.shape[0]):
            for y in range(attention.shape[1]):
                for z in range(attention.shape[2]):
                    if attention[x, y, z] > 0.1:
                        yield {'src': y, 'dst': z, 'head': x, 'weight': attention[x,y,z]}

    dat = {'words': pd.DataFrame({'word': words, 'pos': range(len(words))}), 'attention': pd.DataFrame(attention_to_elements(attention))}

    viz('attention.vg.json', dat)
