
import os, sys, json, time
import benepar

from flask import Flask, request
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)

class ConsParser(Resource):
    def put(self):
        segmented_text = request.form['segment'].split()
        if segmented_text == []:
            return {"tree_string": ""}

        tree = cons_parser.parse(segmented_text)
        return {"tree_string": str(tree)}

api.add_resource(ConsParser, '/')

if __name__ == '__main__':
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print('Loading model...')
    cons_parser = benepar.Parser("benepar_zh")
    print('Constituency parsing service is now available')
    app.run(host='0.0.0.0', port=6060)

