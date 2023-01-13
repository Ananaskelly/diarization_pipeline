import json
from dotmap import DotMap


def parse(_p):
    return DotMap(json.load(open(_p, 'r')))
