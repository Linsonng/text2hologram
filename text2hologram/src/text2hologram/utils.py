import os

def create_dirs(paths):
    for path in paths:
        if not os.path.isdir(path) and path != '':
            os.makedirs(path)

class c:
    title = '\033[96m'
    ok = '\033[92m'
    okb = '\033[94m'
    warn = '\033[93m'
    fail = '\033[31m'
    endc = '\033[0m'
    bold = '\033[1m'
    dark = '\33[90m'
    u = '\033[4m'


