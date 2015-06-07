from __future__ import print_function
from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer
import optparse
import os.path
import pexpect
import time
from jsonrpclib.jsonrpc import ServerProxy
from pprint import pprint
import multiprocessing as mp


class OpenNLP:
    def __init__(self, host='localhost', port=8080):
        uri = "http://%s:%d" % (host, port)
        self.server = ServerProxy(uri)

    def parse(self, text):
        return self.server.parse(text)

class OpenNLPServer():
    def __init__(self, path):
        opennlp = os.path.join(path, 'bin/opennlp')
        tool = 'Parser'
        models = os.path.join(path, 'models/en-parser-chunking.bin')
        cmd = "%s %s %s" % (opennlp, tool, models)

        # spawn the server
        self.process = pexpect.spawn(cmd)
        self.process.setecho(False)
        self.process.expect('done')
        self.process.expect('\r\n')

    def parse(self, text):
        # Clear any pending output
        try:
            self.process.read_nonblocking(2048, 0)
        except:
            pass

        self.process.sendline(text)

        # Workaround pexpect bug
        self.process.waitnoecho()

        # Long text also needs increase in socket timeout
        timeout = 5 + len(text) / 20.0

        self.process.expect('\r\n', timeout)
        results = self.process.before

        return results

def start_server():
    if os.environ.has_key('OPENNLP'):
        DIRECTORY = os.environ['OPENNLP']
    else:
        DIRECTORY = 'apache-opennlp'
    
    parser = optparse.OptionParser(usage="%prog [OPTIONS]")
    parser.add_option('-p', '--port', type="int", default=8080,
                      help="Port to bind to [8080]")
    parser.add_option('--path', default=DIRECTORY,
                      help="Path to OpenNLP install [%s]" % DIRECTORY)
    options, args = parser.parse_args()

    addr = ('localhost', options.port)
    uri = 'http://%s:%s' % addr


    server = SimpleJSONRPCServer(addr)

    print("Starting OpenNLP")
    nlp = OpenNLPServer(options.path)
    server.register_function(nlp.parse)

    print("Serving on %s" % uri)
    server.serve_forever()

def tag(string):
    mp.set_start_method('spawn')
    p = mp.Process(target=start_server)
    p.start()
    p.join()
    nlp = OpenNLP()
    results = nlp.parse(string)
    print(results)
