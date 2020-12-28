import requests
import _init_path
import feature
from examples.record import record

server = "http://localhost:5000/recognize"

# record("record.wav", time=5)  # modify time to how long you want

# f = open("record.wav", "rb")

f = open("/home/db/bing/yuyingshibie/masr/examples/test/123.wav", "rb")

files = {"file": f}

while 1:
    r = requests.post(server, files=files)

    print("")
    print("识别结果:")
    print(r.text)
