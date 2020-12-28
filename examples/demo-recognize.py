import _init_path
from models.conv import GatedConv
import beamdecode

model = GatedConv.load("../pretrained/gated-conv.pth")

while 1:
    text = beamdecode.predict("test/123.wav")

    print("")
    print("识别结果:")
    print(text)
