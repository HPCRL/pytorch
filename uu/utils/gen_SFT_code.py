import os

def rewriteRule(code):
    PATTERN = {"nn.Conv2d":"conv2d.TiledConv2d", 
            "torch.nn.Conv2d":"conv2d.TiledConv2d",
            "nn.MaxPool2d":"maxpool2d.cMaxPool2d"}
    for pstr, nstr in PATTERN.items():
        if pstr in code:
            print("code", code)
            code = code.replace(pstr, nstr)
            print("new code", code)
    
    return code



def rewriteCode(filename):
    with open(filename, "r") as file:
            lines = [line.rstrip() for line in file]
            network_start = lines.index("\"\"\"Network define START()\"\"\"")
            network_end = lines.index("\"\"\"Network define END()\"\"\"")
            print(network_start)
            print(network_end)
            
    f = open("gen_"+filename, "w")
    i = network_start
    while i <= network_end:
        code = rewriteRule(lines[i])
        f.write(code)
        f.write("\n")
        i+=1
    
    f.close()