import json,sys,re
cells=[]
for part in sorted(__import__('glob').glob('src_part*.py')):
    txt=open(part).read()
    for chunk in re.split(r'^# %%', txt, flags=re.M)[1:]:
        if chunk.startswith(' [markdown]'):
            body=chunk.split('\n',1)[1]
            lines=[l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.strip().split('\n')]
            cells.append({"cell_type":"markdown","metadata":{},"source":'\n'.join(lines)})
        else:
            body=chunk.split('\n',1)[1].strip('\n')
            cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":body})
nb={"cells":[{**c,"source":c["source"].splitlines(True)} for c in cells],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python"},"kaggle":{"accelerator":"nvidiaTeslaT4","isInternetEnabled":True}},
    "nbformat":4,"nbformat_minor":5}
json.dump(nb,open(sys.argv[1],'w'),indent=1)
print(len(cells),"cells")
for i,c in enumerate(cells):
    if c["cell_type"]=="code":
        compile(c["source"],f"cell{i}","exec")
print("all code cells compile")
