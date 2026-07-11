import json
with open(r'G:\Meu Drive\QC-VORONOI\src\main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
print(f'Cells: {len(nb["cells"])}')
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    summary = src.replace('\n', ' ')[:100]
    print(f'  [{i:2d}] {c["cell_type"]:8s} | {summary}')
print('JSON válido!')
