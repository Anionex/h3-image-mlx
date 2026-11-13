"""Fetch 16 complete output rows from real BF16 projections, not whole checkpoints.

PYTHONPATH=src python tests/fetch_bf16_samples.py --output /tmp/h3-bf16-samples
H3_IMAGE_TEST_BF16_SAMPLES=/tmp/h3-bf16-samples python -m unittest discover -s tests
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

from h3_image_mlx.models import manifest_for


def read_range(url, start, length):
    request = urllib.request.Request(url + f'?sample_range={start}_{length}',
                                    headers={'Range':f'bytes={start}-{start+length-1}'})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
            raise RuntimeError('Server did not honor the range; refusing a full checkpoint download')
        data = response.read(length + 1)
        if len(data) != length:
            raise RuntimeError('Unexpected range length')
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', choices=('modelscope','huggingface'), default='modelscope')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    keys = [
        ('blocks.0.attn.qkv_proj.weight', 'blocks.0.mlp.fc1.weight'),
        ('model.layers.0.self_attn.q_proj.weight', 'model.layers.0.mlp.gate_proj.weight', 'visual.blocks.0.attn.qkv.weight'),
    ]
    records = []
    for i, (entry, names) in enumerate(zip(manifest_for('bf16')['files'], keys)):
        url = entry[args.source]
        length = struct.unpack('<Q', read_range(url, 0, 8))[0]
        if length > 4 << 20:
            raise ValueError('Unexpectedly large checkpoint header')
        original = json.loads(read_range(url, 8, length))
        header, chunks, offset = {}, [], 0
        for name in names:
            spec = original[name]
            assert spec['dtype'] == 'BF16' and len(spec['shape']) == 2
            shape = [16, spec['shape'][1]]
            data = read_range(url, 8 + length + spec['data_offsets'][0], shape[0]*shape[1]*2)
            header[name] = dict(dtype='BF16', shape=shape, data_offsets=[offset,offset+len(data)])
            chunks.append(data)
            offset += len(data)
        raw = json.dumps(header,separators=(',',':')).encode()
        raw += b' ' * (-len(raw) % 8)
        dest = args.output/f'component-{i}.safetensors'
        dest.write_bytes(struct.pack('<Q',len(raw))+raw+b''.join(chunks))
        records.append(dict(source=url, checkpoint_sha256=entry['sha256'], rows=16,
                            tensors=names, file=dest.name, sample_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        print(f'{dest}: {offset:,} weight bytes',flush=True)
    (args.output/'provenance.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__ == '__main__':
    main()
