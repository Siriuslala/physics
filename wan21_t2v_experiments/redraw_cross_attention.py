"""Materialize selected-token cross-attention PDFs from saved maps on CPU."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import torch

from .utils import (
    _normalize_wan21_t2v_attention_map_per_frame,
    _resolve_wan21_t2v_viz_frame_indices,
    _sanitize_wan21_t2v_token_name,
    _save_wan21_t2v_cross_attention_pdf,
)


def render_bundle(task: tuple) -> list[str]:
    """Render all heads and their probability-map mean for one step/layer/token.

    Input contains a CPU [heads,frames,height,width] tensor and collection display
    settings. Reuse the canonical cross-attention renderer and directory layout.
    Return every generated PDF path; raw maps and collection metadata stay intact.
    """
    directory, key, maps, settings = task
    torch.set_num_threads(1)
    step, layer, word = key
    token_type = settings['token_types'][word]
    output = Path(directory) / f'timestep_{step:03d}' / f'token_{token_type}_{_sanitize_wan21_t2v_token_name(word)}'
    output.mkdir(parents=True, exist_ok=True)
    indices, labels = _resolve_wan21_t2v_viz_frame_indices(
        maps.shape[1], settings['frame_num'], settings['num_viz_frames'], settings.get('viz_frame_indices'))
    paths = []
    for head, attention in [(f'{h:02d}', maps[h]) for h in range(maps.shape[0])] + [('mean', maps.float().mean(0))]:
        if settings.get('attention_pdf_per_frame_normalize', False):
            attention = _normalize_wan21_t2v_attention_map_per_frame(attention)
        path = output / f'layer_{layer:02d}_head_{head}.pdf'
        _save_wan21_t2v_cross_attention_pdf(attention, indices, labels, str(path),
            f'step={step} layer={layer} head={head} token={word}',
            share_color_scale=settings.get('attention_pdf_share_color_scale', False))
        paths.append(str(path))
    return paths


def main() -> None:
    """Filter saved tokens before rendering, preserving original collection provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--tokens', nargs='+', default=['basketball'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--num-viz-frames', type=int, default=None)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    torch.set_num_threads(1)
    settings = json.loads((args.directory/'cross_attention_token_viz_summary.json').read_text())
    if args.num_viz_frames is not None:
        if args.num_viz_frames < 1:
            parser.error('num-viz-frames must be positive')
        settings['num_viz_frames'] = args.num_viz_frames
        settings['viz_frame_indices'] = None
    maps = torch.load(args.directory/'cross_attention_maps.pt', map_location='cpu', mmap=True, weights_only=False)
    keys = sorted(key for key in maps if key[2] in args.tokens)
    if {key[2] for key in keys} != set(args.tokens):
        raise ValueError('One or more selected tokens are absent from saved maps')
    tasks = ((str(args.directory), key, maps[key], settings) for key in keys)
    paths = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        for index, result in enumerate(pool.map(render_bundle, tasks, chunksize=1), 1):
            paths.extend(result)
            if index % 50 == 0 or index == len(keys):
                print(f'{index}/{len(keys)} bundles; {len(paths)} PDFs', flush=True)
    for path in paths:
        with open(path, 'rb') as handle:
            if handle.read(5) != b'%PDF-':
                raise ValueError(f'Invalid PDF: {path}')
    report = dict(tokens=args.tokens, map_groups=len(keys), pdf_count=len(paths),
        steps=sorted({key[0] for key in keys}), layers=sorted({key[1] for key in keys}),
        source=str(args.directory/'cross_attention_maps.pt'), cpu_only=True,
        display_settings={key:settings.get(key) for key in ('num_viz_frames','viz_frame_indices','attention_pdf_per_frame_normalize','attention_pdf_share_color_scale')},
        paths=paths)
    name = '_'.join(_sanitize_wan21_t2v_token_name(token) for token in args.tokens)
    (args.directory/f'cross_attention_redraw_{name}.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Completed and verified PDF headers', flush=True)


if __name__ == '__main__':
    main()
