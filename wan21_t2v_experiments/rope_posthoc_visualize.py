"""Render paired attention overviews and matched-region diagnostic figures on CPU."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def render_pair(task: dict) -> list[dict]:
    """Render two rows of selected frames, with shared scales and audited outlines.

    Input arrays are [2,K,H,W] head-mean probabilities and Boolean supports.
    Yellow repeats the actual query support at every key frame; green outlines
    the reviewed final object support. Neither outline estimates attention peaks.
    Return paths and normalization limits for a reproducible figure index.
    """
    records = []
    for conditional in (False, True):
        arrays = task['attention']
        if conditional:
            sums = arrays.sum((-1, -2), keepdims=True)
            arrays = np.divide(arrays, sums, out=np.zeros_like(arrays), where=sums > 0)
        vmax = max(float(arrays.max()), 1e-12)
        fig, axes = plt.subplots(2, arrays.shape[1], figsize=(22, 5.8), squeeze=False)
        fig.subplots_adjust(left=.008, right=.992, top=.72, bottom=.13, wspace=.025, hspace=.65)
        for row, name in enumerate(('baseline', 'modified')):
            for col in range(arrays.shape[1]):
                axis = axes[row, col]
                axis.imshow(arrays[row, col], cmap='viridis', alpha=.95, vmin=0, vmax=vmax)
                if task['object_valid'][row, col]:
                    region = task['object_masks'][row, col]
                    if region.any() and not region.all():
                        axis.contour(region, levels=[.5], colors=['#00d050'], linewidths=1.8)
                region = task['query_masks'][row]
                if region.any() and not region.all():
                    axis.contour(region, levels=[.5], colors=['#ffff00'], linewidths=1.8, linestyles='dashed')
                axis.axis('off')
            fig.text(.5, axes[row, 0].get_position().y0 - .045, name, ha='center', va='top', fontsize=13)
        mode = 'Within-key-frame conditional probability' if conditional else 'Global attention probability'
        fig.suptitle(f"Seed {task['seed']} | Step {task['step']} | Layer {task['layer']} | Head mean\n{mode}; query: first-frame basketball region", y=.975, fontsize=14)
        fig.text(.5, .845, 'Green: reviewed final object region    Yellow dashed: first-frame query coordinates projected to every frame', ha='center', fontsize=11)
        suffix = '_conditional' if conditional else ''
        path = Path(task['output']) / f"step_{task['step']:02d}_layer_{task['layer']:02d}_headmean{suffix}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        records.append(dict(step=task['step'], layer=task['layer'], normalization='conditional' if conditional else 'global', path=str(path), vmin=0., vmax=vmax))
    return records


def render_overviews(root: Path, seed: int, step_end: int, layers: list[int], workers: int) -> None:
    """Render every requested step/layer from saved maps using the standard ten frames."""
    from .utils import _resolve_wan21_t2v_viz_frame_indices
    output = root / 'comparison' / f'seed_{seed}'
    output.mkdir(parents=True, exist_ok=True)
    conditions, object_masks, query_masks, valid = [], [], [], []
    keys = [(step, layer, 0) for step in range(1, step_end + 1) for layer in layers]
    for condition in ('baseline', 'modified'):
        directory = root / condition / f'seed_{seed}'
        maps = torch.load(directory / 'self_attention/self_attention_viz_maps.pt', map_location='cpu', weights_only=False, mmap=True)
        if not set(keys).issubset(maps):
            raise ValueError(f'Missing requested maps in {directory}')
        conditions.append(maps)
        audit = directory / 'region_audit_key_q95'
        support = torch.load(audit / 'reference_support.pt', map_location='cpu', weights_only=False)
        object_masks.append(support['support_mask_fhw'].numpy())
        query = torch.load(directory / 'region_audit/reference_support.pt', map_location='cpu', weights_only=False)
        query_masks.append(query['support_mask_fhw'][0].numpy())
        reviewed = json.loads((audit / 'region_audit.json').read_text())['frames']
        valid.append({row['latent_frame']: row.get('valid_for_object_metrics') is True for row in reviewed})
    count = len(object_masks[0])
    indices, video_labels = _resolve_wan21_t2v_viz_frame_indices(count, 81, 10)
    tasks = []
    for step, layer, query_frame in keys:
        arrays = np.stack([maps[(step, layer, query_frame)].float().mean(0)[indices].numpy() for maps in conditions])
        tasks.append(dict(seed=seed, step=step, layer=layer, output=str(output), attention=arrays,
                          object_masks=np.stack([mask[indices] for mask in object_masks]),
                          query_masks=np.stack(query_masks), object_valid=np.asarray([[reviews[i] for i in indices] for reviews in valid])))
    records = []
    if workers == 1:
        iterator = map(render_pair, tasks)
        for task_records in iterator:
            records.extend(task_records)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            for index, task_records in enumerate(pool.map(render_pair, tasks, chunksize=2), 1):
                records.extend(task_records)
                if index % 50 == 0:
                    print(f'seed {seed}: {index}/{len(tasks)} step/layer pairs rendered', flush=True)
    with (output / 'overview_index.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (output / 'overview_metadata.json').write_text(json.dumps(dict(seed=seed, steps=list(range(1,step_end+1)), layers=layers,
        latent_frame_indices=indices, video_frame_labels=video_labels, query_frame_included_for_visual_reference=True,
        normalization='Both global probabilities and within-key-frame conditional probabilities are saved.',
        shared_scale='Baseline and modified share limits within each figure; limits differ between figures and are recorded in overview_index.csv.',
        colors=dict(object='green #00d050', same_coordinate_reference='yellow #ffff00 dashed', attention='viridis'),
        object_outline_policy='Only explicitly reviewed object regions are outlined; invalid regions are omitted.',
        query_coordinates='The actual first-frame query support copied to each key frame, not a measured attention hotspot.'), indent=2)+'\n')
    if step_end >= 5 and 27 in layers:
        shutil.copyfile(output / 'step_05_layer_27_headmean_conditional.png', root / f'comparison_overview_seed_{seed}.png')


def load_paired_frame_metrics(output: Path, step_end: int) -> list[dict]:
    """Read head-mean exclusive contrasts, retaining identical finite frame pairs.

    The per-frame CSV already excludes unaudited keys and the query frame. Pairing
    additionally rejects either empty exclusive region in either condition.
    """
    paired = {}
    with (output / 'per_frame_metrics.csv').open() as handle:
        for row in csv.DictReader(handle):
            if row['head'] != 'mean' or int(row['step']) > step_end:
                continue
            key = tuple(int(row[field]) for field in ('step', 'layer', 'key_token_frame'))
            paired.setdefault(key, {})[row['condition']] = float(row['exclusive_density_contrast'])
    rows = []
    for (step, layer, frame), values in sorted(paired.items()):
        if all(name in values and np.isfinite(values[name]) for name in ('baseline', 'modified')):
            rows.append(dict(step=step, layer=layer, key_token_frame=frame, baseline=values['baseline'], modified=values['modified'], change=values['modified']-values['baseline']))
    return rows


def matrix_triptych(path: Path, baseline: np.ndarray, modified: np.ndarray, xlabels: list[int], xlabel: str, title: str) -> None:
    """Show baseline, modified and paired change with centered, explicit metric scales."""
    delta = modified - baseline
    finite = np.concatenate([baseline[np.isfinite(baseline)], modified[np.isfinite(modified)]])
    limit = max(float(np.abs(finite).max()), 1e-12) if len(finite) else 1.
    change = delta[np.isfinite(delta)]
    delta_limit = max(float(np.abs(change).max()), 1e-12) if len(change) else 1.
    cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#d9d9d9')
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for axis, array, label, cap in zip(axes, (baseline, modified, delta), ('baseline', 'modified', 'modified − baseline'), (limit, limit, delta_limit)):
        rendered = axis.imshow(array, cmap=cmap, vmin=-cap, vmax=cap, origin='upper', aspect='auto', interpolation='nearest')
        stride = max(1, len(xlabels)//10)
        ticks = list(range(0,len(xlabels),stride))
        axis.set_xticks(ticks, [xlabels[i] for i in ticks])
        axis.set_yticks(range(array.shape[0]), range(1,array.shape[0]+1))
        axis.set(xlabel=xlabel, ylabel='Denoising step', title=label)
        fig.colorbar(rendered, ax=axis, shrink=.8, label='Exclusive conditional density contrast' if label != 'modified − baseline' else 'Change in contrast')
    fig.suptitle(title+'\nContrast > 0: object-only preference; contrast < 0: same-coordinate-only preference\nPositive change: shift toward object relative to same-coordinate region; grey: no defined paired comparison', fontsize=11)
    fig.tight_layout(rect=(0,0,1,.88))
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_metric_diagnostics(output: Path, seed: int, step_end: int, layers: list[int]) -> None:
    """Plot matched step/layer changes, per-key-frame matrices and all-layer curves."""
    from .rope_posthoc_compare import plot_trends, write_csv
    all_rows = load_paired_frame_metrics(output, 50)
    rows = [row for row in all_rows if row["step"] <= step_end and row["layer"] in layers]
    write_csv(output / 'paired_alignment_frames.csv', rows)
    cells = {}
    for row in rows:
        cells.setdefault((row['step'],row['layer']), []).append(row)
    aggregates = [dict(step=step, layer=layer, paired_frame_count=len(group), **{name:float(np.mean([r[name] for r in group])) for name in ('baseline','modified','change')}) for (step,layer),group in sorted(cells.items())]
    write_csv(output / 'paired_alignment_step_layer.csv', aggregates)
    arrays = {name:np.full((step_end,len(layers)), np.nan) for name in ('baseline','modified')}
    for row in aggregates:
        if row['layer'] in layers:
            for name in arrays:
                arrays[name][row['step']-1,layers.index(row['layer'])] = row[name]
    matrix_triptych(output/'alignment_step_layer.png',arrays['baseline'],arrays['modified'],layers,'DiT layer (zero-based)',f'Seed {seed}: mean paired cross-frame alignment, head mean')
    matrix_triptych(output/'alignment_first5_steps.png',arrays['baseline'][:5],arrays['modified'][:5],layers,'DiT layer (zero-based)',f'Seed {seed}: early motion-planning alignment, head mean')
    counts = np.zeros_like(arrays['baseline'])
    for row in aggregates:
        if row['layer'] in layers: counts[row['step']-1,layers.index(row['layer'])] = row['paired_frame_count']
    fig,axis=plt.subplots(figsize=(10,6)); im=axis.imshow(counts,origin='upper',aspect='auto',vmin=0,interpolation='nearest',cmap='viridis')
    axis.set(xlabel='DiT layer (zero-based)',ylabel='Denoising step',title=f'Seed {seed}: number of identical, defined key frames per comparison')
    axis.set_yticks(range(step_end),range(1,step_end+1));axis.set_xticks(range(0,len(layers),2),layers[::2]);fig.colorbar(im,ax=axis,label='Paired key frames')
    fig.tight_layout();fig.savefig(output/'alignment_valid_frame_counts.png',dpi=150);plt.close(fig)
    fig,axis=plt.subplots(figsize=(7,7))
    for early,color,label in ((True,'#c75000','Steps 1–5'),(False,'#1479b8',f'Steps 6–{step_end}')):
        chosen=[row for row in aggregates if (row['step']<=5)==early]
        axis.scatter([r['baseline'] for r in chosen],[r['modified'] for r in chosen],s=18,alpha=.6,color=color,label=label)
    values=[r[name] for r in aggregates for name in ('baseline','modified')]
    low,high=min(values+[0.]),max(values+[0.]);padding=max((high-low)*.06,1e-6)
    axis.plot([low-padding,high+padding],[low-padding,high+padding],'k--',linewidth=1,label='No change')
    axis.axhline(0,color='grey',linewidth=.6);axis.axvline(0,color='grey',linewidth=.6)
    axis.set(xlim=(low-padding,high+padding),ylim=(low-padding,high+padding),xlabel='Baseline: paired exclusive density contrast',ylabel='Modified: paired exclusive density contrast',title=f'Seed {seed}: one point per step/layer\nAbove diagonal = shift toward object relative to same coordinates')
    axis.set_aspect('equal');axis.legend();axis.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'alignment_paired_scatter.png',dpi=160);plt.close(fig)
    all_frames=list(range(1,21))
    for layer in layers:
        frame_arrays={name:np.full((step_end,len(all_frames)),np.nan) for name in ('baseline','modified')}
        for row in rows:
            if row['layer']==layer:
                for name in frame_arrays:frame_arrays[name][row['step']-1,row['key_token_frame']-1]=row[name]
        matrix_triptych(output/f'alignment_key_frames_layer_{layer:02d}.png',frame_arrays['baseline'],frame_arrays['modified'],all_frames,'Key latent frame (query frame 0 excluded)',f'Seed {seed}, layer {layer}: individual paired key-frame alignment')
    with (output/'per_head_summary.csv').open() as handle:
        summaries=[]
        for row in csv.DictReader(handle):
            if row['head']!='mean':continue
            for field in ('step','layer'):row[field]=int(row[field])
            for field in ('sum_object_global_mass','mean_object_conditional_mass','mean_exclusive_density_contrast'):row[field]=float(row[field])
            summaries.append(row)
    for layer in layers:
        plot_trends(output,summaries,layer,save_name=f'step_trends_layer_{layer:02d}.png',paired_rows=all_rows)
    # Keep the established layer-27 entry, now with exact pairing for contrast.
    if 27 in layers:shutil.copyfile(output/'step_trends_layer_27.png',output/'step_trends.png')
    (output/'alignment_metrics.json').write_text(json.dumps(dict(seed=seed,steps=list(range(1,step_end+1)),layers=layers,
        metric='Conditional density in object-only region minus conditional density in query-coordinate-only region.',
        positive_value='Object-only region receives more attention per token.',positive_change='Relative object preference increases; a still-negative value can remain coordinate-preferring.',
        pairing='Identical reviewed nonquery key frames with finite contrasts in both conditions at each step/layer.',
        undefined='Empty exclusive supports are undefined, excluded from averages and grey in matrices; counts are supplied.',
        head_aggregation='Mean probability map across heads before frame conditioning.',
        curve_scope='All three curves cover all 50 saved steps; contrast uses identical finite frame pairs.',
        sample_count=len(aggregates),increasing_step_layer_count=sum(r['change']>0 for r in aggregates),
        caution='Descriptive paired differences, not significance tests; conditions have different trajectories and query supports.'),indent=2)+'\n')


def main() -> None:
    """Run saved-data visualization for requested seeds without allocating a model."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[8,20,29])
    parser.add_argument('--step-end',type=int,default=20)
    parser.add_argument('--layers',type=int,nargs='+',default=list(range(30)))
    parser.add_argument('--workers',type=int,default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--metrics-only',action='store_true')
    mode.add_argument('--overviews-only',action='store_true')
    args=parser.parse_args()
    if not 1 <= args.step_end <= 50 or args.workers < 1:
        parser.error('step-end must be in [1, 50] and workers must be positive')
    if len(set(args.layers)) != len(args.layers) or any(layer not in range(30) for layer in args.layers):
        parser.error('layers must be distinct indices in [0, 29]')
    torch.set_num_threads(2)
    for seed in args.seeds:
        if not args.metrics_only:render_overviews(args.root,seed,args.step_end,args.layers,args.workers)
        if not args.overviews_only:plot_metric_diagnostics(args.root/'comparison'/f'seed_{seed}',seed,args.step_end,args.layers)
        print(f'Completed seed {seed}',flush=True)


if __name__=='__main__':main()
