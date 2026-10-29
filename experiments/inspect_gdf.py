"""Inspect BCIC IV 2a training GDF metadata; never filter, delete trials or train models."""
import argparse
from bisect import bisect_right
from collections import Counter
from contextlib import redirect_stdout
import io
from pathlib import Path
import platform

CLASS_NAMES = {'769': '左手', '770': '右手', '771': '双脚', '772': '舌头'}
EOG_NAMES = ('EOG-left', 'EOG-central', 'EOG-right')


def analyze_events(events, event_id, end_sample):
    """Associate artifact and cue samples with half-open trial intervals.

    Uses the returned event dictionary, not hard-coded MNE event integers.
    Returns anomalies instead of silently treating ambiguous trials as clean.
    """
    reverse = {v: k for k, v in event_id.items()}
    starts = sorted(int(row[0]) for row in events
                    if reverse.get(int(row[2])) == '768')
    if not starts or len(starts) != len(set(starts)):
        raise ValueError('试次开始标记缺失或重复，无法可靠对应试次。')
    cues = [[] for _ in starts]
    artifacts = Counter()
    anomalies = []
    for sample, _, value in events:
        sample = int(sample)
        code = reverse.get(int(value))
        if code not in CLASS_NAMES and code != '1023':
            continue
        idx = bisect_right(starts, sample) - 1
        if idx < 0 or sample >= end_sample:
            anomalies.append(f'标记 {code}（sample={sample}）位于试次范围之外')
            continue
        if code == '1023':
            artifacts[idx] += 1
        else:
            cues[idx].append(code)
    counts = Counter()
    artifact_counts = Counter()
    rows = []
    for idx, trial_cues in enumerate(cues):
        code = trial_cues[0] if len(trial_cues) == 1 else None
        if code is None:
            anomalies.append(f'试次 {idx + 1} 包含 {len(trial_cues)} 个已知类别提示')
        else:
            counts[code] += 1
            if artifacts[idx]:
                artifact_counts[code] += 1
        if artifacts[idx]:
            rows.append((idx + 1, code, artifacts[idx]))
    return starts, counts, artifact_counts, rows, anomalies


def inspect(path):
    import mne
    import numpy as np
    raw = mne.io.read_raw_gdf(str(path), preload=False, verbose='ERROR')
    missing = [name for name in EOG_NAMES if name not in raw.ch_names]
    if missing:
        raise ValueError(f'未找到预期的眼电通道：{missing}。请先核对数据集和通道。')
    raw.set_channel_types({name: 'eog' for name in EOG_NAMES}, verbose='ERROR')
    events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
    if '768' not in event_id or not all(c in event_id for c in CLASS_NAMES):
        raise ValueError('缺少训练集试次或四类提示标记；此脚本不支持直接分析无类别标签的E文件。')
    starts, counts, bad_counts, rows, anomalies = analyze_events(
        events, event_id, raw.first_samp + raw.n_times)
    print('BCIC IV 2a 训练数据检查（只读，不训练模型）')
    print(f'文件名：{path.name}')
    print(f'Python: {platform.python_version()} | MNE: {mne.__version__} | NumPy: {np.__version__}')
    print(f'采样率：{raw.info["sfreq"]} Hz')
    print(f'总采样点：{raw.n_times}')
    print(f'记录跨度：{(raw.n_times - 1) / raw.info["sfreq"]:.3f} s')
    print('\n通道（顺序从1开始；EEG编号不等于头皮位置）：')
    for i, (name, kind) in enumerate(zip(raw.ch_names, raw.get_channel_types()), 1):
        print(f'{i:02d}\t{name}\t{kind}')
    eeg = [name for name, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == 'eeg']
    print(f'按类型选择EEG时将选中 {len(eeg)} 个通道；EOG共 {len(EOG_NAMES)} 个')
    print('注意：这只核查选择规则；团队模型实际输入仍需查看基线的X shape。')
    print('\n事件对应表：', event_id)
    print('\n各类原始标记数量：')
    for code, count in sorted(Counter(raw.annotations.description).items()):
        print(f'{code}: {count}')
    print(f'\n试次开始数量：{len(starts)}')
    print('含伪迹标记的试次（按相邻试次开始标记划分）：')
    for trial, code, number in rows:
        print(f'试次 {trial}：{CLASS_NAMES.get(code, "类别异常")}；1023标记数={number}')
    print('\n类别\t总试次\t有标记\t无标记')
    for code, name in CLASS_NAMES.items():
        print(f'{name}\t{counts[code]}\t{bad_counts[code]}\t{counts[code] - bad_counts[code]}')
    print(f'对应到 {len(rows)} 个不同的含标记试次')
    print('\n异常检查：')
    if anomalies:
        for item in anomalies:
            print('需核对：' + item)
    else:
        print('未发现试次类别缺失/重复或标记落在试次范围之外。')
    print('\n无标记不等于没有伪迹；本脚本不删除任何试次、不滤波、不生成分类成绩。')
    raw.close()
    return bool(anomalies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gdf', type=Path, help='训练GDF路径，例如 data/A01T.gdf')
    parser.add_argument('--output', type=Path, help='可选：将检查结果保存为UTF-8文本')
    args = parser.parse_args()
    if not args.gdf.is_file():
        parser.error(f'找不到文件：{args.gdf}')
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        anomalies = inspect(args.gdf)
    result = buffer.getvalue()
    print(result, end='')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding='utf-8')
        print(f'检查记录已保存：{args.output}')
    return 2 if anomalies else 0


if __name__ == '__main__':
    raise SystemExit(main())
