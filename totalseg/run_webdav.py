# PyTorch cuda
# pip install tomlkit webdavclient3 humanize TotalSegmentator

import argparse
import os
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

import humanize
import tomlkit
from totalsegmentator.python_api import totalsegmentator
from webdav3.client import Client

speed = [0.0, 0.0, '']


def progress(prefix, current, total):
    global speed
    speed[1] = time.perf_counter()
    _ = float(current) / (speed[1] - speed[0]) if speed[0] != speed[1] else 0.0
    speed[2] = f'{humanize.naturalsize(_, True)}/s'
    print(f'\r{prefix} {speed[2]} {100 * current / total:.2f}%', end='', flush=True)


def main(cfg_path: str, fs_path: str):
    global speed

    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))

    rw = Client(cfg['webdav']['labels'])

    fs_path = Path(fs_path)
    fs = tomlkit.loads(fs_path.read_text('utf-8'))

    todo = fs['todo']
    fs['done'] = fs.get('done', {})

    with tempfile.TemporaryDirectory() as tdir:
        for i, f in enumerate(todo):
            i += 1

            tdir = Path(tdir)
            image = tdir / f

            seconds = time.perf_counter()

            for task in cfg['task']:
                fs['done'][task] = fs['done'].get(task, [])

                if f in fs['done'][task]:
                    continue

                print(f'[进度] {i}/{len(todo)} {100 * i / len(todo):.3f}% {cfg_path.name} {fs_path.name}')
                print(f'[任务] {f} {task}')

                label = tdir / f'{task}.nii.gz'

                # 下载
                if not image.exists():
                    while True:
                        if subprocess.call([
                            sys.executable, 'run_webdav.py',
                            '--config', args.config,
                            '--download',
                            '--remote', f,
                            '--local', image.as_posix(),
                        ], cwd=os.getcwd()) == 0:
                            break
                        print('[下载] 重试')

                # 推理
                while True:
                    if subprocess.call([
                        sys.executable, 'run_webdav.py',
                        '--config', args.config,
                        '--inference',
                        '--task', task,
                        '--image', image.as_posix(),
                        '--label', label.as_posix(),
                    ], cwd=os.getcwd()) == 0:
                        break
                    print('[推理] 重试')

                # 上传
                while True:
                    if subprocess.call([
                        sys.executable, 'run_webdav.py',
                        '--config', args.config,
                        '--upload',
                        '--remote', f'{task}/{f}',
                        '--local', label.as_posix(),
                        '--task', task,
                    ], cwd=os.getcwd()) == 0:
                        break
                    print('[上传] 重试')

                fs['done'][task].append(f)
                fs_path.write_text(tomlkit.dumps(fs), 'utf-8')

                label.unlink(True)
            image.unlink(True)

            seconds = time.perf_counter() - seconds
            if seconds > 1:
                rw.mkdir('.progress')
                for _ in rw.list('.progress'):
                    if str(_).startswith(fs_path.name):
                        rw.clean(f'.progress/{_}')
                rw.upload_to(b'', f'.progress/{fs_path.name}_{i}_{len(todo)}_{100 * i / len(todo):.3f}%_{seconds:.0f}s')
                print(f'[用时] {seconds:.0f} 秒\n')


def download(cfg_path: str, remote: str, local: str):
    global speed

    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))

    ro = Client(cfg['webdav']['images'])

    name = f'[下载] {remote}'
    print(f'\r{name}', end='', flush=True)
    speed = [time.perf_counter()] * 2 + ['']
    t = time.perf_counter()
    ro.download_file(remote, local, partial(progress, name))
    t = time.perf_counter() - t
    print(f'\r{name} {speed[2]} {t:.2f} 秒', flush=True)


def inference(cfg_path: str, task: str, image: str, label: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    ln = cfg['task'][task].get('license_number')

    print(f'\r[推理] {task}', end='', flush=True)
    t = time.perf_counter()
    totalsegmentator(image, label, ml=True, task=task, license_number=ln, quiet=True)
    t = time.perf_counter() - t
    print(f'\r[推理] {task} {t:.2f} 秒', flush=True)


def upload(cfg_path: str, remote: str, local: str, task: str):
    global speed

    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))

    rw = Client(cfg['webdav']['labels'])

    rw.mkdir(task)

    name = f'[上传] {remote}'
    print(f'\r{name}', end='', flush=True)
    speed = [time.perf_counter()] * 2 + ['']
    t = time.perf_counter()
    rw.upload_file(remote, local, partial(progress, name))
    t = time.perf_counter() - t
    print(f'\r{name} {speed[2]} {t:.2f} 秒', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fs', default='')
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--remote', default='')
    parser.add_argument('--local', default='')
    parser.add_argument('--inference', action='store_true', default=False)
    parser.add_argument('--task', default='')
    parser.add_argument('--image', default='')
    parser.add_argument('--label', default='')
    parser.add_argument('--upload', action='store_true', default=False)
    args = parser.parse_args()

    if args.download:
        download(args.config, args.remote, args.local)
    elif args.inference:
        inference(args.config, args.task, args.image, args.label)
    elif args.upload:
        upload(args.config, args.remote, args.local, args.task)
    else:
        main(args.config, args.fs)
