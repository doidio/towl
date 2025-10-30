# PyTorch cuda
# pip install tomlkit webdavclient3 humanize TotalSegmentator

import argparse
import tempfile
import time
from functools import partial
from pathlib import Path

import humanize
import requests
import tomlkit
from totalsegmentator.python_api import totalsegmentator
from webdav3.client import Client

speed = [0.0, 0.0, '']


def progress(prefix, current, total):
    global speed
    speed[1] = time.perf_counter()
    _ = float(current) / (speed[1] - speed[0]) if speed[0] != speed[1] else 0.0
    speed[2] = f'{humanize.naturalsize(_, True)}/s'
    print(f'\r[{prefix}] {100 * current / total:.2f}% {speed[2]}', end='', flush=True)


def main(cfg_path: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))

    ro = Client(cfg['webdav']['images'])
    rw = Client(cfg['webdav']['labels'])
    rw.mkdir('.done')

    if undone := cfg.get('undone'):
        for _ in rw.list('.done'):
            if _ == undone:
                rw.clean(_ := f'.done/{_}')
                del cfg['undone']
                cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')
                print(f'[清理] {_}')
                return

    while len(rw_list := rw.list('.done')) < len(ro_list := ro.list()):
        print(f'[完成] {len(rw_list)}/{len(ro_list)} {100 * len(rw_list) / len(ro_list):.3f}%')

        for f in ro.list():
            done = f'.done/{f}'

            # 跳过完成
            if requests.head(f'{rw.webdav.hostname}/{done}', auth=(rw.webdav.login, rw.webdav.password)).ok:
                continue

            print(f'[新建] {f}')

            # 文件锁
            cfg['undone'] = f
            cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')
            rw.upload_to(b'', done)

            try:
                seconds = time.perf_counter()
                with tempfile.TemporaryDirectory() as tdir:
                    tdir = Path(tdir)
                    image = tdir / 'image.nii.gz'

                    # 下载
                    global speed
                    speed = [time.perf_counter()] * 2 + ['']
                    ro.download_file(f, image, partial(progress, _ := '下载'))
                    print(f'\r[{_}] {speed[2]}', flush=True)

                    # 推理
                    for task in cfg['task']:
                        label = tdir / f'{task}.nii.gz'
                        ln = cfg['task'][task].get('license_number')

                        print(f'[推理] {task}')
                        totalsegmentator(image, label, ml=True, task=task, license_number=ln, quiet=True)

                        # 上传
                        rw.mkdir(task)

                        speed = [time.perf_counter()] * 2 + ['']
                        rw.upload_file(_ := f'{task}/{f}', label, partial(progress, _ := '上传'))
                        print(f'\r[{_}] {speed[2]}', flush=True)

                del cfg['undone']
                cfg_path.write_text(tomlkit.dumps(cfg), 'utf-8')

                seconds = time.perf_counter() - seconds
                print(f'[用时] {seconds:.0f} 秒')

            except Exception as e:
                ro.clean(done)
                raise e

            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    main(args.config)
