import argparse
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path

import tomlkit
from minio import Minio, S3Error


def main(cfg_path: str):
    cfg_path = Path(cfg_path)
    cfg = tomlkit.loads(cfg_path.read_text('utf-8'))
    client = Minio(**cfg['minio']['client'])

    object_names = set()
    for object_name, detect in cfg['detect'].items():
        if detect[0] == '无效' and detect[1] == '无效':
            continue
        if not detect[2]:
            continue

        object_names.add(object_name)

    patients = len(list(client.list_objects('total')))
    total = len([_ for _ in client.list_objects('total', recursive=True)
                 if not _.is_dir and _.object_name.endswith('.nii.gz') and _.size > 0])
    print(f'\n[{total}/{len(object_names)}] {patients} patients')

    for object_name in object_names:
        try:
            if client.stat_object('total', object_name).size > 0:
                continue
        except S3Error:
            pass

        total += 1
        print(f'\n[{total}/{len(object_names)}]', object_name)

        with tempfile.TemporaryDirectory() as tdir:
            image = Path(tdir) / 'image.nii.gz'
            label = Path(tdir) / 'label.nii.gz'

            client.fget_object('nii', object_name, image.as_posix())

            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(seg, image.as_posix(), label.as_posix())
                try:
                    future.result()
                except Exception as e:
                    warnings.warn(str(e))

            if label.exists():
                print(f'[{total}/{len(object_names)}]', object_name, 'done')
                client.fput_object('total', object_name, label.as_posix())
            else:
                print(f'[{total}/{len(object_names)}]', object_name, 'error')
                client.put_object('total', object_name, BytesIO(b''), 0)


def seg(image_path: str, label_path: str):
    from totalsegmentator.python_api import totalsegmentator
    totalsegmentator(image_path, label_path, True, task='total', quiet=True)


# 数据超过 (512,512,1000) Windows 平台报错 OSError: [WinError 87]
# 修改源码 https://github.com/wasserth/TotalSegmentator/issues/533
# def predict_from_data_iterator(self,
#                                data_iterator,
#                                save_probabilities: bool = False,
#                                num_processes_segmentation_export: int = default_num_processes):
#     """
#     each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
#     If 'ofile' is None, the result will be returned instead of written to a file
#     """
#     r = []
#     for preprocessed in data_iterator:
#         data = preprocessed['data']
#         if isinstance(data, str):
#             delfile = data
#             data = torch.from_numpy(np.load(data))
#             os.remove(delfile)
#
#         ofile = preprocessed['ofile']
#
#         properties = preprocessed['data_properties']
#
#         prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()
#
#         if ofile is not None:
#             result = export_prediction_from_logits(
#                 prediction, properties, self.configuration_manager, self.plans_manager,
#                 self.dataset_json, ofile, save_probabilities
#             )
#         else:
#             result = convert_predicted_logits_to_segmentation_with_correct_shape(
#                 prediction, self.plans_manager, self.configuration_manager, self.label_manager,
#                 properties, save_probabilities
#             )
#
#         r.append(result)
#
#     ret = [out for out in r]
#
#     if isinstance(data_iterator, MultiThreadedAugmenter):
#         data_iterator._finish()
#
#     # clear lru cache
#     compute_gaussian.cache_clear()
#     # clear device cache
#     empty_cache(self.device)
#     return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    while True:
        main(args.config)

        for _ in range(60, 0, -1):
            print(f'\rRetry in {_}s...', end='', flush=True)
            time.sleep(1)
