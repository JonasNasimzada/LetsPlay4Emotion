import argparse
import os
import time

import torch.multiprocessing as mp

import run_flame_apply_hifi3d_uv

refer_mesh_path = '/FLAME_w_HIFI3D_UV_V2.obj'
uvmap_name = "/person_0_uv.png"


def apply_uv(chunk_data):
    flame_mesh_path = chunk_data
    save_mtl_path = f'{flame_mesh_path[:-4]}_w_HIFI3D_UV.mtl'

    refer_data = run_flame_apply_hifi3d_uv.read_mesh_obj(refer_mesh_path)
    flame_data = run_flame_apply_hifi3d_uv.read_mesh_obj(flame_mesh_path)

    flame_data['vt'] = refer_data['vt']
    flame_data['fvt'] = refer_data['fvt']
    flame_data['mtl_name'] = os.path.basename(save_mtl_path)

    run_flame_apply_hifi3d_uv.write_mesh_obj(flame_data, flame_mesh_path)


def worker(queue):
    while True:
        chunk_data = queue.get()
        if chunk_data is None:
            break
        apply_uv(chunk_data)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str)
    parser.add_argument('--thread_num', type=int, default=1, help="number of threads")

    args = parser.parse_args()

    num_processes = args.thread_num

    obj_files = [filename for filename in os.listdir(args.input_directory) if filename.endswith(".obj")]

    queue = mp.Queue()
    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=worker, args=(queue,))
        processes.append(p)
        p.start()

    chunk_id = 0
    for obj in obj_files:
        chunk_data = (args.input_directory, obj)
        queue.put(chunk_data)
        chunk_id += 1

    for _ in range(num_processes):
        queue.put(None)

    for p in processes:
        p.join()


if __name__ == '__main__':
    tic = time.time()
    mp.set_start_method('spawn')
    main()
    toc = time.time()
    print(f'coping meshes done, took {toc - tic} seconds.')
