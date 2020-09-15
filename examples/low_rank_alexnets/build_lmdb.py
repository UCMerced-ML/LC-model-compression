import numpy as np
from tensorpack.dataflow import LMDBSerializer, dataset, MultiProcessRunnerZMQ
import argparse
import os

class BinaryILSVRC12(dataset.ILSVRC12Files):
    def __iter__(self):
        for fname, label in super(BinaryILSVRC12, self).__iter__():
            with open(fname, 'rb') as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            yield [jpeg, label]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch LC RankSelection Training')

    # --------------------
    parser.add_argument('--imagenet_folder')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lmdb_file', type=str)

    args = parser.parse_args()

    if args.val and args.train:
        print("Train and Validation options are mutually exclusive! Chose only one.")


    if args.val:
        print("We are generating the lmdb file containing validation images of imagenet.")
        print(f"The file will be saved at {args.lmdb_file}.lmdb")

        ds0 = BinaryILSVRC12(os.path.expanduser(args.imagenet_folder), 'val')
        ds1 = MultiProcessRunnerZMQ(ds0, num_proc=1)
        LMDBSerializer.save(ds1, f"{os.path.expanduser(args.lmdb_file)}.lmdb")
    elif args.train:
        print("We are generating the lmdb file containing training images of imagenet.")
        print(f"The file will be saved at {args.lmdb_file}.lmdb")

        ds0 = BinaryILSVRC12(os.path.expanduser(args.imagenet_folder), 'train')
        ds1 = MultiProcessRunnerZMQ(ds0, num_proc=1)
        LMDBSerializer.save(ds1, f"{os.path.expanduser(args.lmdb_file)}.lmdb")