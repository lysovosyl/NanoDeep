import os
import argparse
import ont_fast5_api.multi_fast5 as ml
import sys

def main():
    parser = argparse.ArgumentParser(description="you should add those parameter")
    parser.add_argument('-i', required=True, type=str, help='fast5 data dir')
    parser.add_argument('-o', default=None, type=str, help='save id file')
    opt = parser.parse_args()

    input_path = opt.i
    save_file = opt.o
    if save_file == None:
        for file in os.listdir(input_path):
            a = ml.MultiFast5File(os.path.join(input_path,file))
            ids = a.get_read_ids()
            for id in ids:
                print(id)
    else:
        f = open(save_file, 'w')
        for file in os.listdir(input_path):
            a = ml.MultiFast5File(os.path.join(input_path, file))
            ids = a.get_read_ids()
            for id in ids:
                f.writelines(id + '\n')
        f.close()


if __name__ == "__main__":
    main()

