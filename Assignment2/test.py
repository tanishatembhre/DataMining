# Press the green button in the gutter to run the script.
from clustering import slice_cluster_brain
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--inputfolder", default="testPatient",
                        help="input folder")
    parser.add_argument("-o", "--outputfolder", default=".",
                        help="output folder")
    args = parser.parse_args()

    input_folder = args.inputfolder
    output_folder = args.outputfolder
    slice_cluster_brain(input_folder, output_folder)
