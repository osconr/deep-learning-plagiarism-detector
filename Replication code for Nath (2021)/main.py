import sys
import getopt
#from mod2vocab200 import execute
from pan2021.siamese_mod import execute
#from random_baseline import execute_random_baseline
import warnings


def warn(*args, **kwargs):
    pass


if __name__ == '__main__':
    warnings.warn = warn
    warnings.filterwarnings("ignore", message=r"\[W108\]", category=UserWarning)
    inputFolder = ""
    outputFolder = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:o:")
    except getopt.GetoptError:
        print("main.py -c <inFolder> -o <outFolder>")  # tira command format
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-c":
            inputFolder = arg
        elif opt == "-o":
            outputFolder = arg
    assert len(inputFolder) > 0      # if not true, stop the process
    print("Input folder is", inputFolder)
    assert len(outputFolder) > 0
    print("Output folder is", outputFolder)
    #execute(inputFolder, outputFolder) #mod200
    execute(inputFolder, outputFolder, train_again=False, model_type ="GRU")
    #execute_random_baseline(inputFolder, outputFolder)
