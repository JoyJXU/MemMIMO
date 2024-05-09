import csv
import os
import matplotlib.pyplot as plt

# Class with utility functions


class utility:
    verbosity = 0
    gpu = False
    par = False

    def __init__(self, _verb=-1, _gpu=False, _par=False):
        if (0 <= _verb <= 2):
            utility.verbosity = _verb
        else:
            raise Exception("Verbosity level can take values 0, 1, or 2!")
        utility.gpu = _gpu
        utility.par = _par

    ############################################################

    # Level 1 debug messages
    def v_print_1(*args, **kwargs):
        if utility.verbosity >= 1:
            print(*args, **kwargs)

    ############################################################

    # Level 2 debug messages
    def v_print_2(*args, **kwargs):
        if utility.verbosity >= 2:
            print(*args, **kwargs)

    #############################################################

    # Calculate error between values of two vectors
    # Does not raise Exception
    def cal_error(reference, modeled):
        error = reference - modeled
        return error

    #############################################################

    def write_to_csv(path_to_file=None, file_name=None, data=None):
        if file_name == None:
            raise Exception("Write to CSV requires parameter \"file_name\"!")
        if data == None:
            raise Exception("Write to CSV requires parameter \"data\"!")
        if path_to_file == None:
            raise Exception("Write to CSV requires parameter \"data\"!")
        if not (os.path.isdir(path_to_file)):
            raise Exception("Path given not valid!")
        file = path_to_file+"/"+file_name
        utility.v_print_2("Path given = ", file)
        with open(file, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(data)
            csv_writer = None
            f.close()

    #############################################################

    def __str__(self):
        return (f"This is a utility class")


    #############################################################

    def plot_distribution(figs, vector, matrix, golden_model, cross, error, relative_error, rpd1=None, rpd2=None, rpd3=None, rpd4=None):
        if figs is not None:
            if rpd1 is not None or rpd2 is not None or rpd3 is not None or rpd4 is not None:
                [ax, bx, cx, dx, ex, fx, gx, hx, ix, jx] = figs
                ax.cla()
                bx.cla()
                cx.cla()
                dx.cla()
                ex.cla()
                fx.cla()
                gx.cla()
                hx.cla()
                ix.cla()
                jx.cla()

                ax.hist(vector.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                ax.set_title('vector')
                bx.hist(matrix.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                bx.set_title('matrix')
                cx.hist(golden_model.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                cx.set_title('golden model')
                dx.hist(cross.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                dx.set_title('crossbar')
                ex.hist(error.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                ex.set_xlim([-0.2, 0.2])
                ex.set_ylim([0, 1000])
                ex.set_title('error')
                fx.hist(relative_error.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                fx.set_title('relative error')
                gx.hist(rpd1.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                gx.set_title('rpd1 error')
                hx.hist(rpd2.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                hx.set_title('rpd2 error')
                ix.hist(rpd3.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                ix.set_title('rpd3 error')
                jx.hist(rpd4.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                jx.set_title('rpd4 error')

            else:
                [ax, bx, cx, dx, ex, fx] = figs
                ax.cla()
                bx.cla()
                cx.cla()
                dx.cla()
                ex.cla()
                fx.cla()

                ax.hist(vector.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                ax.set_title('vector')
                bx.hist(matrix.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                bx.set_title('matrix')
                cx.hist(golden_model.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                cx.set_title('golden model')
                dx.hist(cross.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                dx.set_title('crossbar')
                ex.hist(error.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                ex.set_xlim([-0.2, 0.2])
                ex.set_ylim([0, 1000])
                ex.set_title('error')
                fx.hist(relative_error.flatten().cpu(), bins=100, histtype="stepfilled", alpha=0.6)
                fx.set_title('relative error')

            file_name = "r" + str(vector.shape[1]) + "_c" + str(vector.shape[2])+".png"
            plt.savefig(file_name, dpi=300, bbox_inches='tight')

            plt.draw()
            plt.pause(0.05)
