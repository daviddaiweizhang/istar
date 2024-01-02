import argparse
from utils import load_tsv, write_lines, read_lines


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpfile', type=str, help='e.g. demo/cnts.tsv')
    parser.add_argument(
            'outfile', type=str, help='e.g. demo/gene-names.txt')
    parser.add_argument('--n-top', type=int, default=None, help='e.g. 50')
    parser.add_argument(
            '--extra', type=str, default=None,
            help='demo/marker-genes.txt')
    args = parser.parse_args()
    return args


def main():

    args = get_args()

    cnts = load_tsv(args.inpfile)
    order = cnts.var().to_numpy().argsort()[::-1]
    names = cnts.columns.to_list()
    names_all = [names[i] for i in order]

    names_top = names_all
    if args.n_top is not None:
        names_top = names_top[:args.n_top]

    if args.extra is None:
        names_extra = []
    else:
        names_extra = read_lines(args.extra)
        names_extra = [
                name for name in names_extra
                if (name in names_all) and (name not in names_top)]

    names = names_extra + names_top

    write_lines(names, args.outfile)


if __name__ == '__main__':
    main()
