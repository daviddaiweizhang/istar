import sys
import pandas as pd
from utils import read_lines, load_tsv, save_tsv


def main():
    prefix = sys.argv[1]
    gene_names = read_lines(prefix + 'gene-names.txt')
    dfs = {
            gn: load_tsv(f'{prefix}cnts-clustered/by-genes/{gn}.tsv')
            for gn in gene_names}
    columns = dfs[list(dfs.keys())[0]].columns
    results = {}
    for col in columns:
        results[col] = pd.DataFrame(
                {gn: df[col] for gn, df in dfs.items()})
    results['sum'] = (
            (results['mean'] * results['count'])
            .round().astype(int))
    for key, val in results.items():
        save_tsv(val, f'{prefix}cnts-clustered/by-clusters/{key}.tsv')


if __name__ == '__main__':
    main()
