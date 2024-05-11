# Inferring Supper-Resolution Tissue Architecture by Integrating Spatial Transcriptomics and Histology

This software package implements iStar
(Inferring Supper-resolution Tissue ARchitecture),
which enhances the spatial resolution of spatial transcriptomic data
from a spot-level to a near-single-cell level.
The iStar method is presented in the following paper:

Daiwei Zhang, Amelia Schroeder, Hanying Yan, Haochen Yang, Jian Hu, Michelle Y. Y. Lee, Kyung S. Cho, Katalin Susztak, George X. Xu, Michael D. Feldman, Edward B. Lee, Emma E. Furth, Linghua Wang, Mingyao Li.
Inferring super-resolution tissue architecture by integrating spatial transcriptomics with histology.
*Nature Biotechnology* (2024).
https://doi.org/10.1038/s41587-023-02019-9

## iStar WebUI (Update 2024-05-11)

A web version of iStar is now available at [istar.live](http://istar.live).
New features will be continuously added here as we develop expansions of the model.
Please contact [Daiwei (David) Zhang](mailto:daiwei.zhang@pennmedicine.upenn.edu)
if you encounter any issues or have any questions.

## Get Started

To run the demo,
```python
# Use Python 3.9 or above
pip install -r requirements.txt
./run_demo.sh
```
Using GPUs is highly recommended.

### Data format

- `he-raw.jpg`: Raw histology image
- `cnts.tsv`: Gene count matrix.
    - Row 1: Gene names.
    - Row 2 and after: Each row is a spot.
    - Column 1: Spot ID.
    - Column 2 and after: Each column is a gene.
- `locs-raw.tsv`: Spot locations
    - Row 1: Header
    - Row 2 and after: Each row is a spot. Must match rows in `cnts.tsv`
    - Column 1: Spot ID
    - Column 2: x-coordinate (horizontal axis). Must be in the same space as axis-1 (column) of the array indices of pixels in `he-raw.jpg`.
    - Column 2: y-coordinate (vertical axis). Must be in the same space as axis-0 (row) of the array indices of pixels in `he-raw.jpg`.
- `pixel-size-raw.txt`: Side length (in micrometers) of pixels in `he-raw.jpg`. This value is usually between 0.1 and 1.0.
    - For [Visium](https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/spatial) data, this value can be approximated by `8000 / 2000 * tissue_hires_scalef`, where `tissue_hires_scalef` is stored in `scalefactors_json.json`.
- `radius-raw.txt`: Number of pixels per spot radius in `he-raw.jpg`.
    - For [Visium](https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/spatial) data, this value can be computed by `spot_diameter_fullres * 0.5`, where `spot_diameter_fullres` is stored in `scalefactors_json.json`, and should be close to `55 * 0.5 / pixel_size_raw`.

## License

The software package is licensed under GPL-3.0.
For commercial use, please contact
[Daiwei (David) Zhang](mailto:daiwei.zhang@pennmedicine.upenn.edu) and
[Mingyao Li](mailto:mingyao@pennmedicine.upenn.edu).

## Acknowledgements

The codes for iStar are written by Daiwei (David) Zhang and under active development.
Please open an issue on GitHub if you have any questions about the software package.

The codebase for the hierarchical vision transformer is built upon
[Vision Transformer](https://arxiv.org/abs/2010.11929)
(as implemented by [Hugging Face](https://github.com/huggingface/pytorch-image-models)),
[DINO](https://github.com/facebookresearch/dino), and
[HIPT](https://github.com/mahmoodlab/HIPT).
We thank the authors for releasing the codes and the model weights.

If you find this work useful, please consider citing
```bash
@article{zhang2024inferring,
  title = {Inferring Super-Resolution Tissue Architecture by Integrating Spatial Transcriptomics with Histology},
  author = {Zhang, Daiwei and Schroeder, Amelia and Yan, Hanying and Yang, Haochen and Hu, Jian and Lee, Michelle Y. Y. and Cho, Kyung S. and Susztak, Katalin and Xu, George X. and Feldman, Michael D. and Lee, Edward B. and Furth, Emma E. and Wang, Linghua and Li, Mingyao},
  year = {2024},
  month = jan,
  journal = {Nature Biotechnology},
  pages = {1--6},
  doi = {10.1038/s41587-023-02019-9},
}
```
as well as
[Vision Transformer](https://arxiv.org/abs/2010.11929),
[DINO](https://github.com/facebookresearch/dino), and
[HIPT](https://github.com/mahmoodlab/HIPT).
