# NLS_DL-SCA

Core code for **Deep metric learning-based side-channel analysis with improved robustness and efficiency**, published in *Applied Intelligence*, volume 55, article 712, 2025.

**Authors:** Kaibin Li, Yihuai Liang, Hua Meng, and Zhengchun Zhou.

**Paper:** https://doi.org/10.1007/s10489-025-06586-z

The method combines deep metric learning with template attacks for profiled side-channel analysis. A normalized lifted structured (NLS) loss uses label-informed pairwise distances to learn discriminative trace embeddings. Gaussian templates fitted to these embeddings are then used for key recovery.

## Repository contents

| File | Purpose |
| --- | --- |
| `Triplet_loss.py` | NLS loss implementation and label-informed pairwise distance calculation. |
| `attack20_nls.py` | NLS training and template-attack evaluation, including repeated runs and profiling-set-size experiments. |
| `template_attack_auto_sca_rank.py` | Hyperparameter search using template-attack performance as the search objective. |
| `triplet_semihard_loss.py` | Label-informed semi-hard triplet loss. |
| `Triplet_attack.py` | Training and template-attack evaluation using the semi-hard triplet objective. |

## Using the core loss

`Triplet_loss.py` can be imported independently of the experiment scripts. It requires TensorFlow. The loss factory accepts `alpha_value`, `margin`, and `num_classes`:

```python
from Triplet_loss import lifted_struct_loss

loss_fn = lifted_struct_loss(
    alpha_value=0.2,
    margin=0.6,
    num_classes=256,
)
```

Pass integer class labels and the embedding vectors produced by the feature extractor to this loss. The example uses the identity (ID) leakage model with 256 classes. The Hamming weight (HW) model has 9 classes. Choose the remaining parameters for the experiment being performed.

## Experiment scripts

The scripts contain branches for ASCAD, ASCAD with random keys (`ASCAD_rand`), and AES_HD (`AESHD`), with HW and ID leakage models. Dataset selection, noise settings, training parameters, and output locations are configured in the scripts.

The current `attack20_nls.py` default selects ASCAD_rand under the ID model, 50,000 profiling traces, and **2 training runs**. The filename does not set the number of runs; this is controlled by `num_repeats`.

The full experiment scripts also require a `Util` package containing the dataset loaders, feature extractor, template-attack implementation, key-rank evaluation, and plotting helpers. These modules are **not included in the current repository snapshot**. The public datasets and trained models are not bundled with this release.

Imports in the current scripts include TensorFlow/Keras, NumPy, scikit-learn, and Matplotlib. The hyperparameter-search script additionally imports SciPy and Keras Tuner. Package versions should match the original experiment environment; this release does not yet specify a tested dependency set.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{li2025deep,
  author  = {Li, Kaibin and Liang, Yihuai and Meng, Hua and Zhou, Zhengchun},
  title   = {Deep metric learning-based side-channel analysis with improved robustness and efficiency},
  journal = {Applied Intelligence},
  volume  = {55},
  pages   = {712},
  year    = {2025},
  doi     = {10.1007/s10489-025-06586-z}
}
```
