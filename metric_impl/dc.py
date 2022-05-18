r"""PyTorch implementation of Density and Coverage (D&C). Based on Reliable Fidelity and Diversity Metrics for
Generative Models https://arxiv.org/abs/2002.09797 and repository
https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
"""
from typing import Tuple
import torch

from piq.base import BaseFeatureMetric
from piq.utils import _validate_input

from metric_impl.pr import _compute_nearest_neighbour_distances, _compute_pairwise_distance


class DC(BaseFeatureMetric):
    r"""Interface of Density and Coverage.
    It's computed for a whole set of data and uses features from encoder instead of images itself to decrease
    computation cost. Density and Coverage can compare two data distributions with different number of samples.
    But dimensionalities should match, otherwise it won't be possible to correctly compute statistics.

    Args:
        real_features: Samples from data distribution. Shape :math:`(N_x, D)`
        fake_features: Samples from generated distribution. Shape :math:`(N_y, D)`

    Returns:
        density: Scalar value of the density of image sets features.
        coverage: Scalar value of the coverage of image sets features.

    References:
        Ferjad Naeem M. et al. (2020).
        Reliable Fidelity and Diversity Metrics for Generative Models.
        International Conference on Machine Learning,
        https://arxiv.org/abs/2002.09797
    """

    def __init__(self, nearest_k: int = 5) -> None:
        r"""
        Args:
            nearest_k: Nearest neighbor to compute the non-parametric representation. Shape :math:`1`
        """
        super(DC, self).__init__()

        self.nearest_k = nearest_k

    def compute_metric(self, y_features: torch.Tensor, x_features: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Creates non-parametric representations of the manifolds of real and generated data and computes
        the density and coverage between them.

        Args:
            real_features: Samples from data distribution. Shape :math:`(N_x, D)`
            fake_features: Samples from fake distribution. Shape :math:`(N_x, D)`
        Returns:
            precision: Scalar value of the precision of the generated images.
            recall: Scalar value of the recall of the generated images.
        """
        # _validate_input([real_features, fake_features], dim_range=(2, 2), size_range=(1, 2))
        real_features = y_features
        fake_features = x_features
        real_nearest_neighbour_distances = _compute_nearest_neighbour_distances(real_features, self.nearest_k)
        distance_real_fake = _compute_pairwise_distance(real_features, fake_features)

        density = (1 / self.nearest_k) * (
                distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1)
        ).sum(dim=0).float().mean()

        coverage = (
                distance_real_fake.min(dim=1)[0] < real_nearest_neighbour_distances
        ).float().mean()

        return density, coverage
