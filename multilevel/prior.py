import torch

from deepinv.models.tv import TVDenoiser
from deepinv.optim import Prior


class L12Prior(Prior):
    r"""
    :math:`\ell_{1,2}` mixed norm prior :math:`g(x) = \| x \|_{1,2}`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def g(self, x, l2_axis=-1, **kwargs):
        r"""
        Computes the regularizer

        .. math:

                g(x) = \sum_{j = 1}^M \|x_j\|_2


        where x_j is the jth element in the specified axis

        :param l2_axis: axis along which to compute the l2 norm
        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.Tensor) prior :math:`g(x)`.
        """
        x_l2 = torch.norm(x, p=2, dim=l2_axis)
        return torch.sum(x_l2)

    def prox(self, x, ths=1.0, gamma=1.0, l2_axis=-1, **kwargs):
        r"""Compute the proximity operator of the mixed l12 norm

        Consider the proximity operator of the l2 norm

        .. math:
                \operatorname{prox}_{\|.\|_2}(x) = (1 - \frac{\gamma}{\max(\|x\|_2, \gamma)}) x

        The prox of l12 is the concatenation of the l2 norms in the axis l2_axis.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float ths: threshold parameter :math:`\tau`.
        :param float gamma: stepsize of the proximity operator.
        :param l2_axis: axis on which l2 norm is computed.
        :return: (torch.Tensor) proximity operator at :math:`x`.
        """

        tau_gamma = torch.tensor(ths * gamma)

        z = torch.norm(x, p=2, dim=l2_axis, keepdim=True)
        # Creating a mask to avoid diving by zero
        # if an element of z is zero, then it is zero in x, therefore torch.multiply(z, x) is zero as well
        mask_z = z > 0
        z[mask_z] = torch.max(z[mask_z], tau_gamma)
        z[mask_z] = torch.tensor(1.0) - tau_gamma / z[mask_z]

        return torch.multiply(z, x)


class TVPrior(Prior):
    r"""
    Total variation (TV) prior :math:`\reg{x} = \| D x \|_{1,2}`.

    :param float def_crit: default convergence criterion for the inner solver of the TV denoiser; default value: 1e-8.
    :param int n_it_max: maximal number of iterations for the inner solver of the TV denoiser; default value: 1000.
    """

    def __init__(self, def_crit=1e-6, n_it_max=1000, gamma_moreau=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
        self.TVModel = TVDenoiser(crit=def_crit, n_it_max=n_it_max)
        self.gamma_moreau = gamma_moreau

    def g(self, x, *args, **kwargs):
        r"""
        Computes the regularizer

        .. math::
            g(x) = \|Dx\|_{1,2}


        where D is the finite differences linear operator,
        and the 2-norm is taken on the dimension of the differences.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.Tensor) prior :math:`g(x)`.
        """
        return torch.sum(torch.sqrt(torch.sum(self.nabla(x) ** 2, axis=-1)))

    def prox(self, x, *args, gamma=1.0, **kwargs):
        r"""Compute the proximity operator of TV with the denoiser :class:`~deepinv.models.TVDenoiser`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (torch.Tensor) proximity operator at :math:`x`.
        """
        # Take normalization constant into account for computing the prox
        return self.TVModel(x, ths=gamma / self.nabla_norm())

    def nabla_norm(self):
        # Normalization constant associated with .model.tv.nabla
        return torch.sqrt(torch.tensor(8))

    def nabla(self, x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.
        """
        # Multiply linear operator by the normalization constant
        return self.TVModel.nabla(x) / self.nabla_norm()

    def nabla_adjoint(self, x):
        r"""
        Applies the adjoint of the finite difference operator.
        """
        # Multiply adjoint operator by the normalization constant
        return self.TVModel.nabla_adjoint(x) / self.nabla_norm()

    def moreau_grad(self, x, gamma=1.0):
        r"""Compute the gradient of the Moreau envelope of TV.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param float gamma: parameter associated with the Moreau envelope.
        :return: (torch.Tensor) proximity operator at :math:`x`.
        """

        dx = self.nabla(x)
        l12_prior = L12Prior()
        prox_dx = l12_prior.prox(dx, ths=1.0, gamma=gamma)
        m_grad = 1.0 / gamma * self.nabla_adjoint(dx - prox_dx)
        return m_grad

    def grad(self, x, *args, **kwargs):
        gamma = 1.0
        if self.gamma_moreau is not None:
            gamma = self.gamma_moreau

        return self.moreau_grad(x, gamma=gamma)
