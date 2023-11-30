import gpytorch


class ARDModule(gpytorch.models.ExactGP):
    def __init__(self, likelihood, n_features):
        super(ARDModule, self).__init__(
            train_inputs=None, train_targets=None, likelihood=likelihood
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=n_features)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
