import zfit
import zfit.z.numpy as znp


class PowerLaw(zfit.pdf.ZPDF):
    """Custom 1D Power-Law PDF: f(x) = x^(gamma)."""

    _PARAMS = ("gamma",)

    @zfit.supports(norm=False)
    def _pdf(self, x, norm, params):
        data = x[0]
        gamma = params["gamma"]
        return znp.power(data, gamma)
