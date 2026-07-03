"""JAX/Flax Target Emission."""

import os  # pylint: disable=import-outside-toplevel

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorVisitor
from ml_switcheroo_compiler.backends.common.mixins.nn import GroupNormConfig, NNASTVisitor
from ml_switcheroo_compiler.backends.jax.generator_mixins import (
    JaxAudioVisitor,
    JaxControlFlowVisitor,
    JaxDistributedVisitor,
    JaxMathVisitor,
    JaxVisionVisitor,
)
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("jax")
class JAXCodeGenerator(BaseGenerator):
    """JAX code generator."""

    def __init__(self, graph: object) -> None:
        """Init."""
        super().__init__(graph)
        self.visitors.extend(
            [
                SharedASTGeneratorVisitor(generator=self),
                JaxAudioVisitor(),
                JaxControlFlowVisitor(),
                JaxDistributedVisitor(),
                JaxMathVisitor(),
                JaxVisionVisitor(),
            ]
        )

    def _get_backend_prefix(self) -> str:
        """Function docstring."""
        return "jax"  # pragma: no cover

    def _format_zeros_like(self, op: str, kwargs: object) -> str:
        """Function docstring.

        Args:
        op: Arg.
        kwargs: Arg.
        """
        res = f"jnp.{op}({{shape}})"
        if "dtype" in kwargs:  # pragma: no branch
            res += f", dtype='{kwargs['dtype']}'"  # pragma: no cover
        return res

    def _format_full(self, kwargs: object) -> str:
        """Function docstring.

        Args:
        kwargs: Arg.
        """
        res = "jnp.full({shape}, {fill_value})"
        if "dtype" in kwargs:  # pragma: no branch
            res += f", dtype='{kwargs['dtype']}'"  # pragma: no cover
        return res

    def get_fallback_prefix(self) -> str:
        """Get the fallback prefix for generic operations."""
        return "jnp"

    def get_ops_map(self, kwargs: dict) -> dict[str, str]:
        """Get the operation mapping dictionary.

        Args:
            kwargs: Operation kwargs.

        Returns:
            Dictionary mapping operation type to format string.
        """
        return {
            "Infeed": "jax.lax.infeed(shape={shape})",
            "Outfeed": "jax.lax.outfeed({0})",
            "AxisIndex": "jax.lax.axis_index({axis_name})",
            "WithShardingConstraint": "jax.lax.with_sharding_constraint({0}, {sharding})",
            "RandomCategorical": "jax.random.categorical({0}, {1}, axis={axis}, shape={shape})",
            "Beta": "jax.random.beta({0}, {1}, {2}, {shape})",
            "Gamma": "jax.random.gamma({0}, {1}, {shape})",
            "RngBitGenerator": "jax.random.bits({0}, {shape})",
            "RngUniform": "jax.random.uniform(jax.random.PRNGKey(0), {shape}, minval={0}, maxval={1})",
            "Matmul": "jnp.matmul({0}, {1})",
            "Trace": "tf.linalg.trace",
            "Adjoint": "tf.linalg.adjoint",
            "BandPart": "tf.linalg.band_part",
            "CholeskySolve": "tf.linalg.cholesky_solve",
            "Dot": "jnp.dot({0}, {1})",
            "BroadcastTo": "jnp.broadcast_to({0}, {shape})",
            "Reshape": "jnp.reshape({0}, {shape})",
            "TruncateDiv": "jnp.trunc(jnp.divide({0}, {1}))",
            "TruncateMod": "jnp.fmod({0}, {1})",
            "TrueDivide": "jnp.true_divide({0}, {1})",
            "Arange": "jnp.arange({0})",
            "Zeros": self._format_zeros_like("zeros", kwargs),
            "Ones": self._format_zeros_like("ones", kwargs),
            "Full": self._format_full(kwargs),
            "Sort": "jnp.sort({0}, axis={dimension})",
            "ArgSort": "jnp.argsort({0}, axis={dimension})",
            "Allclose": "jnp.allclose({0}, {1}, rtol={rtol}, atol={atol}, equal_nan={equal_nan})",
            "Fftnd": "jnp.fft.fftn({0})",
            "Ifftnd": "jnp.fft.ifftn({0})",
            "Rfftnd": "jnp.fft.rfftn({0})",
            "Irfftnd": "jnp.fft.irfftn({0})",
            "Fftshift": "jnp.fft.fftshift({0})",
            "Ifftshift": "jnp.fft.ifftshift({0})",
            "Fft": "jnp.fft.fft({0})",
            "Rfft": "jnp.fft.rfft({0})",
            "Fftn": "jnp.fft.fftn({0})",
            "Binomial": "jax.random.binomial({0}, {1}, {2}, {shape})",
            "Cauchy": "jax.random.cauchy({0}, {shape})",
            "Chisquare": "jax.random.chisquare({0}, {1}, {shape})",
            "Dirichlet": "jax.random.dirichlet({0}, {1}, {shape})",
            "DoubleSidedMaxwell": "jax.random.double_sided_maxwell({0}, {1}, {shape})",
            "Exponential": "jax.random.exponential({0}, {shape})",
            "F": "jax.random.f({0}, {1}, {2}, {shape})",
            "Gumbel": "jax.random.gumbel({0}, {shape})",
            "Laplace": "jax.random.laplace({0}, {shape})",
            "Loggamma": "jax.random.loggamma({0}, {1}, {shape})",
            "Logistic": "jax.random.logistic({0}, {shape})",
            "Lognormal": "jax.random.lognormal({0}, {shape})",
            "Maxwell": "jax.random.maxwell({0}, {shape})",
            "MultivariateNormal": "jax.random.multivariate_normal({0}, {1}, {2}, {shape})",
            "Pareto": "jax.random.pareto({0}, {1}, {shape})",
            "Poisson": "jax.random.poisson({0}, {1}, {shape})",
            "Rayleigh": "jax.random.rayleigh({0}, {shape})",
            "T": "jax.random.t({0}, {1}, {shape})",
            "Triangular": "jax.random.triangular({0}, {1}, {2}, {shape})",
            "Wald": "jax.random.wald({0}, {1}, {2}, {shape})",
            "WeibullMin": "jax.random.weibull_min({0}, {1}, {2}, {shape})",
            "Clone": "jax.random.clone({0})",
            "KeyData": "jax.random.key_data({0})",
            "KeyImpl": "jax.random.key_impl({0})",
            "WrapKeyData": "jax.random.wrap_key_data({0})",
            "Bits": "jax.random.bits({0}, {shape})",
            "GeneralizedNormal": "jax.random.generalized_normal({0}, {1}, {shape})",
            "Orthogonal": "jax.random.orthogonal({0}, {1}, {shape})",
            "RandomGammaP": "jax.random.gamma_p({0}, {1})",
            "Vecdot": "jax.numpy.vecdot({0}, {1})",
            "CustomLinearSolve": "jax.scipy.linalg.solve({0}, {1})",
            "DebugInfs": "jax.debug.visualize_array_updates({0})",
            "DebugNans": "jax.debug.visualize_array_updates({0})",
            "AssociativeScan": "jax.lax.associative_scan({0}, {1}, reverse={reverse}, axis={axis})",
            "DevicePutReplicated": "jax.device_put_replicated({0}, {devices})",
            "DevicePutSharded": "jax.device_put_sharded({0}, {devices})",
            "AllToAll": "jax.lax.all_to_all({0}, split_axis={split_axis}, concat_axis={concat_axis}, axis_name={axis_name})",
            "Pmax": "jax.lax.pmax({0}, axis_name={axis_name})",
            "Pmin": "jax.lax.pmin({0}, axis_name={axis_name})",
            "PsumScatter": "jax.lax.psum_scatter({0}, scatter_dimension={scatter_dimension}, axis_name={axis_name})",
            "Pswapaxes": "jax.lax.pswapaxes({0}, axis_name={axis_name}, axis={axis})",
            "Erfinv": "jax.scipy.special.erfinv({0})",
            "Erf": "jax.scipy.special.erf({0})",
            "SpecialGamma": "jax.scipy.special.gamma({0})",
            "BesselJn": "jax.scipy.special.bessel_jn({1}, v={0})",
            "Digamma": "jax.scipy.special.digamma({0})",
            "Polygamma": "jax.scipy.special.polygamma({0}, {1})",
            "Zeta": "jax.scipy.special.zeta({0}, {1})",
            "Igamma": "jax.scipy.special.gammainc({0}, {1})",
            "Igammac": "jax.scipy.special.gammaincc({0}, {1})",
            "Betainc": "jax.scipy.special.betainc({0}, {1}, {2})",
            "BesselI0e": "jax.scipy.special.i0e({0})",
            "BesselI1e": "jax.scipy.special.i1e({0})",
            "NormPdf": "jax.scipy.stats.norm.pdf({0}, loc={1}, scale={2})",
            "NormCdf": "jax.scipy.stats.norm.cdf({0}, loc={1}, scale={2})",
            "GammaPdf": "jax.scipy.stats.gamma.pdf({0}, {1}, loc={2}, scale={3})",
            "GammaCdf": "jax.scipy.stats.gamma.cdf({0}, {1}, loc={2}, scale={3})",
            "BetaPdf": "jax.scipy.stats.beta.pdf({0}, {1}, {2}, loc={3}, scale={4})",
            "BetaCdf": "jax.scipy.stats.beta.cdf({0}, {1}, {2}, loc={3}, scale={4})",
            "PoissonPmf": "jax.scipy.stats.poisson.pmf({0}, {1}, loc={2})",
            "PoissonCdf": "jax.scipy.stats.poisson.cdf({0}, {1}, loc={2})",
            "BinomPmf": "jax.scipy.stats.binom.pmf({0}, {1}, {2}, loc={3})",
            "BinomCdf": "jax.scipy.special.betainc({1} - ({0} - {3}), ({0} - {3}) + 1, 1 - {2})",
            "Convolve2d": "jax.scipy.signal.convolve2d({0}, {1}, mode='{mode}', boundary='{boundary}', fillvalue={fillvalue})",
            "Fftconvolve": "jax.scipy.signal.fftconvolve({0}, {1}, mode='{mode}', axes={axes})",
            "Welch": "jax.scipy.signal.welch({0}, fs={fs}, window='{window}', nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}, detrend='{detrend}', return_onesided={return_onesided}, scaling='{scaling}', axis={axis}, average='{average}')",
            "Convolve": "jax.scipy.signal.convolve({0}, {1}, mode='{mode}')",
            "Cholesky": "jnp.linalg.cholesky({0})",
            "Svd": "jnp.linalg.svd({0})",
            "Qr": "jnp.linalg.qr({0})",
            "Inv": "jnp.linalg.inv({0})",
            "Pinv": "jnp.linalg.pinv({0})",
            "Det": "jnp.linalg.det({0})",
            "Slogdet": "jnp.linalg.slogdet({0})",
            "Eigh": "jnp.linalg.eigh({0})",
            "Eigvalsh": "jnp.linalg.eigvalsh({0})",
            "MatrixPower": "jnp.linalg.matrix_power({0}, {n})",
            "Solve": "jnp.linalg.solve({0}, {1})",
            "TriInv": "jnp.linalg.inv({0})",
            "TriangularSolve": "jax.scipy.linalg.solve_triangular({0}, {1}, lower={lower}, unit_diagonal={unit_diagonal})",
            "Lu": "jax.scipy.linalg.lu({0})",
            "LuFactor": "jax.scipy.linalg.lu_factor({0})",
            "LuSolve": "jax.scipy.linalg.lu_solve(({0}, {1}), {2})",
            "Norm": "jnp.linalg.norm({0}, ord={ord}, axis={axis}, keepdims={keepdims})",
            "MatrixExponential": "jax.scipy.linalg.expm({0})",
            "Cross": "jnp.cross({0}, {1}, axisa={axisa}, axisb={axisb}, axisc={axisc}, axis={axis})",
            "NanToNum": "jnp.nan_to_num({0}, nan={nan}, posinf={posinf}, neginf={neginf})",
            "AssignVariable": "{0}",
            "StopGradient": "jax.lax.stop_gradient({0})",
            "Resize": "jax.image.resize({0}, shape={size}, method={method}, antialias={antialias})",
            "AffineGrid": "jax.image.affine_grid({0}, {size}, align_corners={align_corners})",
            "GridSample": "jax.image.grid_sample({0}, {1}, mode={mode}, padding_mode={padding_mode}, align_corners={align_corners})",
            "DrawBoundingBoxes": "{0}",
            "RgbToYiq": "jax.image.rgb_to_yiq({0})",
            "YiqToRgb": "jax.image.yiq_to_rgb({0})",
            "RgbToYuv": "jax.image.rgb_to_yuv({0})",
            "YuvToRgb": "jax.image.yuv_to_rgb({0})",
            "Ifft": "jnp.fft.ifft({0}, n={n}, axis={axis})",
            "Fft2d": "jnp.fft.fft2({0}, s={s}, axes={axes})",
            "Ifft2d": "jnp.fft.ifft2({0}, s={s}, axes={axes})",
            "Fft3d": "jnp.fft.fftn({0}, s={s}, axes={axes})",
            "Ifft3d": "jnp.fft.ifftn({0}, s={s}, axes={axes})",
            "Rfft2d": "jnp.fft.rfft2({0}, s={s}, axes={axes})",
            "Rfft3d": "jnp.fft.rfftn({0}, s={s}, axes={axes})",
            "Irfft": "jnp.fft.irfft({0}, n={n}, axis={axis})",
            "Irfft2d": "jnp.fft.irfft2({0}, s={s}, axes={axes})",
            "Irfft3d": "jnp.fft.irfftn({0}, s={s}, axes={axes})",
            "Stft": "jax.scipy.signal.stft({0})",
            "Istft": "jax.scipy.signal.istft({0})",
            "HannWindow": "jnp.hanning({window_length})",
            "HammingWindow": "jnp.hamming({window_length})",
            "KaiserWindow": "jnp.kaiser({window_length}, beta={beta})",
            "ReadVariable": "{0}",
            "TensorScatterUpdate": "{0}.at[tuple(jnp.moveaxis({1}, -1, 0))].set({2})",
            "TensorScatterAdd": "{0}.at[tuple(jnp.moveaxis({1}, -1, 0))].add({2})",
            "TensorScatterMax": "{0}.at[tuple(jnp.moveaxis({1}, -1, 0))].max({2})",
            "TensorScatterMin": "{0}.at[tuple(jnp.moveaxis({1}, -1, 0))].min({2})",
        }

    def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
        """Evaluate emit constant assignment.

        Args:
            var_name (str): Argument var_name
            val_repr (str): Argument val_repr
        """
        self.add_line(f"{var_name} = jnp.array({val_repr})")

    def _generate_file_header(self) -> list[str]:
        """Generate file header with module docstrings."""
        return [self.header.strip()]

    def _resolve_imports(self) -> list[str]:
        """Resolve and register required imports."""
        tmpl_path = os.path.join(os.path.dirname(__file__), "jax_prefix.py.tmpl")
        with open(tmpl_path, encoding="utf-8") as f:
            jax_prefix_template = f.read()

        return [
            "import jax",
            "import jax.numpy as jnp",
            *NNASTVisitor(generator=self)._get_group_norm_code(
                GroupNormConfig(
                    "jax",
                    "jax.numpy as jnp",
                    "jnp.reshape",
                    "jnp.mean",
                    "jnp.var",
                    "jnp.sqrt",
                    "axis",
                    "keepdims",
                )
            ),
            "import jax.scipy.special",
            *jax_prefix_template.split("\n"),
        ]

    def _generate_function_signature(self) -> None:
        """Generate the main function signature."""
        self.indent_level = 0
        self.add_line("def apply_model(params, *args, **kwargs):")
        self.indent_level += 1
