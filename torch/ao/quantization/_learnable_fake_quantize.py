import torch
from torch.nn.parameter import Parameter
from typing import Callable, List, Tuple

from torch.ao.quantization.fake_quantize import _is_per_tensor, _is_per_channel, _is_symmetric_quant

import ipdb


__all__: List[str] = []

class _LearnableFakeQuantize(torch.ao.quantization.FakeQuantizeBase):
    r"""Generalized extension of the FakeQuantize module in fake_quantize.py.

    This is an extension of the FakeQuantize module in fake_quantize.py, which
    supports more generalized lower-bit quantization and support learning of the scale
    and zero point parameters through backpropagation. For literature references,
    please see the class _LearnableFakeQuantizePerTensorOp.

    In addition to the attributes in the original FakeQuantize module, the _LearnableFakeQuantize
    module also includes the following attributes to support quantization parameter learning.

    * :attr:`channel_len` defines the length of the channel when initializing scale and zero point
      for the per channel case.

    * :attr:`use_grad_scaling` defines the flag for whether the gradients for scale and zero point are
      normalized by the constant, which is proportional to the square root of the number of
      elements in the tensor. The related literature justifying the use of this particular constant
      can be found here: https://openreview.net/pdf?id=rkgO66VKDS.

    * :attr:`fake_quant_enabled` defines the flag for enabling fake quantization on the output.

    * :attr:`static_enabled` defines the flag for using observer's static estimation for
      scale and zero point.

    * :attr:`learning_enabled` defines the flag for enabling backpropagation for scale and zero point.
    """
    def __init__(self, observer, quant_min=0, quant_max=255, scale=1., zero_point=0., channel_len=-1,
                 use_grad_scaling=False, **observer_kwargs):
        super().__init__()
        assert quant_min < quant_max, 'quant_min must be strictly less than quant_max.'
        self.quant_min = quant_min
        self.quant_max = quant_max
        # also pass quant_min and quant_max to observer
        observer_kwargs["quant_min"] = quant_min
        observer_kwargs["quant_max"] = quant_max
        self.use_grad_scaling = use_grad_scaling
        if channel_len == -1:
            self.scale = Parameter(torch.tensor([scale]))
            self.zero_point = Parameter(torch.tensor([zero_point]))
        else:
            assert isinstance(channel_len, int) and channel_len > 0, "Channel size must be a positive integer."
            self.scale = Parameter(torch.tensor([scale] * channel_len))
            self.zero_point = Parameter(torch.tensor([zero_point] * channel_len))

        self.activation_post_process = observer(**observer_kwargs)
        assert torch.iinfo(self.activation_post_process.dtype).min <= quant_min, \
            'quant_min out of bound'
        assert quant_max <= torch.iinfo(self.activation_post_process.dtype).max, \
            'quant_max out of bound'
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('static_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('learning_enabled', torch.tensor([0], dtype=torch.uint8))

        bitrange = torch.tensor(quant_max - quant_min + 1).double()
        self.bitwidth = int(torch.log2(bitrange).item())
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))

        self.forward = self._get_observation_fake_quant_forward()

    @torch.jit.export
    def extra_repr(self):
        """Define a string representation of the object's attributes."""
        return (
            "forward_call={}, fake_quant_enabled={}, observer_enabled={}, static_enabled={}, "
            "learning_enabled={}, scale={}, zero_point={}, "
            "dtype={}, quant_min={}, quant_max={}, qscheme={}".format(
                self.forward.__name__ if self.forward.__name__ != "<lambda>"
                else self.forward.__defaults__[0].__name__, ## STRING LAMBDAS HERE WITH A "->" BETWEEN THEM TO SHOW HOW THEY'RE FED
                self.fake_quant_enabled.item(),
                int(self.observer_enabled.item()),
                self.static_enabled.item(),
                self.learning_enabled.item(),
                self.scale,
                self.zero_point,
                self.dtype,
                self.activation_post_process.quant_min,
                self.activation_post_process.quant_max,
                self.qscheme,
            )
        )

    def __setattr__(self, name, value):
        """
        Called whenever the attribute values on an instance have been edited. `static_enabled`
        is largely redundant to `observer_enabled`, and so if one has been updated we update the other.
        This allows us to just use `observer_enabled` in a backwards compatible way.
        """
        ipdb.set_trace()
        # NOTE: make sure this doesn't get called every time qparams get changed, that would suck.

        # If `static_enabled` has been updated, we update `observer_enabled` to match it,
        # and vice-versa.
        if torch.is_buffer(value):
            if name is 'static_enabled':
                assert value in [0, 1]
                self.enable_observer(self, enabled=value)
            if name is 'observer_enabled':
                assert value in [0, 1]
                self.toggle_observer_update(enabled=value)

        # Call the original __setattr__ method
        super(self.__class__, self).__setattr__(name, value)

    ##########################
    ## TOGGLING THE INT8 STATE
    ##########################
    @torch.jit.export
    def extra_repr(self) -> str:
        """Define a string representation of the object's attributes."""
        return (
            "fake_quant_enabled={}, observer_enabled={}, static_enabled={}, "
            "learning_enabled={}, scale={}, zero_point={}, "
            "dtype={}, quant_min={}, quant_max={}, qscheme={}".format(
                self.fake_quant_enabled.item(),
                int(self.observer_enabled.item()),
                self.static_enabled.item(),
                self.learning_enabled.item(),
                self.scale,
                self.zero_point,
                self.dtype,
                self.activation_post_process.quant_min,
                self.activation_post_process.quant_max,
                self.qscheme,
            )
        )

    @torch.jit.export
    def enable_param_learning(self):
        r"""Enable parameter learning over static observer estimates.

        Enables learning of quantization parameters and
        disables static observer estimates. Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=True) \
            .toggle_fake_quant(enabled=True) \
            .toggle_observer_update(enabled=False)
        return self

    @torch.jit.export
    def enable_static_estimate(self):
        """Enable static estimates of quantization parameters.

        Enables static observer estimates and disables learning of
        quantization parameters. Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=False) \
            .toggle_fake_quant(enabled=True) \
            .toggle_observer_update(enabled=True)

    @torch.jit.export
    def enable_static_observation(self):
        """Enable accumulation of data without updating quantization parameters.

        Enables static observer accumulating data from input but doesn't
        update the quantization parameters. Forward path returns the original X.
        """
        self.toggle_qparam_learning(enabled=False) \
            .toggle_fake_quant(enabled=False) \
            .toggle_observer_update(enabled=True)

    @torch.jit.export
    def toggle_observer_update(self, enabled=True):
        self.static_enabled[0] = int(enabled)  # type: ignore[operator]
        return self

    @torch.jit.export
    def enable_observer(self, enabled=True):
        self.toggle_observer_update(enabled)

    @torch.jit.export
    def toggle_qparam_learning(self, enabled=True):
        self.learning_enabled[0] = int(enabled)  # type: ignore[operator]
        self.scale.requires_grad = enabled
        self.zero_point.requires_grad = enabled
        return self

    @torch.jit.export
    def toggle_fake_quant(self, enabled=True):
        self.fake_quant_enabled[0] = int(enabled)
        return self

    @torch.jit.export
    def observe_quant_params(self):
        print(f'_LearnableFakeQuantize Scale: {self.scale.detach()}')
        print(f'_LearnableFakeQuantize Zero Point: {self.zero_point.detach()}')

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]
        scale = self.scale.detach()
        zero_point = self.zero_point.detach().round().clamp(self.quant_min, self.quant_max).long()
        return scale, zero_point

    ######################
    # FORWARD CALL GETTERS
    ######################
    def _get_float_forward(self) -> Callable[[torch.Tensor], torch.Tensor]:
        r"""
        Returns a callable that performs the float forward, merely returning the input without
        any quantization operations.
        """
        return self.float_forward

    def _get_observation_float_forward(self) -> Callable[[torch.Tensor], torch.Tensor]:
        r"""
        Returns a callable with the observation (PTQ) + float (no quantization) forward call.
        """
        return self.observation_forward

    def _get_fake_quant_forward(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Returns a callable that performs the fake-quantize operation, depending
        on `self.qscheme`. Supported qschemes are:
        - Affine, per-tensor
        - Affine, per-channel
        - Symmetric, per-tensor
        - Symmetric, per-channel
        """
        if _is_per_tensor(self.qscheme):
            if _is_symmetric_quant(self.qscheme):
                fake_quant_forward = self.fake_quant_forward_per_tensor_symmetric
            else:
                fake_quant_forward = self.fake_quant_forward_per_tensor_affine

        elif _is_per_channel(self.qscheme):
            if _is_symmetric_quant(self.qscheme):
                fake_quant_forward = self.fake_quant_forward_per_channel_symmetric
            else:
                fake_quant_forward = self.fake_quant_forward_per_channel_affine
        else:
            raise NotImplementedError(
                "_LearnableFakeQuantize currently only supports symmetric/affine and per-channel/per-tensor forward calls."
            )

        return fake_quant_forward

    def _get_observation_fake_quant_forward(self) -> Callable[[torch.Tensor], torch.Tensor]:
        r"""
        Returns a callable that performs PTQ and then the fake-quantize operation, sequentially.
        """
        fake_quant_forward = self._get_fake_quant_forward()
        return fake_quant_forward(self.observation_forward) # NOTE: check this works, may need a lambda

    ################
    ## FORWARD CALLS
    ################
    # NOTE: have these all in shared file with other qconfig forwards.
    def float_forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        The floating-point forward call. We do no quantization,
        and merely return the floating point tensor.
        """
        return X

    # Fake-Quantization forward calls
    def fake_quant_forward_per_tensor_affine(self, X: torch.Tensor) -> torch.Tensor:
        """
        Affine, per-tensor fake-quantization of X.
        """
        # Keeps the scale from learning too-small values.
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]

        # Grad-scaling calculation.
        grad_factor = self._grad_scaling(X)

        X = torch._fake_quantize_learnable_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.quant_min, self.quant_max, grad_factor)
        return X

    def fake_quant_forward_per_channel_affine(self, X: torch.Tensor) -> torch.Tensor:
        """
        Affine, per-channel fake-quantization of X.
        """
        # Keeps the scale from learning too-small values.
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]

        # Grad-scaling calculation.
        grad_factor = self._grad_scaling(X)

        X = torch._fake_quantize_learnable_per_channel_affine(
                    X, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
        return X

    def fake_quant_forward_per_tensor_symmetric(self, X: torch.Tensor) -> torch.Tensor:
        """
        Symmetric, per-tensor fake-quantization forward call.
        """
        # We center the zero-points for symmetric quantization
        self.zero_point.data.zero_()
        X = self.fake_quant_forward_per_tensor_affine(self, X)
        return X

    def fake_quant_forward_per_channel_symmetric(self, X: torch.Tensor) -> torch.Tensor:
        """
        Symmetric, per-channel fake-quantization forward call.
        """
        # We center the zero-points for symmetric quantization
        self.zero_point.data.zero_()
        X = self.fake_quant_forward_per_channel_affine(self, X)
        return X

    # PTQ forward call
    def observation_forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calls the PTQ observer on X to calculate the qparams,
        and updates the qparams of `self`.
        """
        self.activation_post_process(X.detach())
        _scale, _zero_point = self.activation_post_process.calculate_qparams()
        _scale = _scale.to(self.scale.device)
        _zero_point = _zero_point.to(self.zero_point.device)
        self.scale.data.copy_(_scale)
        self.zero_point.data.copy_(_zero_point)
        return X

    def _grad_scaling(self, X: torch.Tensor) -> float:
        r"""
        Calculates grad-scaling factor.
        """
        import ipdb
        ipdb.set_trace()
        # CHeck if return is tensor or float
        if self.use_grad_scaling:
            grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
        else:
            grad_factor = 1.0

        return grad_factor
