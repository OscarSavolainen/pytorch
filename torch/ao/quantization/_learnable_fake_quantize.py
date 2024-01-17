import torch
from torch.nn.parameter import Parameter
from typing import List

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

        self.forward = self.float_forward

    @torch.jit.export
    def extra_repr(self):
        """
        Verbose logging for when one calls this object.
        """
        return (
            "forward_call={}, fake_quant_enabled={}, observer_enabled={}, static_enabled={}, "
            "learning_enabled={}, scale={}, zero_point={}, "
            "dtype={}, quant_min={}, quant_max={}, qscheme={}".format(
                self.forward.__name__,
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
        r"""
        Called whenever the buffer values have been edited.
        """
        ipdb.set_trace()
        # NOTE: make sure this doesn't get called every time qparams get changed, that would suck.

        # We check if the int8-state buffers have been updated, and if so we update the forward call
        self._update_forward_call(name, value)

        # Call the original __setattr__ method
        super(_LearnableFakeQuantize, self).__setattr__(name, value)


    def _update_forward_call(self, name, value):
        r"""
        If the int-state buffers have been edited, and depending on the new buffer values,
        we set the forward call of self to be either the float forward or the fake-quant forward,
        and we either do PTQ or not.
        """
        # Check if the attribute being set is a int8-state toggling buffer
        if name in ['fake_quant_enabled', 'static_enabled']:
            # Compare the new value with the current value to detect changes
            ipdb.set_trace()
            # Check if the value has to change for this code to work,
            # may just be able to use the buffer value directly
            # if hasattr(self, name) and getattr(self, name) != value:
            # Fake quant disabled, we use the fake-quant forward
            if hasattr(self, name) and getattr(self, name) != value:
                if value not in [0, 1]:
                    raise ValueError(f"The value of {name} should be 0 or 1.")

            # If we're in float, we either do PTQ or we don't
            if not self.fake_quant_enabled:
                if self.static_enabled:
                    self.forward = self._get_PTQ_forward() # X
                else:
                    self.forward = self._get_float_forward() # X

            # If fake-quant is enabled, it can be either PTQ or QAT
            else:
                if self.static_enabled:
                    self.forward = self._get_PTQ_fake_quant_forward()
                else:
                    self.forward = self._get_fake_quant_forward()


    def _get_float_forward(self):
        r"""
        Sets the forward call to be the float forward, merely returning the input without
        any quantization operations.
        """
        return self.float_forward

    def _get_PTQ_float_forward(self):
        r"""
        Sets the forward call to be the PTQ + float operation.
        """
        return self._PTQ


    def _get_fake_quant_forward(self):
        r"""
        Sets the forward call to be the fake-quantize operation, depending
        on the qscheme. Supported qschemes are:
        - Affine, per-tensor
        - Affine, per-channel
        - Symmetric, per-tensor
        - Symmetric, per-channel
        """
        if _is_per_tensor(self.qscheme):
            if _is_symmetric_quant(self.qscheme):
                # Symmetric, per-tensor
                fake_quant_forward = self.fake_quant_forward_per_tensor_symetric
            else:
                # Affine, per-tensor
                fake_quant_forward = self.fake_quant_forward_per_tensor
        elif _is_per_channel(self.qscheme):
            if _is_symmetric_quant(self.qscheme):
                # Symmetric, per-channel
                fake_quant_forward = self.fake_quant_forward_per_channel_symmetric
            else:
                # Affine, per-channel
                fake_quant_forward = self.fake_quant_forward_per_channel
        else:
            raise NotImplementedError(
                "We currently only have LearnableFakeQuant symmetric/affine per-channel/per-tensor implementations."
            )

        return fake_quant_forward


    def _get_PTQ_fake_quant_forward(self):
        r"""
        We call PTQ and fake-quant forward (depending on qscheme), sequentially.
        """
        fake_quant_forward = self._get_fake_quant_forward()
        return fake_quant_forward(self._PTQ) # NOTE: check this works, may need a lambda


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
    def calculate_qparams(self):
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]
        scale = self.scale.detach()
        zero_point = self.zero_point.detach().round().clamp(self.quant_min, self.quant_max).long()
        return scale, zero_point

    ################
    ## FORWARD CALLS
    ################
    def float_forward(self, X):
        r"""
        The floating-point forward call. We do no quantization,
        and merely return the floating point tensor.
        """
        return X


    # Fake-Quantization forward calls
    def fake_quant_forward_per_tensor(self, X):
        r"""
        Affine, per-tensor fake-quantization of X.
        """
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]

        grad_factor = self._grad_scaling(X)

        X = torch._fake_quantize_learnable_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.quant_min, self.quant_max, grad_factor)
        return X

    def fake_quant_forward_per_channel(self, X):
        r"""
        Affine, per-channel fake-quantization of X.
        """
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]

        grad_factor = self._grad_scaling(X)

        X = torch._fake_quantize_learnable_per_channel_affine(
                    X, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
        return X

    def fake_quant_forward_per_tensor_symmetric(self, X):
        r"""
        Symmetric, per-tensor fake-quantization forward call.
        """
        self.zero_point.data.zero_()
        X = self.fake_quant_forward_per_channel(self, X)
        return X

    def fake_quant_forward_per_channel_symmetric(self, X):
        r"""
        Symmetric, per-channel fake-quantization forward call.
        """
        self.zero_point.data.zero_()
        X = self.fake_quant_forward_per_channel(self, X)
        return X

    # PTQ forward call
    def _PTQ(self, X):
        r"""
        Helper function for PTQ. Calls the PTQ observer on X to calculate the qparams,
        and updates the qparams of `self`.
        """
        self.activation_post_process(X.detach())
        _scale, _zero_point = self.activation_post_process.calculate_qparams()
        _scale = _scale.to(self.scale.device)
        _zero_point = _zero_point.to(self.zero_point.device)
        self.scale.data.copy_(_scale)
        self.zero_point.data.copy_(_zero_point)
        return X

    def _grad_scaling(self, X):
        r"""
        Grad-scaling calculation.
        """
        if self.use_grad_scaling:
            grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
        else:
            grad_factor = 1.0

        return grad_factor
