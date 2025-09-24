
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

from .linear_flipout import LinearFlipout

__all__ = ["QuantizedLinearFlipout"]

class QuantizedLinearFlipout(LinearFlipout):
    def __init__(self,
                 in_features,
                 out_features):

        super(QuantizedLinearFlipout, self).__init__(
                 in_features,
                 out_features)

        self.is_dequant = False

    def get_scale_and_zero_point(self, x, upper_bound=100, target_range=255):
        """ An implementation for symmetric quantization
        
        Parameters
        ----------
        x: tensor
            Input tensor.
        upper_bound: int, optional
            Restrict the maximum value of the original tensor (select 100 empirically).
        target_range: int, optional
            The range of target data type (255 for int8)

        Returns
        ----------
        scale: float

        zero_point: int

        """
        scale = torch.zeros(1).to(x.device) # initialize
        zero_point = torch.zeros(1).to(x.device) # zero point is zero since we only consider symmetric quantization
        xmax = torch.clamp(x.abs().max(), 0, upper_bound) # determine and restrict the maximum value (minimum value should be 0 since the absolute value is always non-negative)
        scale = xmax*2/target_range # original range divided by target range
        return scale, zero_point

    def get_quantized_tensor(self, x, default_scale=0.1):
        """ Quantize tensors

        Parameters
        ----------
        x: tensors
            Input tensor.

        default_scale: float, optional
            Default scale for the case that the computed scale is zero.


        Returns
        ----------
        quantized_x: tensors


        """
        scale, zero_point = self.get_scale_and_zero_point(x)
        if scale == 0:
            scale = torch.tensor([default_scale]) # avoid zero scale
        quantized_x = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)

        return quantized_x

    def get_dequantized_tensor(self, x):

        dequantized_x = x.dequantize()

        return dequantized_x


    def quantize(self):
        self.quantized_mu_weight = Parameter(self.get_quantized_tensor(self.mu_weight), requires_grad=False)
        self.quantized_sigma_weight = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_weight))), requires_grad=False)
        delattr(self, "mu_weight")
        delattr(self, "rho_weight")

        self.quantized_mu_bias = Parameter(self.get_quantized_tensor(self.mu_bias), requires_grad=False)
        self.quantized_sigma_bias = Parameter(self.get_quantized_tensor(torch.log1p(torch.exp(self.rho_bias))), requires_grad=False)
        delattr(self, "mu_bias")
        delattr(self, "rho_bias")

    def dequantize(self):
        self.mu_weight = self.get_dequantized_tensor(self.quantized_mu_weight)
        self.sigma_weight = self.get_dequantized_tensor(self.quantized_sigma_weight)

        self.mu_bias = self.get_dequantized_tensor(self.quantized_mu_bias)
        self.sigma_bias = self.get_dequantized_tensor(self.quantized_sigma_bias)
        return

    def forward(self, x, normal_scale=6/255, default_scale=0.1, default_zero_point=128, return_kl=True):
        """ Forward pass

        Parameters
        ----------
        x: tensors
            Input tensor.
        
        normal_scale: float, optional
            Scale for quantized tensor sampled from normal distribution.
            since 99.7% values will lie within 3 standard deviations, the original range is set as 6.
        
        default_scale: float, optional
            Default scale for quantized input tensor and quantized output tensor.
            Set to 0.1 by grid search.

        default_zero_point: int, optional
            Default zero point for quantized input tensor and quantized output tensor.
            Set to 128 for quint8 tensor.



        Returns
        ----------
        out: tensors
            Output tensor. Already dequantized.
        KL: float
            set to 0 since we diable KL divergence computation in quantized layers.


        """

        if self.dnn_to_bnn_flag:
            return_kl = False

        bias = None
        if self.quantized_mu_bias is not None:
            if not self.is_dequant:
                    self.dequantize()
                    self.is_dequant = True
            bias = self.mu_bias

        outputs = torch.nn.quantized.functional.linear(x, self.quantized_mu_weight, bias, scale=default_scale, zero_point=default_zero_point) # input: quint8, weight: qint8, bias: fp32

        # sampling perturbation signs
        sign_input = torch.zeros(x.shape).uniform_(-1, 1).sign()
        sign_output = torch.zeros(outputs.shape).uniform_(-1, 1).sign()
        sign_input = torch.quantize_per_tensor(sign_input, default_scale, default_zero_point, torch.quint8)
        sign_output = torch.quantize_per_tensor(sign_output, default_scale, default_zero_point, torch.quint8)

         # getting perturbation weights
        eps_weight = torch.quantize_per_tensor(self.eps_weight.data.normal_(), normal_scale, 0, torch.qint8)
        new_scale = (self.quantized_sigma_weight.q_scale())*(eps_weight.q_scale())
        delta_weight = torch.ops.quantized.mul(self.quantized_sigma_weight, eps_weight, new_scale, 0)

        bias = None
        if self.quantized_sigma_bias is not None:
            eps_bias = self.eps_bias.data.normal_()
            bias = (self.sigma_bias * eps_bias)

        # perturbed feedforward
        x = torch.ops.quantized.mul(x, sign_input, default_scale, default_zero_point)

        perturbed_outputs = torch.nn.quantized.functional.linear(x,
                            weight=delta_weight, bias=bias, scale=default_scale, zero_point=default_zero_point)
        perturbed_outputs = torch.ops.quantized.mul(perturbed_outputs, sign_output, default_scale, default_zero_point)
        out = torch.ops.quantized.add(outputs, perturbed_outputs, default_scale, default_zero_point)
        out = out.dequantize()

        if return_kl:
            return out, 0
        
        return out
