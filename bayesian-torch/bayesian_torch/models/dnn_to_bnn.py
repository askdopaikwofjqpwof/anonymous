

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import bayesian_torch.layers as bayesian_layers
from bayesian_torch.utils.util import get_rho


# --------------------------------------------------------------------------------
# Parameters used to define BNN layyers.
#    bnn_prior_parameters = {
#       "prior_mu": 0.0,
#       "prior_sigma": 1.0,
#       "posterior_mu_init": 0.0,
#       "posterior_rho_init": -4.0,
#       "type": "Reparameterization",  # Flipout or Reparameterization
# }


def bnn_linear_layer(params, d):
    layer_type = d.__class__.__name__ + params["type"]
    layer_fn = getattr(bayesian_layers, layer_type)  # Get BNN layer
    bnn_layer = layer_fn(
        in_features=d.in_features,
        out_features=d.out_features,
        prior_mean=params["prior_mu"],
        prior_variance=params["prior_sigma"],
        prior_type=params["prior_type"],
        posterior_mu_init=params["posterior_mu_init"],
        posterior_rho_init=params["posterior_rho_init"],
        bias=d.bias is not None,
    )
    # if MOPED is enabled initialize mu and sigma
    if params["moped_enable"]:
        delta = params["moped_delta"]
        bnn_layer.mu_weight.data.copy_(d.weight.data)
        bnn_layer.rho_weight.data.copy_(get_rho(d.weight.data, delta))
        if bnn_layer.mu_bias is not None:
            bnn_layer.mu_bias.data.copy_(d.bias.data)
            bnn_layer.rho_bias.data.copy_(get_rho(d.bias.data, delta))
    bnn_layer.dnn_to_bnn_flag = True
    return bnn_layer


def bnn_conv_layer(params, d):
    layer_type = d.__class__.__name__ + params["type"]
    layer_fn = getattr(bayesian_layers, layer_type)  # Get BNN layer
    bnn_layer = layer_fn(
        in_channels=d.in_channels,
        out_channels=d.out_channels,
        kernel_size=d.kernel_size,
        stride=d.stride,
        padding=d.padding,
        dilation=d.dilation,
        groups=d.groups,
        prior_mean=params["prior_mu"],
        prior_variance=params["prior_sigma"],
        prior_type=params["prior_type"],
        posterior_mu_init=params["posterior_mu_init"],
        posterior_rho_init=params["posterior_rho_init"],
        bias=d.bias is not None,
    )

    # if MOPED is enabled,  initialize mu and sigma
    if params["moped_enable"]:
        delta = params["moped_delta"]
        bnn_layer.mu_kernel.data.copy_(d.weight.data)
        bnn_layer.rho_kernel.data.copy_(get_rho(d.weight.data, delta))
        if bnn_layer.mu_bias is not None:
            bnn_layer.mu_bias.data.copy_(d.bias.data)
            bnn_layer.rho_bias.data.copy_(get_rho(d.bias.data, delta))
    bnn_layer.dnn_to_bnn_flag = True
    return bnn_layer


def bnn_lstm_layer(params, d):
    layer_type = d.__class__.__name__ + params["type"]
    layer_fn = getattr(bayesian_layers, layer_type)  # Get BNN layer
    bnn_layer = layer_fn(
        in_features=d.input_size,
        out_features=d.hidden_size,
        prior_mean=params["prior_mu"],
        prior_variance=params["prior_sigma"],
        posterior_mu_init=params["posterior_mu_init"],
        posterior_rho_init=params["posterior_rho_init"],
        bias=d.bias is not None,
    )
    # if MOPED is enabled initialize mu and sigma
    if params["moped_enable"]:
        print("WARNING: MOPED method is not supported for LSTM layers!!!")
    bnn_layer.dnn_to_bnn_flag = True
    return bnn_layer


# replaces linear and conv layers
# bnn_prior_parameters - check the template at the top.
def dnn_to_bnn(m, bnn_prior_parameters):
    for name, value in list(m._modules.items()):
        if m._modules[name]._modules:
            dnn_to_bnn(m._modules[name], bnn_prior_parameters)
        elif "Conv" in m._modules[name].__class__.__name__:
            setattr(
                m,
                name,
                bnn_conv_layer(
                    bnn_prior_parameters,
                    m._modules[name]))
        elif "Linear" in m._modules[name].__class__.__name__:
            setattr(
                m,
                name,
                bnn_linear_layer(
                    bnn_prior_parameters,
                    m._modules[name]))
        elif "LSTM" in m._modules[name].__class__.__name__:
            setattr(
                m,
                name,
                bnn_lstm_layer(
                    bnn_prior_parameters,
                    m._modules[name]))
        else:
            pass
    return


def get_kl_loss(m):
    kl_loss = None
    for layer in m.modules():
        if hasattr(layer, "kl_loss"):
            if kl_loss is None:
                kl_loss = layer.kl_loss()
            else:
                kl_loss += layer.kl_loss()
    return kl_loss
