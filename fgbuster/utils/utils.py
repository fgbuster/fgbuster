import jax
from jax import jit, jacfwd, jacrev , grad , numpy as jnp
import sympy
import jax.numpy as jnp
from jax import jit 
from collections import defaultdict
from functools import partial
import chex

def bandpass_integration(f):
    ''' Decorator for bandpass integration

    Parameters
    ----------
    f: callable
        Function to evaluate an SED. Its first argument must be a frequency
        array. The other positional or keyword arguments are arbitrary.

    Returns
    -------
    f: callable
        The function now accepts as the first argument

        * array with the frequencies, as before (delta bandpasses)
        * the list or tuple with the bandpasses. Each entry is a pair of arrays
          (frequencies, transmittance). The SED is evaluated at these frequencies
          multiplied by the transmittance and integrated with the trapezoid rule.

        Note that the routine does not perform anything more that this. In
        particular it does NOT:

        * normalize the transmittance to 1 or any other value
        * perform any unit conversion before integrating the SED

        Make sure you normalize and "convert the units" of the
        transmittance in such a way that you get the correct result.
    '''
    def integrated_f(nu, *params, **kwargs):
        # It is user's responsibility to provide weights in the same units
        # as the components
        if isinstance(nu, (list, tuple)):
            out_shape = f(jnp.array(100.), *params, **kwargs).shape[:-1]
            res = jnp.empty(out_shape + (len(nu),))
            for i, (band_nu, band_w) in enumerate(nu):
                res[..., i] = jnp.trapz(
                    f(band_nu, *params, **kwargs) * band_w,
                    band_nu * 1e9)
            return res
        return f(nu, *params)

    return integrated_f

def broadcasted(f):
    def wrapper(comp,*args, **kwargs):
        broadcastable_args = make_broadcastable_ndim(*args)
        return f(comp,*broadcastable_args, **kwargs)
    return wrapper


def Lambdify(x, y):
    # Return the lambdified function directly
    return bandpass_integration(sympy.lambdify(x, y, 'jax'))

def GetJac(func,params):
    argnums = tuple(range(1, len(params) + 1)) 
    return jacfwd(func,argnums=1)

def GetGrad(func, params):
    argnums = tuple(range(1, len(params) + 1))
    grads = [jax.jacfwd(func, argnums=i) for i in range(1, len(params) + 1)]

    def grad_func(*args, **kwargs):
        return [g(*args, **kwargs) for g in grads]

    return grad_func

def make_broadcastable_ndim(*args):
    # Step 1: Inspect dimensions and prepare mapping
    chex.assert_tree_has_only_ndarrays(args)

    dims_map = {}
    reverse_dims_map = defaultdict(list)
    broadcastable_args = []
    for pos, arg in enumerate(args):
        # check using chex that that arg is a jnp.array
        #print(type(arg))
        dim = arg.shape[0] if arg.ndim > 0 else 0  # Use 0 for scalars
        if pos == 0:
            dims_map[pos] = 1
            reverse_dims_map[dim].append(pos)
            broadcastable_args.append(arg)
        else:
            if len(reverse_dims_map[dim]) != 0:
                arg_broadcast_dim = dims_map[reverse_dims_map[dim][0]]
                dims_map[pos] = arg_broadcast_dim
                reverse_dims_map[dim].append(pos)
                # add new axis as much as arg_broadcast_dim
                broadcastable_args.append(arg[(...,) + (None,) * (arg_broadcast_dim - arg.ndim)])
            elif dim != 0:
                dims_map[pos] = max(dims_map.values()) + 1
                reverse_dims_map[dim].append(pos)
                broadcastable_args.append(arg[(...,) + (None,) * (dims_map[pos] - arg.ndim)])
            else:
                dims_map[pos] = 0
                reverse_dims_map[dim].append(pos)
                broadcastable_args.append(arg)
            
    return broadcastable_args

# Broadcast if dim is not one in all cases
# add an extra dimension if the dim is not 1

def make_broadcastable(*args):
    # Step 1: Inspect dimensions and prepare mapping
    chex.assert_tree_has_only_ndarrays(args)
    
    dims_map = {}
    reverse_dims_map = defaultdict(list)
    broadcastable_args = []
    for pos, arg in enumerate(args):
        dim = arg.shape[0] if arg.ndim > 0 else 0  # Use 0 for scalars

        if pos == 0:
            dims_map[pos] = 1
            reverse_dims_map[dim].append(pos)
            broadcastable_args.append(arg)
        else:
            if dim != 0:
                dims_map[pos] = max(dims_map.values()) + 1
                reverse_dims_map[dim].append(pos)
                broadcastable_args.append(arg[(...,) + (None,) * (dims_map[pos] - arg.ndim)])
            else:
                dims_map[pos] = 0
                reverse_dims_map[dim].append(pos)
                broadcastable_args.append(arg)
            
    return broadcastable_args
