from ferminet import networks
from ferminet import hamiltonian
import jax
from jax import lax
import jax.numpy as jnp

def potential_energy_2d(r_ae, r_ee, atoms, charges):
    v_ee = jnp.sum(jnp.triu(jnp.log(r_ee[..., 0]), k=1))
    v_ae = -jnp.sum(charges*jnp.log(r_ae[..., 0]))
    r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
    v_aa = jnp.sum(
        jnp.triu(
            (charges[None, ...] * charges[..., None]) * jnp.log(r_aa), k=1
        )
    )
    return v_ee + v_ae + v_aa

def local_energy_2d(f, atoms, charges):
    ke = hamiltonian.local_kinetic_energy(f)

    def _e_l(params, x):
        _, _, r_ae, r_ee = networks.construct_input_features(x, atoms, ndim=2)
        potential = potential_energy_2d(r_ae, r_ee, atoms, charges)
        kinetic = ke(params, x)
        return potential + kinetic

    return _e_l