"""Unpolarised 14 electron simple cubic homogeneous electron gas."""

from ferminet import base_config
from ferminet.utils import system
from ferminet.pbc import envelopes

import numpy as np


def _sc_lattice_vecs(rs: float, nelec: int) -> np.ndarray:
  """Returns simple cubic lattice vectors with Wigner-Seitz radius rs."""
  volume = (4 / 3) * np.pi * (rs**3) * nelec
  length = volume**(1 / 3)
  return length * np.eye(3)


def get_config():
  """Returns config for running unpolarised 14 electron gas with FermiNet."""
  # Get default options.
  cfg = base_config.default()
  cfg.system.electrons = (7, 7)
  # A ghost atom at the origin defines one-electron coordinate system.
  # Element 'X' is a dummy nucleus with zero charge
  cfg.system.molecule = [system.Atom("X", (0., 0., 0.))]
  # Pretraining is not currently implemented for systems in PBC
  cfg.pretrain.method = None

  lattice = _sc_lattice_vecs(1.0, sum(cfg.system.electrons))
  kpoints = envelopes.make_kpoints(lattice, cfg.system.electrons)

  cfg.system.make_local_energy_fn = "ferminet.pbc.hamiltonian.local_energy"
  cfg.system.make_local_energy_kwargs = {"lattice": lattice, "heg": True}
  cfg.network.make_feature_layer_fn = \
    "ferminet.pbc.feature_layer.make_pbc_feature_layer"
  cfg.network.make_feature_layer_kwargs = {
      "lattice": lattice,
      "include_r_ae": False
  }
  cfg.network.make_envelope_fn = \
    "ferminet.pbc.envelopes.make_multiwave_envelope"
  cfg.network.make_envelope_kwargs = {"kpoints": kpoints}
  cfg.network.full_det = True
  return cfg
