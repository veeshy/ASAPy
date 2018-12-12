from abc import ABCMeta, abstractmethod
from io import StringIO

import data
from mixin import EqualityMixin

class AngleEnergy(EqualityMixin, metaclass=ABCMeta):
    """Distribution in angle and energy of a secondary particle."""
    @abstractmethod
    def to_hdf5(self, group):
        pass

    @staticmethod
    def from_hdf5(group):
        """Generate angle-energy distribution from HDF5 data

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from

        Returns
        -------
        AngleEnergy
            Angle-energy distribution

        """
        dist_type = group.attrs['type'].decode()
        if dist_type == 'uncorrelated':
            return uncorrelated.UncorrelatedAngleEnergy.from_hdf5(group)
        elif dist_type == 'correlated':
            return correlated.CorrelatedAngleEnergy.from_hdf5(group)
        elif dist_type == 'kalbach-mann':
            return kalbach_mann.KalbachMann.from_hdf5(group)
        elif dist_type == 'nbody':
            return NBodyPhaseSpace.from_hdf5(group)

    @staticmethod
    def from_ace(ace, location_dist, location_start, rx=None):
        """Generate an angle-energy distribution from ACE data

        Parameters
        ----------
        ace : ace.Table
            ACE table to read from
        location_dist : int
            Index in the XSS array corresponding to the start of a block,
            e.g. JXS(11) for the the DLW block.
        location_start : int
            Index in the XSS array corresponding to the start of an energy
            distribution array
        rx : Reaction
            Reaction this energy distribution will be associated with

        Returns
        -------
        distribution : AngleEnergy
            Secondary angle-energy distribution

        """
        # Set starting index for energy distribution
        idx = location_dist + location_start - 1

        law = int(ace.xss[idx + 1])
        location_data = int(ace.xss[idx + 2])

        # Position index for reading law data
        idx = location_dist + location_data - 1

        # Parse energy distribution data
        if law == 2:
            distribution = uncorrelated.UncorrelatedAngleEnergy()
            distribution.energy = energy_distribution.DiscretePhoton.from_ace(ace, idx)
        elif law in (3, 33):
            distribution = uncorrelated.UncorrelatedAngleEnergy()
            distribution.energy = energy_distribution.LevelInelastic.from_ace(ace, idx)
        elif law == 4:
            distribution = uncorrelated.UncorrelatedAngleEnergy()
            distribution.energy = energy_distribution.ContinuousTabular.from_ace(
                ace, idx, location_dist)
        elif law == 5:
            distribution = uncorrelated.UncorrelatedAngleEnergy()
            distribution.energy = GeneralEvaporation.from_ace(ace, idx)
        elif law == 7:
            distribution = uncorrelated.UncorrelatedAngleEnergy()
            distribution.energy = MaxwellEnergy.from_ace(ace, idx)
        elif law == 9:
            distribution = uncorrelated.UncorrelatedAngleEnergy()
            distribution.energy = Evaporation.from_ace(ace, idx)
        elif law == 11:
            distribution = uncorrelated.UncorrelatedAngleEnergy()
            distribution.energy = WattEnergy.from_ace(ace, idx)
        elif law == 44:
            distribution = kalbach_mann.KalbachMann.from_ace(
                ace, idx, location_dist)
        elif law == 61:
            distribution = correlated.CorrelatedAngleEnergy.from_ace(
                ace, idx, location_dist)
        elif law == 66:
            distribution = NBodyPhaseSpace.from_ace(
                ace, idx, rx.q_value)
        else:
            raise ValueError("Unsupported ACE secondary energy "
                             "distribution law {}".format(law))

        return distribution

# get around cyclic dependencies the lazy way!
import kalbach_mann 
import uncorrelated
import energy_distribution
import correlated