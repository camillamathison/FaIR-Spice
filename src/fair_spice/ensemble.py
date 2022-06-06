from typing import Hashable, Iterable, Optional
import xarray as xr
import netCDF4
from pathlib import Path


class EnsembleNetcdf:
    def __init__(self, filename, run0: xr.Dataset, dim="ensemble_member", mode="w"):
        if mode not in ("a", "w"):
            raise ValueError('mode must be "w" or "a".')

        path = Path(filename)
        self._unlimited_dim = dim
        self._cursor = 0

        if mode == "w":
            # write mode means create a new file.
            # don't overwrite existing files
            if path.exists():
                raise FileExistsError(f"Wont overwrite existing files. Delete {path} first.")
            else:
                # use run0 as a template to create the netcdf file
                run0.to_netcdf(filename, unlimited_dims=(dim,))

        self.dset = netCDF4.Dataset(filename, "a")

        if mode == "a":
            # work out where we need to start appending from
            # before we append the next iteration
            self._cursor = self.dset.dimensions[dim].size
            self.add_member(run0)

    def _write_member(self, dataset: xr.Dataset, n: int):
        for var, data in dataset.variables.items():
            if self._unlimited_dim in data.dims:
                if len(data.dims) == 1:
                    self.dset.variables[var][n] = data
                else:
                    self.dset.variables[var][n, :] = data
            else:
                self.dset.variables[var][:] = data

    def add_member(self, dataset: xr.Dataset):
        self._cursor = self._cursor + 1
        self._write_member(dataset, self._cursor)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.dset.sync()
        self.dset.close()


def stream_ensemble(filename: str, ensemble: Iterable[xr.Dataset], mode="w"):
    """Run a model with an ensemble of initial conditions and stream the results to a netcdf file.

    Arguments:
        filename: The file to save the output.
        model: A callable model that takes a dict of input parameters and
               returns a xarray Dataset of results.
        configs: An iterable of dicts that define the ensemble. Each will be
               passed to the model as a single input.
        mode: "w": Write a new file
              "a": Append to an existing file.

    """
    iensemble = iter(ensemble)
    run0 = next(iensemble)
    with EnsembleNetcdf(filename, run0, mode=mode) as dset:
        for member in iensemble:
            dset.add_member(member)


def make_member(
    dset: xr.Dataset,
    number: Optional[int] = None,
    invariant: Optional[list[Hashable]] = None,
) -> xr.Dataset:
    """Make an xarray Dataset an ensemble member.

    An `ensemble_member` dimension of size 1 is added to the dataset.
    By default all variables and coords are considered to vary across the
    ensemble.
    Some coordinates might be the same in all ensemble members.
    Pass any the names of invariant coordinates as `invariant`
    to prevent broadcasting ensemble members in those dimensions.

    Arguments:
        dset: The ensemble member dataset
        number: The ensemble member number. If None, no data is attached to the
                ensemble_member coordinate
        invariant: A list of coordinates that do not vary by ensemble_member.
    """
    dim = "ensemble_member"
    if number is not None:
        dset = dset.assign_coords({dim: number})
    dset = dset.expand_dims(dim)
    invariant = (invariant or []) + list(dset.dims)
    for c in dset.coords:
        if c not in invariant:
            dset[c] = dset[c].expand_dims(dim)
    return dset
