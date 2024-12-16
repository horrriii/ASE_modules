import pytest
import numpy as np
from ase.build import molecule
from ase.optimize import LBFGS, LBFGSLineSearch, BFGS, BFGSLineSearch 
from ase.io import read, write

@pytest.fixture
def atoms():
    atoms = molecule('H2')
    atoms.positions -= atoms.positions[0]
    assert atoms.positions[0] == pytest.approx([0, 0, 0])
    atoms.pbc = 1
    atoms.cell = [5, 5, 6]
    return atoms

optclasses = [
    LBFGS, BFGS,
    LBFGSLineSearch, BFGSLineSearch, 
]

@pytest.mark.filterwarnings("ignore::ResourceWarning")
@pytest.mark.parametrize('optcls', optclasses)
@pytest.mark.calculator('espresso', tprnfor=True)
def test_h2_bond_relax(factory, optcls, atoms):
    atoms.calc = factory.calc()
    opt = optcls(atoms, logfile="opt.log", trajectory="opt.traj", append_trajectory=False)
    fmax = 0.01
    opt.run(fmax=fmax)


@pytest.mark.filterwarnings("ignore::ResourceWarning")
@pytest.mark.parametrize('optcls', optclasses)
@pytest.mark.calculator('espresso', tprnfor=True)
def test_h2_bond_relax_restart(factory, optcls, atoms):
    atoms.calc = factory.calc()
    opt = optcls(atoms, logfile="opt.log", trajectory="opt.traj", append_trajectory=True)
    fmax = 0.01
    opt.run(fmax=fmax, steps=1)
    
    atoms = read("opt.traj")
    atoms.calc = factory.calc()
    opt = optcls(atoms, logfile="opt.log", trajectory="opt.traj", append_trajectory=True)
    fmax = 0.01
    opt.run(fmax=fmax, steps=1)

    
