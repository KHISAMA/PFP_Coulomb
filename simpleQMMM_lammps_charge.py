from __future__ import print_function
import numpy as np

from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import convert_string_to_fd


class SimpleQMMM_charge(Calculator):
    """Simple QMMM calculator."""
    implemented_properties = ['energy', 'forces', 'stress', 'charges']

    def __init__(self, selection, qmcalc, mmcalc1, mmcalc2, selection_charge, charge_values, vacuum=None,
                mmcalc1_2=None, mmcalc2_2=None):
        """SimpleQMMM object.

        The energy is calculated as::

                    _          _          _
            E = E  (R  ) - E  (R  ) + E  (R   )
                 QM  QM     MM  QM     MM  all

        parameters:

        selection: list of int, slice object or list of bool
            Selection out of all the atoms that belong to the QM part.
        qmcalc: Calculator object
            QM-calculator.
        mmcalc1: Calculator object
            MM-calculator used for QM region.
        mmcalc2: Calculator object
            MM-calculator used for everything.
        vacuum: float or None
            Amount of vacuum to add around QM atoms.  Use None if QM
            calculator doesn't need a box.
        
        * EXTRA CHARGE CALCULATION 
        * (ONLY AVAILABLE WITH MODIFIED LAMMPSLIB CALCULATOR AS MMCALC1 AND MMCALC2)
        * author Kaoru HISAMA
        *
        selection_charge: list of int
            Selection out of all the atoms that belong to the Charged MM Part.
        charge_values: numpy ndarray
            charge value of the atoms specified by selection_charge

        """
        self.selection = selection
        self.qmcalc = qmcalc
        self.mmcalc1 = mmcalc1
        self.mmcalc2 = mmcalc2
        self.vacuum = vacuum

        self.qmatoms = None
        self.center = None

        self.name = '{0}-{1}+{1}'.format(qmcalc.name, mmcalc1.name)
        
        self.selection_charge = selection_charge
        self.charge_values = charge_values
        
        self.mmcalc1_2 = mmcalc1_2
        self.mmcalc2_2 = mmcalc2_2

        Calculator.__init__(self)

    def initialize_qm(self, atoms):
        constraints = atoms.constraints
        atoms.constraints = []
        self.qmatoms = atoms[self.selection]
        atoms.constraints = constraints
        #self.qmatoms.pbc = False
        self.qmatoms.pbc = True
        if self.vacuum:
            self.qmatoms.center(vacuum=self.vacuum)
            self.center = self.qmatoms.positions.mean(axis=0)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        #if self.qmatoms is None:
        #    self.initialize_qm(atoms)
        self.initialize_qm(atoms) #initialize every time

        self.qmatoms.positions = atoms.positions[self.selection]
        if self.vacuum:
            self.qmatoms.positions += (self.center -
                                       self.qmatoms.positions.mean(axis=0))

        # CALCULATION OF EXTRA CHARGES
        # getting charge for qmcalc
        
        ids = np.arange(len(atoms))
        #print("ids shape: ",ids.shape)
        ids_qm = ids[self.selection]
        
        #print("ids_qm: ", ids_qm)
        
        charges = np.zeros_like(ids,dtype='float')
        charges_qm = self.qmcalc.get_charges(self.qmatoms)
        charges_qm = charges_qm.reshape(-1)
        #print("charges by QM calculator for qmatoms: ", charges_qm)
        
        for iqm, iall in enumerate(ids_qm):
            charges[iall] = charges_qm[iqm]
            #print("iqm iall charges[iall] charges_qm[iqm]: ", iqm, iall, charges[iall], charges_qm[iqm] )
            
        #print("charges by QM calculator: ", charges)
        
        # setting charge for mmcalc part
        
        for icharge, iall in enumerate(self.selection_charge):
            charges[iall] = self.charge_values[icharge]
            #print("ichage iall charges[iall] charge_values[iqm]: ", icharge, iall, charges[iall], self.charge_values[icharge] )
        
        #print("with extra charges: ", charges)
        
        # passing charges to mmcalc1 and mmcalc2
        
        self.mmcalc2.reset_additional_command()
        self.mmcalc1.reset_additional_command()
        
        for iatom, qi in enumerate(charges):
            cmd = "set atom " + str(iatom+1) + " charge " + str(qi)
            self.mmcalc2.set_additional_command(cmd)
        
        for iatom, qi in enumerate(charges_qm):
            cmd = "set atom " + str(iatom+1) + " charge " + str(qi)
            self.mmcalc1.set_additional_command(cmd)
            #self.mmcalc2.set_additional_command(cmd)
            
        # CALCULATION OF EXTRA CHARGES DONE!
            
        energy = self.qmcalc.get_potential_energy(self.qmatoms)
        qmforces = self.qmcalc.get_forces(self.qmatoms)
        energy += self.mmcalc2.get_potential_energy(atoms)
        forces = self.mmcalc2.get_forces(atoms)
        
        if(self.mmcalc2_2 is not None):
            energy += self.mmcalc2_2.get_potential_energy(atoms)
            forces += self.mmcalc2_2.get_forces(atoms)                   

        if self.vacuum:
            qmforces -= qmforces.mean(axis=0)
        forces[self.selection] += qmforces

        energy -= self.mmcalc1.get_potential_energy(self.qmatoms)
        forces[self.selection] -= self.mmcalc1.get_forces(self.qmatoms)
        
        if(self.mmcalc1_2 is not None):
            energy -= self.mmcalc1_2.get_potential_energy(self.qmatoms)
            forces[self.selection] -= self.mmcalc1_2.get_forces(self.qmatoms)  
        
        volume = atoms.get_volume()
        virial0 = volume*self.qmcalc.get_stress(self.qmatoms)
        virial1 = volume*self.mmcalc1.get_stress(self.qmatoms)
        virial2 = volume*self.mmcalc2.get_stress(atoms)
        
        if(self.mmcalc2_2 is not None):
            virial2 += volume*self.mmcalc2_2.get_stress(atoms)
        if(self.mmcalc1_2 is not None):
            virial1 += volume*self.mmcalc1_2.get_stress(self.qmatoms)
        
        stress = (virial0 + virial2 - virial1)/volume
        
        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress
        self.results['charges'] = charges.reshape([-1,1])