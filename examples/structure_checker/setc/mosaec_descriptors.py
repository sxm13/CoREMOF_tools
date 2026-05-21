"""
Module handling the calculation of the MOSAEC descriptors 
(i.e. oxidation state and formal charges for all atomic nodes).

"""

from __future__ import annotations

# handle differing python versions (< 3.8, & > 3.9) of CSD python API
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import os
import math
import mendeleev

from ccdc import io
from ccdc import crystal
from ccdc import molecule
from ccdc import descriptors

from ccdc.crystal import Crystal
from ccdc.molecule import Atom, Bond, Molecule

from logger import *

CODE_DIR = os.path.dirname(os.path.realpath(__file__))


def readentry(input_cif: str) -> Crystal:
    """
    Reads a CIF file containing structure data and converts it to a
    standard atom labeling convention using the ccdc.crystal module.

    Parameters:
        input_cif (str): filename (.CIF) containing crystal structure data.

    Returns:
        newcif (ccdc.crystal.Crystal): Crystal object containing structural data
                                       in the standard atom labeling convention.
    """
    # read in the cif to a crystal object
    with io.CrystalReader(input_cif, format="cif") as readcif:
        cif = readcif[0]
    readcif.close()

    # to remove duplicate atoms, need the empirical formula
    formula = cif.formula
    elamnt = formula.split(" ")

    # now convert to standard labelling convention and identify
    # duplicate atoms to be removed
    cif_at_order = []
    with open(input_cif, "r") as file:
        file.seek(0)
        newstring = str()
        lines = file.readlines()
        loop_pos = 0
        start = 0
        end = 0
        columncount = 0
        type_pos = 0
        label_pos = 0
        x_pos = 0
        y_pos = 0
        z_pos = 0
        for i, line in enumerate(lines):
            lines[i] = lines[i].lstrip()
        for i, line in enumerate(lines):
            # locate atom type and site label columns
            if "loop_" in line:
                loop_pos = i
            if ("_atom" in line) and (not "_geom" in line) and (not "_aniso" in line):
                start = loop_pos + 1
                end = i + 1
        for i in range(start, end):
            if "atom_site_type_symbol" in lines[i]:
                type_pos = columncount
            if "atom_site_label" in lines[i]:
                label_pos = columncount
            columncount += 1
        counting = {}
        cutoff = {}
        to_remove = []
        for i in range(end, len(lines)):
            if "loop_" in lines[i]:
                break
            # lines with atom information will contain a ., so only look at these
            if "." in lines[i]:
                # split lines by whitespace
                col = lines[i].split()
                # keep count of how many of each element type
                if not col[type_pos] in counting:
                    counting[col[type_pos]] = 1
                elif col[type_pos] in counting:
                    counting[col[type_pos]] += 1
                # new atom labels
                newlabel = f"{col[type_pos]}{counting[col[type_pos]]}"
                lines[i] = lines[i].replace(col[label_pos], newlabel)
                # Store label order in cif to keep consistent with pymatgen
                cif_at_order.append(newlabel)
                # cutoff repeated atoms
                if newlabel in elamnt:
                    cutoff[col[type_pos]] = counting[col[type_pos]]
                if col[type_pos] in cutoff:
                    if counting[col[type_pos]] > cutoff[col[type_pos]]:
                        to_remove.append(lines[i])
        # remove unnecessary atoms
        for i in to_remove:
            lines.remove(i)
        # combine to new string
        for i in lines:
            newstring += i
        # read into new crystal object and assign bonds
        newcif = crystal.Crystal.from_string(newstring, format="cif")
        newcif.assign_bonds()
        file.close()
    return newcif, cif_at_order


def read_CSD_entry(input_refcode: str) -> Crystal:
    """
    Read entries directly from the CSD CrystalReader according to CSD refcode.

    Parameters:
        input_refcode (str): string used to identify materials in the CSD.

    Returns:
        cif (ccdc.crystal.Crystal): Crystal object containing structural data
                                    in the standard atom labeling convention.
    """
    # read in the cif to a crystal object
    csd_crystal_reader = io.CrystalReader("CSD")
    cif = csd_crystal_reader.crystal(input_refcode)
    cif.assign_bonds()
    csd_crystal_reader.close()
    return cif


def get_no_metal_molecule(inputmolecule: Molecule) -> Molecule:
    """
    Remove metal atoms from the input Molecule object.

    Parameters:
        inputmolecule (ccdc.molecule.Molecule): original Molecule object.

    Returns:
        workingmol (ccdc.molecule.Molecule): Molecule object with all metal
                                             atoms removed.
    """
    workingmol = inputmolecule.copy()
    for atom in workingmol.atoms:
        if atom.is_metal:
            workingmol.remove_atom(atom)
    workingmol.assign_bond_types(which="All")
    return workingmol


def get_unique_sites(mole: Molecule, asymmole: Molecule) -> list[Atom]:
    """
    Get the unique atoms in a structure belonging to the asymmetric unit.

    Parameters:
        mole (ccdc.molecule.Molecule): original structure Molecule object.
        asymmole (ccdc.molecule.Molecule): asymmetric unit of the structure.

    Returns:
        uniquesites (list[ccdc.molecule.Atom]): list of unique atoms in the structure
                                                that belong to the asymmetric unit.
    """
    # blank list for unique sites
    uniquesites = []
    labels = []
    asymmcoords = []
    molecoords = []
    duplicates = []
    for atom in asymmole.atoms:
        asymmcoords.append(atom.coordinates)
    for atom in mole.atoms:
        if atom.coordinates in asymmcoords:
            if not atom.coordinates in molecoords:
                if not atom.label in labels:
                    uniquesites.append(atom)
                    molecoords.append(atom.coordinates)
                    labels.append(atom.label)
                else:
                    duplicates.append(atom)
            else:
                duplicates.append(atom)
    if len(duplicates) >= 1:
        for datom in duplicates:
            for atom in uniquesites:
                if any(
                    [
                        (datom.coordinates == atom.coordinates),
                        (datom.label == atom.label),
                    ]
                ):
                    if datom.atomic_symbol == atom.atomic_symbol:
                        if len(datom.neighbours) > len(atom.neighbours):
                            uniquesites.remove(atom)
                            uniquesites.append(datom)
                    if not datom.label in labels:
                        uniquesites.append(datom)
                        labels.append(datom.label)
    return uniquesites


def get_metal_sites(sites: list[Atom]) -> list[Atom]:
    """
    Get the metal sites in a structure belonging to the asymmetric unit.

    Parameters:
        sites (list[ccdc.molecule.Atom]): list of unique atoms in the structure
                                          that belong to the asymmetric unit.

    Returns:
        metalsites (list[ccdc.molecule.Atom]): list of metal sites in the structure
                                               that belong to the asymmetric unit.
    """
    metalsites = []
    for site in sites:
        if site.is_metal == True:
            metalsites.append(site)
    return metalsites


def get_ligand_sites(
    metalsites: list[Atom], sites: list[Atom]
) -> dict[Atom, list[Atom]]:
    """
    Get the ligand sites binding each metal atom in a structure.

    Parameters:
        metalsites (list[ccdc.molecule.Atom]): list of metal sites in the structure
                                               that belong to the asymmetric unit.
        sites (list[ccdc.molecule.Atom]):  list of unique atoms in the structure
                                           that belong to the asymmetric unit.

    Returns:
        metal_sphere (dict[ccdc.molecule.Atom, list[ccdc.molecule.Atom]]):
                        dictionary with metal Atom object as keys and the the list
                        of ligand atoms which bind them as values.
    """
    metal_sphere = {}
    for metal in metalsites:
        sphere1 = []
        for ligand in metal.neighbours:
            if not ligand.is_metal == True:
                for site in sites:
                    if ligand.label == site.label:
                        sphere1.append(site)
        metal_sphere[metal] = sphere1
    return metal_sphere


def get_binding_sites(metalsites: list[Atom], uniquesites: list[Atom]) -> list[Atom]:
    """
    Get the binding sites in a structure, given the list of unique metal atoms
    and all unique atoms.

    Parameters:
        metalsites (list[ccdc.molecule.Atom]): list of unique metal atoms.
        uniquesites (list[ccdc.molecule.Atom]): list of unique atoms.

    Returns:
        binding_sites (list[ccdc.molecule.Atom]): list of binding sites connecting
                                                  metal atoms and ligands.
    """
    binding_sites = set()
    for metal in metalsites:
        for ligand in metal.neighbours:
            for site in uniquesites:
                if ligand.label == site.label:
                    binding_sites.add(site)
    return binding_sites


def ringVBOs(mole: Molecule) -> dict[int, int]:
    """
    Calculates the VBO (valence bond order) for each atom in the structure.

    Parameters:
        mole (ccdc.molecule.Molecule): Molecule object representing the structure.

    Returns:
        ringVBO (dict[int, int]): dictionary with each atom's index in mole.atoms
                                  as keys and VBO (valence bond order) as values.
    """
    ringVBO = {}
    unassigned = mole.atoms
    ringcopy = mole.copy()
    oncycle_atoms = []
    offcycle_atoms = []
    oncycle_labels = []
    offcycle_labels = []
    cyclic_periodic = []
    cyclic_periodic_labels = []
    offcycle_periodic = []

    # remove all the metals, this
    # prevents metal-containing rings (i.e. pores)
    # from interfering
    for atom in ringcopy.atoms:
        if atom.is_metal:
            ringcopy.remove_atom(atom)

    # collect all the cyclic atoms
    for atom in ringcopy.atoms:
        if atom.is_cyclic:
            if not atom in oncycle_atoms:
                oncycle_atoms.append(atom)
                oncycle_labels.append(atom.label)

    # we also need everything that the cyclic atoms are bound to
    for atom in oncycle_atoms:
        for neighbour in atom.neighbours:
            if not neighbour in oncycle_atoms:
                if not neighbour in offcycle_atoms:
                    offcycle_atoms.append(neighbour)
                    offcycle_labels.append(neighbour.label)

    # combine cyclic atoms and 1st coordination sphere
    cyclicsystem = oncycle_atoms + offcycle_atoms

    # initialize ringVBO dictionary
    for atom in unassigned:
        if atom.label in oncycle_labels:
            ringVBO[atom] = 0

    # CSD doesn't do periodic boundary conditions, need a workaround
    # check for any periodic copies of cyclic atoms
    for atom in ringcopy.atoms:
        if all([(not atom in oncycle_atoms), (atom.label in oncycle_labels)]):
            if not atom in cyclic_periodic:
                cyclic_periodic.append(atom)
                cyclic_periodic_labels.append(atom.label)
    for atom in cyclic_periodic:
        # print (atom.neighbours)
        for neighbour in atom.neighbours:
            if not neighbour in (offcycle_periodic + cyclic_periodic):
                if not neighbour.label in (oncycle_labels):
                    offcycle_periodic.append(neighbour)

    # remove every atom that isn't part of or directly bound to a cycle
    for atom in ringcopy.atoms:
        if not atom in (cyclicsystem + cyclic_periodic):
            ringcopy.remove_atom(atom)

    # find all non-cyclic bonds
    # single bonds between cycles, break and cap with H
    for bond in ringcopy.bonds:
        if not bond.is_cyclic:
            # bonds between cycles
            if all(
                [
                    (all((member.label in oncycle_labels for member in bond.atoms))),
                    (
                        all(
                            (
                                not member.label in cyclic_periodic_labels
                                for member in bond.atoms
                            )
                        )
                    ),
                ]
            ):
                member1 = bond.atoms[0]
                member2 = bond.atoms[1]
                Hcap1 = molecule.Atom("H", coordinates=member1.coordinates)
                Hcap2 = molecule.Atom("H", coordinates=member2.coordinates)
                Hcap1_id = ringcopy.add_atom(Hcap1)
                Hcap2_id = ringcopy.add_atom(Hcap2)
                ringcopy.add_bond(bond.bond_type, Hcap1_id, member2)
                ringcopy.add_bond(bond.bond_type, Hcap2_id, member1)
                ringcopy.remove_bond(bond)

    # cap off-cycle atoms
    for offatom in offcycle_atoms + offcycle_periodic:
        # get the VBO for each off-cycle atom
        # (VBO with respect to cyclic atoms)
        offVBO = 0

        # quick check for delocalized systems in the ring
        # if there are any, get the delocalised bond orders
        if any(bond.bond_type == "Delocalised" for bond in offatom.bonds):
            offdVBO = delocalisedLBO(ringcopy)
        # add the non-delocalized bond orders
        for bond in offatom.bonds:
            # only interested in bonds to cyclic atoms
            if any(batom.label in oncycle_labels for batom in bond.atoms):
                # Each bond contributes to Ligand Bond Order according to its type
                if bond.bond_type == "Single":
                    offVBO += 1
                elif bond.bond_type == "Double":
                    offVBO += 2
                elif bond.bond_type == "Triple":
                    offVBO += 3
                elif bond.bond_type == "Quadruple":
                    offVBO += 4
                elif bond.bond_type == "Delocalised":
                    offVBO += offdVBO[offatom]
                elif bond.bond_type == "Aromatic":
                    offVBO += 0
                    print("impossible Aromatic bond")
        # cap with appropriate element for VBO
        if offVBO == 1:
            offatom.atomic_symbol = "H"
        elif offVBO == 2:
            offatom.atomic_symbol = "O"
        elif offVBO == 3:
            offatom.atomic_symbol = "N"
        elif offVBO == 4:
            offatom.atomic_symbol = "C"
        elif offVBO == 5:
            offatom.atomic_symbol = "P"
        elif offVBO == 6:
            offatom.atomic_symbol = "S"
        elif offVBO > 6:
            print("no, that's too many")

    # for each cyclic system, reassign bonds, kekulize, and get VBO
    # the bond and atom pruning we did above ensures that fused cycles
    # will be treated as a single system
    # while non-fused cycles that are connected via bonding are treated
    # as seperate systems
    for cyclesys in ringcopy.components:
        # reassign bonds and kekulize
        cyclesys.assign_bond_types()
        cyclesys.kekulize()

        # porhpyrins and similar molecules are misassigned, we will code a hard fix
        # first identify and isolate the inner porphyrin(/like) atoms
        # (these atoms determine overall charge)
        # store these in a dictionary for later
        joining_atoms = dict()
        joining_rings = dict()
        subring_labels = dict()
        porphyrinatoms = dict()
        porphyrin_to_correct = set()
        # ring by ring
        for subring in cyclesys.rings:
            subring_labels[subring] = []
            # get a list of all atom labels in each subring
            for sratom in subring.atoms:
                subring_labels[subring].append(sratom.label)
            # check atom by atom
            for sratom in subring.atoms:
                # check each atom neighbour
                for srneighbour in sratom.neighbours:
                    srn_label = srneighbour.label
                    # if the neighbour is not part of the current ring
                    if not srn_label in subring_labels[subring]:
                        # if the nighbour IS a cyclic atom
                        # consider this a "joining atom"
                        if srn_label in oncycle_labels:
                            try:
                                joining_atoms[srn_label].append(subring)
                            except KeyError:
                                joining_atoms[srn_label] = [subring]
                            try:
                                joining_rings[subring].append(srn_label)
                            except KeyError:
                                joining_rings[subring] = [srn_label]
        for jring in joining_rings:
            if all([(len(jring) == 16), (jring.is_fully_conjugated)]):
                for patom in jring.atoms:
                    plabel = patom.label
                    if not plabel in joining_atoms:
                        ncyclicbonds = 0
                        for pbond in patom.bonds:
                            if pbond.is_cyclic:
                                ncyclicbonds += 1
                        if ncyclicbonds == 2:
                            try:
                                porphyrinatoms[jring].append(patom)
                            except KeyError:
                                porphyrinatoms[jring] = [patom]
        for porph in porphyrinatoms:
            if all(i.atomic_symbol == "N" for i in porphyrinatoms[porph]):
                protonated = 0
                for patom in porphyrinatoms[porph]:
                    if any(i.atomic_symbol == "H" for i in patom.neighbours):
                        protonated += 1
                if protonated == 0:
                    for patom in porphyrinatoms[porph]:
                        porphyrin_to_correct.add(patom.label)

        # quick check for delocalized systems in the ring
        # if there are any, get the delocalised bond orders
        if any(bond.bond_type == "Delocalised" for bond in cyclesys.bonds):
            rdVBO = delocalisedLBO(cyclesys)

        # assign VBO for each on-cycle atom
        for ratom in cyclesys.atoms:
            rVBO = 0
            if ratom.label in oncycle_labels:
                if ratom.label in porphyrin_to_correct:
                    rVBO -= 0.5
                for rbond in ratom.bonds:
                    # Each bond contributes to Ligand Bond Order
                    # according to its type except periodic copies
                    if any(
                        [
                            (rbond.is_cyclic),
                            (
                                not all(
                                    (
                                        mem.label in cyclic_periodic_labels
                                        for mem in rbond.atoms
                                    )
                                )
                            ),
                        ]
                    ):
                        if rbond.bond_type == "Single":
                            rVBO += 1
                        elif rbond.bond_type == "Double":
                            rVBO += 2
                        elif rbond.bond_type == "Triple":
                            rVBO += 3
                        elif rbond.bond_type == "Quadruple":
                            rVBO += 4
                        elif rbond.bond_type == "Delocalised":
                            rVBO += rdVBO[ratom]
                        elif rbond.bond_type == "Aromatic":
                            rVBO += 0
                            print("impossible Aromatic bond")

                # the VBOs are currently associated to atom objects
                # in molecule objects that we have modified
                # we need these to be associated to atom objects in
                # the parent (unmodified) molecule object
                for matom in unassigned:
                    if matom.label == ratom.label:
                        ringVBO[matom] += rVBO
                        # unassigned.remove(matom)
    return ringVBO


def assign_VBS(atom: Atom, rVBO: dict[int, int], dVBO: dict[int, float]) -> int:
    """
    Assigns a Valence-Bond-Sum (VBS) to an atom.

    Parameters:
        atom (ccdc.molecule.Atom): Atom object.
        rVBO (dict[int, int]): dictionary with each atom's index in mole.atoms
                               as keys and VBO (valence bond order) as values.
        dVBO (dict[int, float]): dictionary with delocalized bond-possessing
                                 atom's index in mole.atoms as keys and their
                                 corresponding (delocalized-only) VBS.

    Returns:
        VBO (int): valence bond sum value.
    """
    VBO = 0
    if atom.is_metal:
        return 0
    if atom in rVBO:
        VBO = rVBO[atom]
    else:
        for bond in atom.bonds:
            if any(batom.is_metal for batom in bond.atoms):
                VBO += 0
            # Each bond contributes to Ligand Bond Order according to its type
            elif bond.bond_type == "Single":
                VBO += 1
            elif bond.bond_type == "Double":
                VBO += 2
            elif bond.bond_type == "Triple":
                VBO += 3
            elif bond.bond_type == "Quadruple":
                VBO += 4
            elif bond.bond_type == "Delocalised":
                VBO += dVBO[atom]
            elif bond.bond_type == "Aromatic":
                VBO += dVBO[atom]
    return VBO


def delocalisedLBO(molecule: Molecule) -> dict[int, float]:
    """
    Writes a dictionary of all atoms in the molecule with delocalized bonds
    and their (delocalized-only) valence bond sum (VBS).

    Parameters:
        molecule (ccdc.molecule.Molecule): Molecule object.

    Returns:
        delocal_dict (dict[int, float]): dictionary with delocalized bond-possessing
                                        atom's index in mole.atoms as keys and their
                                        corresponding (delocalized-only) VBS.
    """

    def TerminusCounter(atomlist: list[Atom]) -> int:
        """
        Counts the number of termini in the input delocalized bond system.

        Parameters:
            atomlist (list[ccdc.molecule.Atom]): list of atoms in delocalised system.

        Returns:
            NTerminus (int): number of termini in delocalized bond system.
        """
        NTerminus = 0
        for member in atomlist:
            connectivity = 0
            for bond in member.bonds:
                if bond.bond_type == "Delocalised":
                    connectivity += 1
            if connectivity == 1:
                NTerminus += 1
        return NTerminus

    def delocal_crawl(atomlist: list[Atom]) -> list[Atom]:
        """
        Recursively searches for atoms in delocalised bond systems starting from
        an input list containing at least one delocalised bonding atom.

        Parameters:
            atomlist (list[ccdc.molecule.Atom)]: list of atoms in delocalised system.

        Returns:
            atomlist (list[ccdc.molecule.Atom]): modified list of atoms in
                                                 delocalised system.
        """
        for delocatom in atomlist:
            for bond in delocatom.bonds:
                if bond.bond_type == "Delocalised":
                    for member in bond.atoms:
                        if not member in atomlist:
                            atomlist.append(member)
                            return delocal_crawl(atomlist)
        return atomlist

    delocal_dict = {}
    for atom in molecule.atoms:
        if all(
            [
                (any(bond.bond_type == "Delocalised" for bond in atom.bonds)),
                (not atom in delocal_dict),
            ]
        ):
            delocal_dict[atom] = []
            delocal_system = delocal_crawl([atom])
            NTerminus = TerminusCounter(delocal_system)
            for datom in delocal_system:
                connectivity = 0
                delocLBO = 0
                for neighbour in datom.neighbours:
                    if neighbour in delocal_system:
                        connectivity += 1
                if connectivity == 1:
                    # terminus
                    delocLBO = (NTerminus + 1) / NTerminus
                if connectivity > 1:
                    # node
                    delocLBO = (connectivity + 1) / connectivity
                delocal_dict[datom] = delocLBO
    return delocal_dict


def iVBS_FormalCharge(atom: Atom) -> int:
    """
    Determines the formal charge of an atom NOT involved in any aromatic or
    delocalized bonding system.

    Parameters:
        atom (ccdc.molecule.Atom): Atom object

    Returns:
        charge (int): formal charge of the input atom.
    """
    VBO = 0
    if atom.is_metal:
        return VBO
    CN = 0
    for neighbour in atom.neighbours:
        if not neighbour.is_metal:
            CN += 1
    valence = valence_e(atom)
    charge = 0
    for bond in atom.bonds:
        if any(batom.is_metal for batom in bond.atoms):
            VBO += 0
        # Each bond contributes to Ligand Bond Order according to its type
        elif bond.bond_type == "Single":
            VBO += 1
        elif bond.bond_type == "Double":
            VBO += 2
        elif bond.bond_type == "Triple":
            VBO += 3
        elif bond.bond_type == "Quadruple":
            VBO += 4
    # need the unpaired electrons
    unpaired_e = 4 - abs(4 - valence)
    # expanded valences require special handling
    if VBO <= (unpaired_e):
        charge = VBO - unpaired_e
    # Expanded (2e) valences:
    elif (VBO > unpaired_e) and (VBO < valence):
        diff = VBO - unpaired_e
        if diff <= 2:
            UPE = valence - unpaired_e - 2
        elif diff <= 4:
            UPE = valence - unpaired_e - 4
        elif diff <= 6:
            UPE = valence - unpaired_e - 6
        elif diff <= 8:
            UPE = valence - unpaired_e - 8
        charge = valence - (VBO + UPE)
    elif VBO >= (valence):
        charge = valence - VBO
    return charge


def get_CN(atom: Atom) -> int:
    """
    Determines the coordination number of the input atom.

    Parameters:
        atom (ccdc.molecule.Atom): Atom object.

    Returns:
        coord_number (int): Atom's coordination number.
    """
    CN = 0
    for neighbour in atom.neighbours:
        if not neighbour.is_metal:
            CN += 1
    return CN


def valence_e(elmnt: Atom) -> int:
    """
    Determines the number of valence electrons of an atom/element.

    Parameters:
        elmnt (ccdc.molecule.Atom): Atom object.

    Returns:
        valence (int): Atom's valence electron count.
    """
    atom = mendeleev.element(elmnt.atomic_symbol)
    if atom.block == "s":
        valence = atom.group_id
    if atom.block == "p":
        valence = atom.group_id - 10
    if atom.block == "d":
        valence = atom.group_id
    if atom.block == "f":
        if atom.atomic_number in range(56, 72):
            valence = atom.atomic_number - 57 + 3
        elif atom.atomic_number in range(88, 104):
            valence = atom.atomic_number - 89 + 3
    if atom.group_id == 18:
        valence = 8
    if atom.symbol == "He":
        valence = 2
    return valence


def carbocation_check(atom: Atom) -> Literal["tetrahedral", "trigonal"]:
    """
    Check carbocation/carbanion geometry according to bond angles.

    Parameters:
        atom (ccdc.molecule.Atom): Atom object.

    Returns:
        Literal["tetrahedral", "trigonal"]: geometry at input atom.
    """
    abc = []
    # get atom neighbours
    for neighbours in atom.neighbours:
        if not neighbours.is_metal:
            abc.append(neighbours)
    # get all three relevant bond angles
    angle1 = descriptors.MolecularDescriptors.atom_angle(abc[0], atom, abc[1])
    angle2 = descriptors.MolecularDescriptors.atom_angle(abc[0], atom, abc[2])
    angle3 = descriptors.MolecularDescriptors.atom_angle(abc[1], atom, abc[2])
    # average the angels
    AVGangle = abs(angle1 + angle2 + angle3) / 3
    # take the difference between the averaged bond angles and
    # ideal trigonal planar/tetrahedral bond angles
    tet = abs(AVGangle - 109.5)
    trig = abs(AVGangle - 120)
    if tet < trig:
        return "tetrahedral"
    if trig < tet:
        return "trigonal"


def carbene_type(atom: Atom) -> Literal["singlet", "triplet"]:
    """
    Distinguishes between singlet and triplet carbenes.

    Parameters:
        atom (ccdc.molecule.Atom): Atom object(s) suspected of belonging to a
                                   carbene (2-coordinate carbon II).

    Returns:
        Literal["singlet", "triplet"]: carbene type at input atom.
    """
    # get alpha-atoms
    alpha = atom.neighbours
    alpha_type = []
    # get element symbols for alpha atoms
    for a in alpha:
        if not a.is_metal:
            alpha_type.append(a.atomic_symbol)
    # if any alpha atom is a heteroatom, return "singlet"
    # these are Fischer carbenes
    for a in alpha_type:
        if not any([(a == "C"), (a == "H")]):
            return "singlet"
    # if the carbene C is in a heterocycle,
    # return "singlet"
    # there are Arduengo carbenes (NHCs, CAACs)
    if atom.is_cyclic == True:
        for ring in atom.rings:
            for species in ring.atoms:
                if not species.atomic_symbol == "C":
                    return "singlet"
    # for all other carbenes, return "triplet"
    # these are Schrock carbenes
    return "triplet"


def hapticity(atom: Atom, metalsite: list[Atom]) -> bool:
    """
    Determines if a ligand binding site possesses hapticity (any n-hapto).

    Parameters:
        atom (ccdc.molecule.Atom): Atom object.
        metalsites (list[ccdc.molecule.Atom]): list of metal sites in the structure
                                               that belong to the asymmetric unit.

    Returns:
        bool: whether the the input ligand is hapto-.
    """
    for atom2 in atom.neighbours:
        if not atom2.is_metal:
            if any(n2.label == metalsite.label for n2 in atom2.neighbours):
                return True
    return False


def bridging(atom: Atom) -> int:
    """
    Determines how many metal atoms the input atom binds to search for
    bridging sites.

    Parameters:
        atom (ccdc.molecule.Atom): binding site Atom object.

    Returns:
       bridge (int): number of metal atoms bound to the atom.
    """
    bridge = 0
    for n in atom.neighbours:
        if n.is_metal:
            bridge += 1
    return bridge


def iVBS_Oxidation_Contrib(
    unique_atoms: list[Atom], rVBO: dict[int, int], dVBO: dict[int, float]
) -> dict[Atom, float]:
    """
    Determines the oxidation state contribution of all unique atoms.

    Parameters:
        unique_atoms (list[ccdc.molecule.Atom]): unique atoms belonging to the
                                                 asymmetric unit.
        rVBO (dict[int, int]): dictionary with each atom's index in mole.atoms
                               as keys and VBO (valence bond order) as values.
        dVBO (dict[int, float]): dictionary with delocalized bond-possessing atom's
                                index in mole.atoms as keys and their corresponding
                                (delocalized-only) VBS.

    Returns:
        oxi_contrib (dict[ccdc.molecule.Atom, float)]: dictionary with Atom object
                         as keys and their oxidation state contribution as values.
    """
    VBS = 0
    CN = 0
    valence = 0
    oxi_contrib = {}
    # for each unique atom
    for atom in unique_atoms:
        # assign valence-bond-sum
        VBS = assign_VBS(atom, rVBO, dVBO)
        # determine coordination number
        CN = get_CN(atom)
        #  determine number of valence electrons
        valence = valence_e(atom)
        # get number of unpaired electrons in the free element
        unpaired_e = 4 - abs(4 - valence)

        #  metals do not contribute:
        if atom.is_metal:
            oxi_contrib[atom] = 0
        # Normal valences:
        elif VBS <= (unpaired_e):
            oxi_contrib[atom] = unpaired_e - VBS
        # Expanded (2e) valences:
        elif (VBS > unpaired_e) and (VBS < valence):
            diff = VBS - unpaired_e
            if diff <= 2:
                UPE = valence - unpaired_e - 2
            elif diff <= 4:
                UPE = valence - unpaired_e - 4
            elif diff <= 6:
                UPE = valence - unpaired_e - 6
            elif diff <= 8:
                UPE = valence - unpaired_e - 8
            oxi_contrib[atom] = VBS + UPE - valence
        elif VBS >= (valence):
            oxi_contrib[atom] = VBS - valence

        # need to check for 3-coordinate carbocations,
        # 3-coordinate carbanions, carbenes, and heavier
        # homologues (these are not immediately detectable)
        if any(
            [
                (atom.atomic_symbol == "C"),
                (atom.atomic_symbol == "Si"),
                (atom.atomic_symbol == "Ge"),
                (atom.atomic_symbol == "Pb"),
            ]
        ):
            if not atom in rVBO:
                # 3 coordinate and VBS 3 could be
                # carbanion or carbocation
                if VBS == 3 and CN == 3:
                    geom = carbocation_check(atom)
                    if geom == "trigonal":
                        oxi_contrib[atom] = -1
                    if geom == "tetrahedral":
                        oxi_contrib[atom] = 1
            # VBS 2 and 2 coordinate is carbene,
            # but singlet or triplet?
            if VBS == 2 and CN == 2:
                carbene = carbene_type(atom)
                if carbene == "singlet":
                    oxi_contrib[atom] = 0
                if carbene == "triplet":
                    oxi_contrib[atom] = 2

        # Nitro groups frequently have both N-O bonds assigned
        # as double bonds, giving incorrect VBS of 5
        # and oxidation contribution of -2
        # this block catches this and applies a fix
        if all(
            [
                (atom.atomic_symbol == "N"),
                (VBS == 5 and CN == 3),
            ]
        ):
            N_sphere1 = atom.neighbours
            O_count = 0
            for neighbour in N_sphere1:
                if neighbour.atomic_symbol == "O":
                    O_count += 1
            geom = carbocation_check(atom)
            if O_count == 2 and geom == "trigonal":
                oxi_contrib[atom] = 0

    return oxi_contrib


def redundantAON(AON: dict[Atom, float], molecule: Molecule) -> dict[Atom, float]:
    """
    Maps the oxidation contributions of unique atom sites to the redundant atom
    sites according to their shared atom labels.

    Parameters:
        AON (dict[ccdc.molecule.Atom, float]): dictionary with Atom object as keys
                        and their oxidation state contribution as values for unique
                        Atom objects.
        molecule (ccdc.molecule.Molecule): Molecule object.

    Returns:
        redAON (dict[ccdc.molecule.Atom, float]): dictionary with Atom object as keys
                        and their oxidation state contribution as values for all
                        (including redundant) Atom objects.
    """
    redAON = {}
    for rsite1 in molecule.atoms:
        for usite1 in AON:
            redAON[usite1] = AON[usite1]
            if rsite1.label == usite1.label:
                redAON[rsite1] = AON[usite1]
    return redAON


def binding_domain(
    binding_sites: list[Atom],
    AON: dict[Atom, float],
    molecule: Molecule,
    usites: list[Atom],
) -> dict[Atom, list[Atom]]:
    """
    Builds bonding domains within the crystal structure to determine which
    metal binding sites (Atom objects directly bonded to a metal) are connected
    via conjugation. Function accounts for the inconsistent assignment of
    delocalized bonds, by using the bonding domains (see methodology section
    for details on the implementation and validation).

    Parameters:
        binding_sites (list[ccdc.molecule.Atom]): list of binding sites connecting
                                                  metal atoms and ligands.
        AON (dict[ccdc.molecule.Atom, float]): dictionary with Atom object as keys
                        and their oxidation state contribution as values for unique
                        Atom objects.
        molecule (ccdc.molecule.Molecule): Molecule object.
        uniquesites (list[ccdc.molecule.Atom]): list of unique atoms.

    Returns:
        sitedomain (dict[ccdc.molecule.Atom, list[ccdc.molecule.Atom]]):
                        dictionary with Atom object as keys and a list of Atoms
                        connected through bonding that form a binding domain
                        as values.
    """

    def arom_domains(
        site: Atom, usites: list[Atom], aromlist: list[Atom], bondset: list[Bond]
    ) -> list[Atom]:
        """
        Recursively generate aromatic binding domains.
        """
        for bond in site.bonds:
            bondset.add(bond)
        for bond in bondset:
            for member in bond.atoms:
                if all(
                    [
                        (not member in aromlist),
                        (not member.is_metal),
                        (any(mbond.bond_type == "Aromatic" for mbond in member.bonds)),
                    ]
                ):
                    aromlist.append(member)
                    for mbond in member.bonds:
                        bondset.add(mbond)
                    return arom_domains(site, usites, aromlist, bondset)
        # aromlist currently contains non-unique instances of atoms
        # this will cause problems further down the line, so correct
        for index, member in enumerate(aromlist):
            aromlist[index] = usites[member.label]
        return aromlist

    def deloc_domains(
        site: Atom,
        usites: list[Atom],
        AON: dict[Atom, float],
        molecule: Molecule,
        deloclist: list[Atom],
        bondset: list[Bond],
        checked_bonds: list[Bond],
    ) -> list[Atom]:
        """
        Recursively generate delocalised binding domains.
        """
        for bond in site.bonds:
            if not bond in bondset:
                bondset.add(bond)
        for bond in bondset:
            if not bond in checked_bonds:
                for member in bond.atoms:
                    if all(
                        [
                            (not member in deloclist),
                            (not member.is_metal),
                            (
                                not any(
                                    mbond.bond_type == "Aromatic"
                                    for mbond in member.bonds
                                )
                            ),
                            (
                                any(
                                    [
                                        (
                                            len(
                                                molecule.shortest_path_bonds(
                                                    site, member
                                                )
                                            )
                                            <= 2
                                        ),
                                        (bond.bond_type == "Delocalised"),
                                        (bond.is_conjugated),
                                        (
                                            all(
                                                [
                                                    (bond.bond_type == "Single"),
                                                    (not AON[member] == 0),
                                                ]
                                            )
                                        ),
                                        (
                                            all(
                                                [
                                                    (
                                                        not any(
                                                            mbond.bond_type == "Single"
                                                            for mbond in member.bonds
                                                        )
                                                    ),
                                                    (
                                                        not any(
                                                            mbond.bond_type
                                                            == "Aromatic"
                                                            for mbond in member.bonds
                                                        )
                                                    ),
                                                    (
                                                        not any(
                                                            mbond.bond_type
                                                            == "Delocalised"
                                                            for mbond in member.bonds
                                                        )
                                                    ),
                                                ]
                                            )
                                        ),
                                    ]
                                )
                            ),
                        ]
                    ):
                        deloclist.append(member)
                        for mbond in member.bonds:
                            bondset.add(mbond)
                checked_bonds.add(bond)
                return deloc_domains(
                    site, usites, AON, molecule, deloclist, bondset, checked_bonds
                )
        # deloclist currently contains non-unique instances of atoms
        # this will cause problems further down the line, so correct
        for index, member in enumerate(deloclist):
            deloclist[index] = usites[member.label]
        return deloclist

    sitedomain = {}
    for site in binding_sites:
        if not site.is_metal == True:
            if any(sbond.bond_type == "Aromatic" for sbond in site.bonds):
                sitedomain[site] = arom_domains(
                    site, usites, aromlist=[site], bondset=set()
                )
            if not any(sbond.bond_type == "Aromatic" for sbond in site.bonds):
                sitedomain[site] = deloc_domains(
                    site,
                    usites,
                    AON,
                    molecule,
                    deloclist=[site],
                    bondset=set(),
                    checked_bonds=set(),
                )

    for site in sitedomain:
        olapset = set()
        for site2 in sitedomain:
            for member in sitedomain[site]:
                if member in sitedomain[site2]:
                    olapset.add(site2)
        for olap in olapset:
            sitedomain[site] = list(set(sitedomain[site]) | set(sitedomain[olap]))
            sitedomain[olap] = sitedomain[site]
    return sitedomain


def binding_contrib(
    binding_sphere: dict[Atom, list[Atom]],
    binding_sites: list[Atom],
    AON: dict[Atom, float],
) -> dict[Atom, float]:
    """
    Redistributes oxidation state contributions within a binding domain.
    Equal distribution is assumed across connected binding sites in each domain.

    Parameters:
        binding_sphere (dict[ccdc.molecule.Atom, list[ccdc.molecule.Atom]]):
                        dictionary with Atom object as keys and a list of Atoms
                        connected through bonding that form a binding domain as values.
        binding_sites (list[ccdc.molecule.Atom]): list of binding sites connecting
                                                  metal atoms and ligands.
        AON (dict[ccdc.molecule.Atom, float]): dictionary with Atom object as keys
                        and their oxidation state contribution as values for unique Atoms.

    Returns:
        site_contrib (dict[ccdc.molecule.Atom, float]): dictionary with Atom object as keys
                        and their updated oxidation state contribution as values accounting
                        for distribution within the binding domain.
    """
    site_contrib = {}
    for site in binding_sphere:
        site_contrib[site] = 0
        nbinding = 0
        for member in binding_sphere[site]:
            if member in binding_sites:
                nbinding += 1
            site_contrib[site] += AON[member]
        site_contrib[site] /= nbinding
    return site_contrib


def outer_sphere_domain(
    uniquesites: list[Atom], binding_domains: dict[Atom, list[Atom]]
) -> list[Atom]:
    """
    Identifies sites outside of the binding domains which must be checked for
    outer sphere charge contributions.

    Parameters:
        uniquesites (list[ccdc.molecule.Atom]): list of unique atoms in the structure
                                                belonging to the asymmetric unit.
        binding_domains (dict[ccdc.molecule.Atom, list[ccdc.molecule.Atom]]):
                        dictionary with Atom object as keys and a list of Atoms
                        connected through bonding that form a binding domain as values.

    Returns:
        outer_sphere (list[ccdc.molecule.Atom]): list of unique, non-metal atoms
                                                 outside of binding domains.
    """
    outer_sphere = []
    for site in uniquesites:
        if all(
            [
                (
                    not any(
                        site in binding_domains[domain] for domain in binding_domains
                    )
                ),
                (not site.is_metal),
            ]
        ):
            outer_sphere.append(site)
    return outer_sphere


def outer_sphere_contrib(outer_sphere: list[Atom], AON: dict[Atom, float]) -> int:
    """
    Calculates the total oxidation state contribution of the outer sphere atoms as
    the sum of their formal charge/contributions.

    Parameters:
        outer_sphere (list[ccdc.molecule.Atom]): list of unique, non-metal atoms
                                                 outside of binding domains.
        AON (dict[ccdc.molecule.Atom, float]): dictionary with Atom object as keys
                        and their oxidation state contribution as values for unique Atoms.

    Returns:
        contrib (int): sum of outer sphere charge contributions.
    """
    contrib = 0
    for site in outer_sphere:
        contrib += AON[site]
    return contrib


def get_metal_networks(
    ligand_sites: dict[Atom, list[Atom]],
    binding_sphere: dict[Atom, list[Atom]],
    bindingAON: dict[Atom, float],
) -> dict[Atom, list[Atom]]:
    """
    Determines the metal atoms that are connected through binding domains and
    charged ligands. Any connections through neutral ligands are ignored as they
    do not contribute to the charge accounting.

    Parameters:
        ligand_sites (dict[ccdc.molecule.Atom, list[ccdc.molecule.Atom]]):
                            dictionary with metal Atom object as key and the
                            list of ligand atoms which bind them as values.
        binding_sphere (dict[ccdc.molecule.Atom, list[ccdc.molecule.Atom]]):
                            dictionary with Atom object as keys and a list of Atoms
                            connected through bonding that form a binding domain
                            as values.
        bindingAON (dict[ccdc.molecule.Atom, float]): dictionary with Atom object as
                            keys and their updated oxidation state contribution as
                            values accounting for distribution within the binding domain.

    Returns:
        network_dict (dict[ccdc.molecule.Atom, list[ccdc.molecule.Atom]]):
                            dictionary with as metal Atom objects as keys and a list
                            of other metal Atom objects connected through binding
                            domains/charged ligands as values. Ignores neutral ligand
                            connections.
    """

    def network_crawl(
        ligand_sites: dict[Atom, list[Atom]],
        binding_sphere: dict[Atom, list[Atom]],
        bindingAON: dict[Atom, float],
        metal_networks: list[Atom],
        checked_sites: list[Atom],
        group: list[Atom],
    ) -> list[Atom]:
        """
        Recursively crawl through bonds to identify metals connected through direct bondings
        or delocalised/conjugated systems.
        """
        for metal in group:
            # This block will find all metals connected to an input metal by metal-metal bonds
            checked_sites.append(metal)
            for neighbour in metal.neighbours:
                if neighbour.is_metal:
                    if not neighbour in checked_sites:
                        checked_sites.append(neighbour)
                        for site in ligand_sites:
                            if neighbour.label == site.label:
                                if not site in group:
                                    group.append(site)
                        return network_crawl(
                            ligand_sites,
                            binding_sphere,
                            bindingAON,
                            metal_networks,
                            checked_sites,
                            group,
                        )
            # this block will find all metals connected to an input metal by
            # conjugation and delocalized charge ligands
            # metals connected through NEUTRAL ligands will be ignored
            for site in ligand_sites[metal]:
                if all([(not bindingAON[site] == 0), (not site in checked_sites)]):
                    for dsite in binding_sphere[site]:
                        if all(
                            [(not dsite in checked_sites), (dsite in binding_sphere)]
                        ):
                            checked_sites.append(dsite)
                            for environ in dsite.neighbours:
                                if environ.is_metal:
                                    if environ in ligand_sites:
                                        if all(
                                            [
                                                (
                                                    all(
                                                        not environ in network
                                                        for network in metal_networks
                                                    )
                                                ),
                                                (not environ in group),
                                            ]
                                        ):
                                            group.append(environ)
                                    else:
                                        for umetal in ligand_sites:
                                            if all(
                                                [
                                                    (umetal.label == environ.label),
                                                    (
                                                        all(
                                                            not umetal in network
                                                            for network in metal_networks
                                                        )
                                                    ),
                                                    (not umetal in group),
                                                ]
                                            ):
                                                group.append(umetal)
                    return network_crawl(
                        ligand_sites,
                        binding_sphere,
                        bindingAON,
                        metal_networks,
                        checked_sites,
                        group,
                    )
        return group

    metal_networks = []
    for metal in ligand_sites:
        if all(not metal in network for network in metal_networks):
            metal_networks.append(
                network_crawl(
                    ligand_sites,
                    binding_sphere,
                    bindingAON,
                    metal_networks,
                    checked_sites=[],
                    group=[metal],
                )
            )

    network_dict = {}
    for network in metal_networks:
        for metal in network:
            network_dict[metal] = network
    return network_dict


def distribute_ONEC(
    sONEC: dict[Atom, list[float, float]],
    metal_networks: dict[Atom, list[Atom]],
    IEs: dict[str, list[float]],
    ONP: dict[str, list[float]],
    highest_known_ON: dict[str, int],
    metal_CN: dict[Molecule, int],
    most_probable_ON: dict[str, int],
) -> dict[Atom, list[float, float]]:
    """
    Redistributes the oxidation state contributions across all metal atoms in
    the structure according to their metal networks (fully local distribution)
    & calculates their associated electron counts. Features utilizing electron
    counts are minimally implemented at this time.

    Parameters:
        sONEC (dict[ccdc.molecule.Atom, list[float, float]]): dictionary with
                        metal Atom object as keys and lists containing the initial
                        oxidation state and electron count implied by only the
                        equal splitting of binding domain charges as values.
        metal_networks (dict[ccdc.molecule.Atom, list[ccdc.molecule.Atom]]):
                        dictionary with as metal Atom objects as keys and a list of
                        other metal Atom objects connected through binding domains/
                        charged ligands as values. Ignores neutral ligand connections.
        IEs (dict[str, list[float]]): dictionary with metal element symbols as keys
                        and a list of their reported ionization energies as values.
        ONP (dict[str, list[float]]): dictionary with metal element symbols as keys
                        and a list of the probability at the relevant oxidation states
                        as values.
        highest_known_ON (dict[str, int]) : dictionary with metal element symbols as
                        keys and a their highest known oxidation state as values.
        metal_CN (dict[ccdc.molecule.Molecule, int]): dictionary with as metal Atom
                        objects as keys and their effective coordination number as values.
        most_probable_ON (dict[str, int]) : dictionary with metal element symbols as
                        keys and a their oxidation state with the highest probability
                        as values.

    Returns:
        distributed_ONEC (dict[ccdc.molecule.Atom, list[float, float]]): dictionary
                        with metal Atom object as keys and lists containing their
                        redistributed oxidation state and electron count as values.
    """

    def recursive_distributor_single_network(
        iONEC: dict[Atom, list[float, float]],
        available_charge: int,
        sorted_metals: dict[str, Atom],
        IEs: dict[str, list[float]],
        ONP: dict[str, list[float]],
        highest_known_ON: dict[str, int],
    ) -> dict[Atom, list[float, float]]:
        """
        Distribute network charge according to ionization energy and probability
        until all charge is distributed. Performed after tallying available
        network charge and sorting network metals by element type.
        """
        # initialize working dictionary
        dONEC = {}
        dONEC = dict(iONEC)

        # positive contribution?
        if available_charge > 0:
            # get list of improbable and improbable next oxidations
            prob_metal_type = []
            improb_metal_type = []
            for metal_type in sorted_metals:
                try:
                    prob = float(
                        100
                        * ONP[metal_type][
                            math.floor(dONEC[sorted_metals[metal_type][0]][0]) + 1
                        ]
                    )
                except IndexError:
                    prob = 0
                if prob >= 1:
                    prob_metal_type.append(metal_type)
                else:
                    improb_metal_type.append(metal_type)

            # if only one metal type has a probable next oxidation state, do that
            if len(prob_metal_type) == 1:
                lowestMetal = prob_metal_type[0]

            # if more than one metal type has a probable next oxidation state,
            # determine next lowest ionization energy among probable next
            # oxidation states
            elif len(prob_metal_type) > 1:
                # find lowest next ionization energy
                for metal_type in prob_metal_type:
                    # metal in a negative oxidation state? Use IE = 0.
                    if dONEC[sorted_metals[metal_type][0]][0] < 0:
                        currentIE = 0
                    # metal oxidation state at or higher than highest known? Set IE arbitrarily high.
                    elif (
                        dONEC[sorted_metals[metal_type][0]][0]
                        >= highest_known_ON[metal_type]
                    ):
                        currentIE = 9999
                    # otherwise, use the appropriate IE.
                    else:
                        currentIE = float(
                            IEs[metal_type][
                                math.floor(dONEC[sorted_metals[metal_type][0]][0])
                            ]
                        )
                    if not "lowestIE" in locals():
                        lowestIE = currentIE
                        lowestMetal = metal_type
                    else:
                        if currentIE < lowestIE:
                            lowestIE = currentIE
                            lowestMetal = metal_type

            # if there is no probable next oxidation state available,
            # determine lowest ionization energy among improbable next oxidation states
            elif len(prob_metal_type) == 0:
                # find lowest next ionization energy
                for metal_type in improb_metal_type:
                    # metal in a negative oxidation state? Use IE = 0.
                    if dONEC[sorted_metals[metal_type][0]][0] < 0:
                        currentIE = 0
                    # metal oxidation state at or higher than highest known? Set IE arbitrarily high.
                    elif (
                        dONEC[sorted_metals[metal_type][0]][0]
                        >= highest_known_ON[metal_type]
                    ):
                        currentIE = 9999
                    # otherwise, use the appropriate IE.
                    else:
                        currentIE = float(
                            IEs[metal_type][
                                math.floor(dONEC[sorted_metals[metal_type][0]][0])
                            ]
                        )
                    if not "lowestIE" in locals():
                        lowestIE = currentIE
                        lowestMetal = metal_type
                    else:
                        if currentIE < lowestIE:
                            lowestIE = currentIE
                            lowestMetal = metal_type

            # distribute one ionization energy level worth of charge
            if available_charge >= len(sorted_metals[lowestMetal]):
                for metal in sorted_metals[lowestMetal]:
                    dONEC[metal][0] += 1
                    available_charge -= 1
            elif available_charge < len(sorted_metals[lowestMetal]):
                for metal in sorted_metals[lowestMetal]:
                    dONEC[metal][0] += available_charge / (
                        len(sorted_metals[lowestMetal])
                    )
                available_charge = 0

        # negative contribution?
        if available_charge < 0:
            # get list of improbable and improbable next oxidations
            prob_metal_type = []
            improb_metal_type = []
            for metal_type in sorted_metals:
                try:
                    prob = float(
                        100
                        * ONP[metal_type][
                            math.floor(dONEC[sorted_metals[metal_type][0]][0]) - 1
                        ]
                    )
                except IndexError:
                    prob = 0
                if prob >= 1:
                    prob_metal_type.append(metal_type)
                else:
                    improb_metal_type.append(metal_type)

            # if only one metal type has a probable next oxidation state, do that
            if len(prob_metal_type) == 1:
                highestMetal = prob_metal_type[0]

            # if more than one metal type has a probable next oxidation state,
            # determine next highest ionization energy among probable next
            # oxidation states
            elif len(prob_metal_type) > 1:
                for metal_type in prob_metal_type:
                    # metal in a negative oxidation state? Use IE = 0.
                    if dONEC[sorted_metals[metal_type][0]][0] <= 0:
                        currentIE = 0
                    # metal oxidation state higher than highest known? Set IE arbitrarily high.
                    elif (
                        dONEC[sorted_metals[metal_type][0]][0]
                        > highest_known_ON[metal_type]
                    ):
                        currentIE = 9999
                    # otherwise, use the appropriate IE.
                    else:
                        if not any(
                            [
                                (
                                    math.isclose(
                                        (dONEC[sorted_metals[metal_type][0]][0] % 1),
                                        0,
                                        abs_tol=0.0001,
                                    )
                                ),
                                (
                                    math.isclose(
                                        (dONEC[sorted_metals[metal_type][0]][0] % 1),
                                        1,
                                        abs_tol=0.0001,
                                    )
                                ),
                            ]
                        ):
                            currentIE = float(
                                IEs[metal_type][
                                    math.floor(dONEC[sorted_metals[metal_type][0]][0])
                                ]
                            )
                        else:
                            currentIE = float(
                                IEs[metal_type][
                                    math.floor(dONEC[sorted_metals[metal_type][0]][0])
                                    - 1
                                ]
                            )
                    if not "highestIE" in locals():
                        highestIE = currentIE
                        highestMetal = metal_type
                    else:
                        if currentIE > highestIE:
                            highestIE = currentIE
                            highestMetal = metal_type

            # if no probable next oxidation states are available,
            # determine next highest ionization energy among probable next
            # oxidation states
            elif len(improb_metal_type) > 0:
                for metal_type in improb_metal_type:
                    # metal in a negative oxidation state? Use IE = 0.
                    if dONEC[sorted_metals[metal_type][0]][0] <= 0:
                        currentIE = 0
                    # metal oxidation state higher than highest known? Set IE arbitrarily high.
                    elif (
                        dONEC[sorted_metals[metal_type][0]][0]
                        > highest_known_ON[metal_type]
                    ):
                        currentIE = 9999
                    # otherwise, use the appropriate IE.
                    else:
                        if not any(
                            [
                                (
                                    math.isclose(
                                        (dONEC[sorted_metals[metal_type][0]][0] % 1),
                                        0,
                                        abs_tol=0.0001,
                                    )
                                ),
                                (
                                    math.isclose(
                                        (dONEC[sorted_metals[metal_type][0]][0] % 1),
                                        1,
                                        abs_tol=0.0001,
                                    )
                                ),
                            ]
                        ):
                            currentIE = float(
                                IEs[metal_type][
                                    math.floor(dONEC[sorted_metals[metal_type][0]][0])
                                ]
                            )
                        else:
                            currentIE = float(
                                IEs[metal_type][
                                    math.floor(dONEC[sorted_metals[metal_type][0]][0])
                                    - 1
                                ]
                            )
                    if not "highestIE" in locals():
                        highestIE = currentIE
                        highestMetal = metal_type
                    else:
                        if currentIE > highestIE:
                            highestIE = currentIE
                            highestMetal = metal_type

            # distribute one ionization energy level worth of charge
            if (-1 * available_charge) >= len(sorted_metals[highestMetal]):
                for metal in sorted_metals[highestMetal]:
                    dONEC[metal][0] -= 1
                    available_charge += 1
            elif (-1 * available_charge) < len(sorted_metals[highestMetal]):
                for metal in sorted_metals[highestMetal]:
                    dONEC[metal][0] += available_charge / (
                        len(sorted_metals[highestMetal])
                    )
                available_charge = 0

        # if all charge has been distributed, we're done, otherwise, roll again
        if available_charge == 0:
            return dONEC
        else:
            return recursive_distributor_single_network(
                dONEC, available_charge, sorted_metals, IEs, ONP, highest_known_ON
            )

    # operate on each network individually
    distributed_ONEC = {}
    for network in metal_networks:
        # sort metals by element type
        sorted_metals = {}
        for metal in metal_networks[network]:
            sorted_metals[metal.atomic_symbol] = []
        for metal in metal_networks[network]:
            sorted_metals[metal.atomic_symbol].append(metal)

        # tally up network charge to be distributed
        # and initialize metals to most probable ON
        # (adjust network charge accordingly)
        network_charge = 0
        for metal in metal_networks[network]:
            network_charge += sONEC[metal][0]
            distributed_ONEC[metal] = [most_probable_ON[metal.atomic_symbol]]
            network_charge -= most_probable_ON[metal.atomic_symbol]
            distributed_ONEC[metal].append(int(sONEC[metal][1]))

        # if the most probable oxidation distribution has already balanced the charge, we're done
        # if not, recursively distribute network charge according to ionization energy
        if not (math.isclose(network_charge, 0, abs_tol=0.0001)):
            distributed_ONEC = recursive_distributor_single_network(
                distributed_ONEC,
                network_charge,
                sorted_metals,
                IEs,
                ONP,
                highest_known_ON,
            )

        # finally, adjust electron count to new oxidation state (OiL RiG)
        for metal in metal_networks[network]:
            distributed_ONEC[metal][1] = (
                valence_e(metal) + (2 * metal_CN[metal]) - distributed_ONEC[metal][0]
            )

    return distributed_ONEC


def distribute_OuterSphere(
    sONEC: dict[Atom, list[float, float]],
    outer_sphere_charge: int,
    IEs: dict[Atom, list[Atom]],
    ONP: dict[str, list[float]],
    highest_known_ON: dict[str, int],
    metal_CN: dict[Molecule, int],
) -> dict[Atom, list[float, float]]:
    """
    Redistributes the oxidation state contributions across all metal atoms in
    the structure according to the outer sphere charge contribution (partially
    local distribution) & calculates their associated electron counts.
    Features utilizing electron counts are minimally implemented at this time.

    Parameters:
        sONEC (dict[ccdc.molecule.Atom, list[float, float]]): dictionary with
                        metal Atom object as keys and lists containing the initial
                        oxidation state and electron count implied by only the equal
                        splitting of binding domain charges as values.
        outer_sphere_charge (int): sum of outer sphere charge contributions.
        IEs (dict[str, list(float)]): dictionary with metal element symbols as keys
                        and a list of  their reported ionization energies as values.
        ONP (dict[str, list(float)]): dictionary with metal element symbols as keys
                        and a list of the probability at the relevant oxidation
                        states as values.
        highest_known_ON (dict[str, int]) : dictionary with metal element symbols
                        as keys and a their highest known oxidation state as values.
        metal_CN (dict[ccdc.molecule.Molecule, int]): dictionary with as metal Atom
                        objects as keys and their  effective coordination number
                        as values.

    Returns:
        distributed_ONEC (dict[ccdc.molecule.Atom, list[float, float]]): dictionary
                        with metal Atom object as keys and lists containing their
                        redistributed oxidation state and electron count as values.
    """

    def recursive_distributor(
        iONEC: dict[Atom, list[float, float]],
        available_charge: int,
        IEs: dict[str, list[float]],
        ONP: dict[str, list[float]],
        highest_known_ON: dict[str, int],
    ) -> dict[Atom, list[float, float]]:
        """
        Distribute network charge according to ionization energy and highest
        allowable oxidation state until all charge is distributed. Performed after
        tallying available network charge and sorting network metals by element type.
        """
        # initialize working dictionary
        dONEC = {}
        dONEC = dict(iONEC)

        # positive contribution?
        if available_charge > 0:
            # get list of probable and improbable next oxidations
            prob_metals = []
            improb_metals = []
            for metal in dONEC:
                try:
                    prob = float(
                        100 * ONP[metal.atomic_symbol][math.floor(dONEC[metal][0]) + 1]
                    )
                except IndexError:
                    prob = 0
                if prob >= 1:
                    prob_metals.append(metal)
                else:
                    improb_metals.append(metal)

            if len(prob_metals) == 1:
                lowestMetals = prob_metals
            elif len(prob_metals) > 1:
                for metal in prob_metals:
                    # metal in a negative oxidation state? Use IE = 0.
                    if dONEC[metal][0] < 0:
                        currentIE = 0
                    # metal oxidation state at or higher than highest known? Set IE arbitrarily high.
                    elif dONEC[metal][0] >= highest_known_ON[metal.atomic_symbol]:
                        currentIE = 9999
                    # otherwise, use the appropriate IE.
                    else:
                        currentIE = float(
                            IEs[metal.atomic_symbol][math.floor(dONEC[metal][0])]
                        )
                    if not "lowestIE" in locals():
                        lowestIE = currentIE
                        lowestMetals = [metal]
                    else:
                        if currentIE == lowestIE:
                            lowestMetals.append(metal)
                        if currentIE < lowestIE:
                            lowestIE = currentIE
                            lowestMetals = [metal]
            elif len(prob_metals) == 0:
                for metal in improb_metals:
                    # metal in a negative oxidation state? Use IE = 0.
                    if dONEC[metal][0] < 0:
                        currentIE = 0
                    # metal oxidation state at or higher than highest known? Set IE arbitrarily high.
                    elif dONEC[metal][0] >= highest_known_ON[metal.atomic_symbol]:
                        currentIE = 9999
                    # otherwise, use the appropriate IE.
                    else:
                        currentIE = float(
                            IEs[metal.atomic_symbol][math.floor(dONEC[metal][0])]
                        )
                    if not "lowestIE" in locals():
                        lowestIE = currentIE
                        lowestMetals = [metal]
                    else:
                        if currentIE == lowestIE:
                            lowestMetals.append(metal)
                        if currentIE < lowestIE:
                            lowestIE = currentIE
                            lowestMetals = [metal]
            # distribute one ionization energy level worth of charge
            if available_charge >= len(lowestMetals):
                for metal in lowestMetals:
                    dONEC[metal][0] += 1
                    available_charge -= 1
            elif available_charge < len(lowestMetals):
                for metal in lowestMetals:
                    dONEC[metal][0] += available_charge / (len(lowestMetals))
                available_charge = 0

        # negative contribution?
        if available_charge < 0:
            # get list of improbable and improbable next oxidations
            prob_metals = []
            improb_metals = []
            for metal in dONEC:
                try:
                    prob = float(
                        100 * ONP[metal.atomic_symbol][math.floor(dONEC[metal][0]) - 1]
                    )
                except IndexError:
                    prob = 0
                if prob >= 1:
                    prob_metals.append(metal)
                else:
                    improb_metals.append(metal)

            if len(prob_metals) == 1:
                highestMetals = prob_metals
            elif len(prob_metals) > 1:
                for metal in prob_metals:
                    # metal in a negative oxidation state? Use IE = 0.
                    if dONEC[metal][0] <= 0:
                        currentIE = 0
                    # metal oxidation state higher than highest known? Set IE arbitrarily high.
                    elif dONEC[metal][0] > highest_known_ON[metal.atomic_symbol]:
                        currentIE = 9999
                    # otherwise, use the appropriate IE.
                    else:
                        if not any(
                            [
                                (
                                    math.isclose(
                                        (dONEC[metal][0] % 1), 0, abs_tol=0.0001
                                    )
                                ),
                                (
                                    math.isclose(
                                        (dONEC[metal][0] % 1), 1, abs_tol=0.0001
                                    )
                                ),
                            ]
                        ):
                            currentIE = float(
                                IEs[metal.atomic_symbol][math.floor(dONEC[metal][0])]
                            )
                        else:
                            currentIE = float(
                                IEs[metal.atomic_symbol][
                                    math.floor(dONEC[metal][0]) - 1
                                ]
                            )
                    if not "highestIE" in locals():
                        highestIE = currentIE
                        highestMetals = [metal]
                    else:
                        if currentIE == highestIE:
                            highestMetals.append(metal)
                        if currentIE > highestIE:
                            highestIE = currentIE
                            highestMetals = [metal]

            else:
                for metal in improb_metals:
                    # metal in a negative oxidation state? Use IE = 0.
                    if dONEC[metal][0] <= 0:
                        currentIE = 0
                    # metal oxidation state higher than highest known? Set IE arbitrarily high.
                    elif dONEC[metal][0] > highest_known_ON[metal.atomic_symbol]:
                        currentIE = 9999
                    # otherwise, use the appropriate IE.
                    else:
                        if not any(
                            [
                                (
                                    math.isclose(
                                        (dONEC[metal][0] % 1), 0, abs_tol=0.0001
                                    )
                                ),
                                (
                                    math.isclose(
                                        (dONEC[metal][0] % 1), 1, abs_tol=0.0001
                                    )
                                ),
                            ]
                        ):
                            currentIE = float(
                                IEs[metal.atomic_symbol][math.floor(dONEC[metal][0])]
                            )
                        else:
                            currentIE = float(
                                IEs[metal.atomic_symbol][
                                    math.floor(dONEC[metal][0]) - 1
                                ]
                            )
                    if not "highestIE" in locals():
                        highestIE = currentIE
                        highestMetals = [metal]
                    else:
                        if currentIE == highestIE:
                            highestMetals.append(metal)
                        if currentIE > highestIE:
                            highestIE = currentIE
                            highestMetals = [metal]

            # distribute one ionization energy level worth of charge
            if (-1 * available_charge) >= len(highestMetals):
                for metal in highestMetals:
                    dONEC[metal][0] -= 1
                    available_charge += 1
            elif (-1 * available_charge) < len(highestMetals):
                for metal in highestMetals:
                    dONEC[metal][0] += available_charge / (len(highestMetals))
                available_charge = 0

        # if all charge has been distributed, we're done, otherwise, roll again
        if available_charge == 0:
            return dONEC
        else:
            return recursive_distributor(
                dONEC, available_charge, IEs, ONP, highest_known_ON
            )

    if outer_sphere_charge == 0:
        return sONEC

    distributed_ONEC = {}

    # initialize dictionary for charge distribution
    for metal in sONEC:
        distributed_ONEC[metal] = [sONEC[metal][0]]
        distributed_ONEC[metal].append(int(sONEC[metal][1]))

    # recursively distribute network charge according to ionization energy
    distributed_ONEC = recursive_distributor(
        distributed_ONEC, outer_sphere_charge, IEs, ONP, highest_known_ON
    )

    # finally, adjust electron count to new oxidation state (OiL RiG)
    for metal in sONEC:
        distributed_ONEC[metal][1] = (
            valence_e(metal) + (2 * metal_CN[metal]) - distributed_ONEC[metal][0]
        )

    return distributed_ONEC


def global_charge_distribution(
    metalONdict: dict[Atom, list[float, float]],
    IEs: dict[str, list[float]],
    ONP: dict[str, list[float]],
    highest_known_ON: dict[str, int],
    metal_CN: dict[Molecule, int],
    most_probable_ON: dict[str, int],
) -> dict[Atom, list[float, float]]:
    """
    Redistributes the oxidation state contributions across all metal atoms in
    the structure according to full/global shating (fully delocalized distribution)
    & calculates their associated electron counts. Features utilizing electron
    counts are minimally implemented at this time.

    Parameters:
        metalONdict (dict[ccdc.molecule.Atom, list[float, float]]): dictionary with
                        metal Atom object as keys and lists containing the initial
                        oxidation state and electron count implied by only the
                        equal splitting of binding domain charges as values.
        IEs (dict[str, list[float]]): dictionary with metal element symbols as keys
                        and a list of their reported ionization energies as values.
        ONP (dict[str, list[float]]): dictionary with metal element symbols as keys
                        and a list of the probability at the relevant oxidation
                        states as values.
        highest_known_ON (dict[str, int]) : dictionary with metal element symbols
                        as keys and a their highest known oxidation state as values.
        metal_CN (dict[ccdc.molecule.Molecule, int]): dictionary with as metal Atom
                        objects as keys and their effective coordination number
                        as values.
        most_probable_ON (dict[str, int]) : dictionary with metal element symbols
                        as keys and a their oxidation state with the highest
                        probability as values.

    Returns:
        global_ONEC (dict[ccdc.molecule.Atom, list[float, float]]): dictionary with
                        metal Atom object  as keys and lists containing their
                        redistributed oxidation state and electron count as values.
    """
    global_ONEC = {}

    def recursive_distributor_global(
        iONEC: dict[Atom, list[float, float]],
        available_charge: int,
        sorted_metals: dict[str, Atom],
        IEs: dict[str, list[float]],
        ONP: dict[str, list[float]],
        highest_known_ON: dict[str, int],
    ) -> dict[Atom, list[float, float]]:
        """
        Distribute network charge according to ionization energy and probability
        until all charge is distributed. Performed after tallying available network
        charge and sorting network metals by element type.
        """
        # initialize working dictionary
        dONEC = {}
        dONEC = dict(iONEC)

        # positive contribution?
        if available_charge > 0:
            # get list of improbable and improbable next oxidations
            prob_metal_type = []
            improb_metal_type = []
            for metal_type in sorted_metals:
                try:
                    prob = float(
                        100
                        * ONP[metal_type][
                            math.floor(dONEC[sorted_metals[metal_type][0]][0]) + 1
                        ]
                    )
                except IndexError:
                    prob = 0
                if prob >= 1:
                    prob_metal_type.append(metal_type)
                else:
                    improb_metal_type.append(metal_type)

            # if only one metal type has a probable next oxidation state, do that
            if len(prob_metal_type) == 1:
                lowestMetal = prob_metal_type[0]

            # if more than one metal type has a probable next oxidation state,
            # determine next lowest ionization energy among probable next
            # oxidation states
            elif len(prob_metal_type) > 1:
                # find lowest next ionization energy
                for metal_type in prob_metal_type:
                    # metal in a negative oxidation state? Use IE = 0.
                    if dONEC[sorted_metals[metal_type][0]][0] < 0:
                        currentIE = 0
                    # metal oxidation state at or higher than highest known? Set IE arbitrarily high.
                    elif (
                        dONEC[sorted_metals[metal_type][0]][0]
                        >= highest_known_ON[metal_type]
                    ):
                        currentIE = 9999
                    # otherwise, use the appropriate IE.
                    else:
                        currentIE = float(
                            IEs[metal_type][
                                math.floor(dONEC[sorted_metals[metal_type][0]][0])
                            ]
                        )
                    if not "lowestIE" in locals():
                        lowestIE = currentIE
                        lowestMetal = metal_type
                    else:
                        if currentIE < lowestIE:
                            lowestIE = currentIE
                            lowestMetal = metal_type

            # if there is no probable next oxidation state available,
            # determine lowest ionization energy among improbable next oxidation states
            elif len(prob_metal_type) == 0:
                # find lowest next ionization energy
                for metal_type in improb_metal_type:
                    # metal in a negative oxidation state? Use IE = 0.
                    if dONEC[sorted_metals[metal_type][0]][0] < 0:
                        currentIE = 0
                    # metal oxidation state at or higher than highest known? Set IE arbitrarily high.
                    elif (
                        dONEC[sorted_metals[metal_type][0]][0]
                        >= highest_known_ON[metal_type]
                    ):
                        currentIE = 9999
                    # otherwise, use the appropriate IE.
                    else:
                        currentIE = float(
                            IEs[metal_type][
                                math.floor(dONEC[sorted_metals[metal_type][0]][0])
                            ]
                        )
                    if not "lowestIE" in locals():
                        lowestIE = currentIE
                        lowestMetal = metal_type
                    else:
                        if currentIE < lowestIE:
                            lowestIE = currentIE
                            lowestMetal = metal_type

            # distribute one ionization energy level worth of charge
            if available_charge >= len(sorted_metals[lowestMetal]):
                for metal in sorted_metals[lowestMetal]:
                    dONEC[metal][0] += 1
                    available_charge -= 1
            elif available_charge < len(sorted_metals[lowestMetal]):
                for metal in sorted_metals[lowestMetal]:
                    dONEC[metal][0] += available_charge / (
                        len(sorted_metals[lowestMetal])
                    )
                available_charge = 0

        # negative contribution?
        if available_charge < 0:
            # get list of improbable and improbable next oxidations
            prob_metal_type = []
            improb_metal_type = []
            for metal_type in sorted_metals:
                try:
                    prob = float(
                        100
                        * ONP[metal_type][
                            math.floor(dONEC[sorted_metals[metal_type][0]][0]) - 1
                        ]
                    )
                except IndexError:
                    prob = 0
                if prob >= 1:
                    prob_metal_type.append(metal_type)
                else:
                    improb_metal_type.append(metal_type)

            # if only one metal type has a probable next oxidation state, do that
            if len(prob_metal_type) == 1:
                highestMetal = prob_metal_type[0]

            # if more than one metal type has a probable next oxidation state,
            # determine next highest ionization energy among probable next
            # oxidation states
            elif len(prob_metal_type) > 1:
                for metal_type in prob_metal_type:
                    # metal in a negative oxidation state? Use IE = 0.
                    if dONEC[sorted_metals[metal_type][0]][0] <= 0:
                        currentIE = 0
                    # metal oxidation state higher than highest known? Set IE arbitrarily high.
                    elif (
                        dONEC[sorted_metals[metal_type][0]][0]
                        > highest_known_ON[metal_type]
                    ):
                        currentIE = 9999
                    # otherwise, use the appropriate IE.
                    else:
                        if not any(
                            [
                                (
                                    math.isclose(
                                        (dONEC[sorted_metals[metal_type][0]][0] % 1),
                                        0,
                                        abs_tol=0.0001,
                                    )
                                ),
                                (
                                    math.isclose(
                                        (dONEC[sorted_metals[metal_type][0]][0] % 1),
                                        1,
                                        abs_tol=0.0001,
                                    )
                                ),
                            ]
                        ):
                            currentIE = float(
                                IEs[metal_type][
                                    math.floor(dONEC[sorted_metals[metal_type][0]][0])
                                ]
                            )
                        else:
                            currentIE = float(
                                IEs[metal_type][
                                    math.floor(dONEC[sorted_metals[metal_type][0]][0])
                                    - 1
                                ]
                            )
                    if not "highestIE" in locals():
                        highestIE = currentIE
                        highestMetal = metal_type
                    else:
                        if currentIE > highestIE:
                            highestIE = currentIE
                            highestMetal = metal_type

            # if no probable next oxidation states are available,
            # determine next highest ionization energy among probable next
            # oxidation states
            elif len(improb_metal_type) > 0:
                for metal_type in improb_metal_type:
                    # metal in a negative oxidation state? Use IE = 0.
                    if dONEC[sorted_metals[metal_type][0]][0] <= 0:
                        currentIE = 0
                    # metal oxidation state higher than highest known? Set IE arbitrarily high.
                    elif (
                        dONEC[sorted_metals[metal_type][0]][0]
                        > highest_known_ON[metal_type]
                    ):
                        currentIE = 9999
                    # otherwise, use the appropriate IE.
                    else:
                        if not any(
                            [
                                (
                                    math.isclose(
                                        (dONEC[sorted_metals[metal_type][0]][0] % 1),
                                        0,
                                        abs_tol=0.0001,
                                    )
                                ),
                                (
                                    math.isclose(
                                        (dONEC[sorted_metals[metal_type][0]][0] % 1),
                                        1,
                                        abs_tol=0.0001,
                                    )
                                ),
                            ]
                        ):
                            currentIE = float(
                                IEs[metal_type][
                                    math.floor(dONEC[sorted_metals[metal_type][0]][0])
                                ]
                            )
                        else:
                            currentIE = float(
                                IEs[metal_type][
                                    math.floor(dONEC[sorted_metals[metal_type][0]][0])
                                    - 1
                                ]
                            )
                    if not "highestIE" in locals():
                        highestIE = currentIE
                        highestMetal = metal_type
                    else:
                        if currentIE > highestIE:
                            highestIE = currentIE
                            highestMetal = metal_type

            # distribute one ionization energy level worth of charge
            if (-1 * available_charge) >= len(sorted_metals[highestMetal]):
                for metal in sorted_metals[highestMetal]:
                    dONEC[metal][0] -= 1
                    available_charge += 1
            elif (-1 * available_charge) < len(sorted_metals[highestMetal]):
                for metal in sorted_metals[highestMetal]:
                    dONEC[metal][0] += available_charge / (
                        len(sorted_metals[highestMetal])
                    )
                available_charge = 0

        # if all charge has been distributed, we're done, otherwise, roll again
        if available_charge == 0:
            return dONEC
        else:
            return recursive_distributor_global(
                dONEC, available_charge, sorted_metals, IEs, ONP, highest_known_ON
            )

    # sort metals by element type
    sorted_metals = {}
    for metal in metalONdict:
        sorted_metals[metal.atomic_symbol] = []
    for metal in metalONdict:
        sorted_metals[metal.atomic_symbol].append(metal)

    # tally up the global charge to be distributed
    # and initialize global ON to most probable for all metals
    global_charge = 0
    for metal in metalONdict:
        global_charge += metalONdict[metal][0]
        global_ONEC[metal] = [most_probable_ON[metal.atomic_symbol]]
        global_charge -= most_probable_ON[metal.atomic_symbol]
        global_ONEC[metal].append(int(metalONdict[metal][1]))

    # recursively distribute network charge according to ionization energy
    if math.isclose(global_charge, 0, abs_tol=0.0001):
        distributed_ONEC = global_ONEC
    else:
        distributed_ONEC = recursive_distributor_global(
            global_ONEC, global_charge, sorted_metals, IEs, ONP, highest_known_ON
        )

    # finally, adjust electron count to new oxidation state (OiL RiG)
    for metal in metalONdict:
        global_ONEC[metal][1] = (
            valence_e(metal) + (2 * metal_CN[metal]) - distributed_ONEC[metal][0]
        )
    return global_ONEC


def KnownONs() -> dict[str, list[int]]:
    """
    Reads in the known oxidation states for each metal element.

    Parameters:
        None. (info stored in KnownON.csv file)

    Returns:
        KONs (dict[str, list(int)]) : dictionary with metal element symbols as keys
                              and a list of their known oxidation states as values.
    """
    KONs = {}
    with open(os.path.join(CODE_DIR, "mosaec/KnownON.csv")) as ONs:
        for ON in ONs.readlines():
            ONlist = []
            splitON = ON.split(",")
            for split in splitON:
                split.replace(",", "")
            while "" in splitON:
                splitON.remove("")
            while "\n" in splitON:
                splitON.remove("\n")
            for i in range(1, len(splitON)):
                ONlist.append(splitON[i])
            KONs[splitON[0]] = ONlist
    ONs.close()
    return KONs


def IonizationEnergies() -> dict[str, list[float]]:
    """
    Reads in the reported ionization energies for each metal element.

    Parameters:
        None. (info stored in Ionization_Energies.csv file)

    Returns:
        KIEs (dict[str, list[float]]): dictionary with metal element symbols as keys
                          and a list of their reported ionization energies as values.
    """
    KIEs = {}
    IElist = []
    with open(os.path.join(CODE_DIR, "mosaec/Ionization_Energies.csv")) as IEs:
        for IE in IEs.readlines():
            splitIE = IE.split(",")
            for split in splitIE:
                split.replace(",", "")
                if r"\n" in split:
                    split.replace(r"\n", "")
            while "" in splitIE:
                splitIE.remove("")
            while r"\n" in splitIE:
                splitIE.remove(r"\n")
            IElist.append(splitIE)
        for entry in IElist:
            if entry[1] in KIEs:
                KIEs[(entry[1])].append(entry[2])
            else:
                KIEs[entry[1]] = []
                KIEs[entry[1]].append(entry[2])
    IEs.close()
    return KIEs


def HighestKnownONs() -> dict[str, int]:
    """
    Determines the highest known oxidation states for each metal element.

    Parameters:
        None. (info stored in Known.csv file)

    Returns:
        HKONs (dict[str, int]) : dictionary with metal element symbols as keys
                              and their highest known oxidation state as values.
    """
    HKONs = {}
    with open(os.path.join(CODE_DIR, "mosaec/KnownON.csv")) as ONs:
        for ON in ONs.readlines():
            highest = 0
            splitON = ON.split(",")
            for split in splitON:
                split.replace(",", "")
            while "" in splitON:
                splitON.remove("")
            while "\n" in splitON:
                splitON.remove("\n")
            for i in range(1, len(splitON)):
                if int(splitON[i]) >= highest:
                    highest = int(splitON[i])
            HKONs[splitON[0]] = int(highest)
    ONs.close()
    return HKONs


def ONprobabilities() -> dict[str, list[float]]:
    """
    Reads in the probability of each oxidation state for all metal elements.
    Approximate probabilities are assessed by their relative frequency of
    occurence in the CSD metadata.

    Parameters:
        None. (info stored in Oxidation_Probabilities.csv file)

    Returns:
        ONP (dict[str, list[float]]): dictionary with metal element symbols as keys
                        and a list of the probability at the relevant oxidation
                        states as values.
    """
    ONP = {}
    with open(os.path.join(CODE_DIR, "mosaec/Oxidation_Probabilities.csv")) as ONPs:
        for ON in ONPs.readlines():
            ONPlist = []
            splitONP = ON.split(",")
            for split in splitONP:
                split.replace(",", "")
            while "" in splitONP:
                splitONP.remove("")
            while "\n" in splitONP:
                splitONP.remove("\n")
            for i in range(1, len(splitONP)):
                ONPlist.append(float(splitONP[i]))
            ONP[splitONP[0]] = ONPlist
    ONPs.close()
    return ONP


def ONmostprob(iONP: dict[str, list[float]]) -> dict[str, int]:
    """
    Determines the highest probability oxidation state for each metal element.
    These values are utilized during charge distribution routines.

    Parameters:
        iONP (dict[str, list[float]]): dictionary with metal element symbols as
                        keys and a list of the probability at the relevant oxidation
                        states as values.

    Returns:
        MPOS (dict[str, int]) : dictionary with metal element symbols as keys and
                        their oxidation state with the highest probability as values.
    """
    MPOS = {}
    for metal in iONP:
        hprob = 0
        for index, prob in enumerate(iONP[metal]):
            if prob >= hprob:
                hprob = prob
                MPOS[metal] = index
    return MPOS


def getCN(lsites: dict[Atom, list[Atom]]) -> dict[Molecule, int]:
    """
    Determines the highest probability oxidation state for each metal element.
    These values are utilized during charge distribution routines.

    Parameters:
        lsites (dict[Atom, list[Atom]]): dictionary with metal Atom object as keys
                        and the list of ligand atoms which bind them as values.

    Returns:
        CNdict (dict[Molecule, int]): dictionary with as metal Atom objects as
                        keys and effective coordination number as values.
    """
    CNdict = {}
    for metal in lsites:
        CNdict[metal] = 0
        for ligand in lsites[metal]:
            if hapticity(ligand, metal):
                CNdict[metal] += 0.5
            else:
                CNdict[metal] += 1
        for neighbour in metal.neighbours:
            if neighbour.is_metal:
                CNdict[metal] += 0.5
    return CNdict


cwd = os.getcwd()
KnownON = KnownONs()
KnownIE = IonizationEnergies()
HighestKnownON = HighestKnownONs()
ONProb = ONprobabilities()
HighestProbON = ONmostprob(ONProb)


def generate_mosaec(file: str) -> tuple[dict[str, float], ...]:
    # read in the .cif, extract the underlying molecule,
    # identify the unique sites, metal sites, binding sites, etc.
    if str(file).endswith(".cif"):
        try:
            cif, cif_order = readentry((file))
            mol = cif.molecule
            asymmol = cif.asymmetric_unit_molecule
        except Exception as e:
            print(f"{file} >>>> CSD failed to read cif data")
            return None
    # print ('identifying unique sites')
    uniquesites = get_unique_sites(mol, asymmol)
    usitedict = {}
    for usite in uniquesites:
        usitedict[usite.label] = usite
    # print ('getting metal sites')
    metalsites = get_metal_sites(uniquesites)
    if len(metalsites) == 0:
        print(f"{file} >>>> No metal sites detected...Skipping OS calculation")
        return None
    # print ('identifying binding sites')
    ligand_sites = get_ligand_sites(metalsites, uniquesites)
    binding_sites = get_binding_sites(metalsites, uniquesites)
    # Now get the localized oxidation state contribution of each atom
    # need delocalized bond contributions
    dVBO = delocalisedLBO(mol)
    # then need aromatic bond contributions
    rVBO = ringVBOs(mol)
    # finally combine delocal/aromatic bond conrtibutions with localized bonding
    AON = iVBS_Oxidation_Contrib(uniquesites, rVBO, dVBO)
    # Previous only assigns an oxidation contribution to unique images of atoms,
    # also need to assign these values to redundant sites:
    rAON = redundantAON(AON, mol)
    # Split the MOF into binding domains and assign oxidation state
    # contributions to binding sites
    # print ('getting binding domains')
    binding_sphere = binding_domain(binding_sites, rAON, mol, usitedict)
    # print ('partitioning charge')
    bindingAON = binding_contrib(binding_sphere, binding_sites, rAON)
    # get metal connection network
    connected_metals = get_metal_networks(ligand_sites, binding_sphere, bindingAON)
    # get metal effective coordination number:
    mCN = getCN(ligand_sites)

    ONEC_inout = {}
    noint_balance = {}

    # this block assigns the oxidation state and electron count only considering binding domains
    ONEC_inner = {}
    for metal in ligand_sites:
        oxidation_state = 0
        valence = valence_e(metal)
        electron_count = valence

        for ligand in ligand_sites[metal]:
            LBO = bindingAON[ligand]
            Nbridge = bridging(ligand)
            Ox = LBO / Nbridge
            oxidation_state += Ox
            if Ox >= 2:
                mCN[metal] += 1
            if Ox >= 3:
                mCN[metal] += 1

        electron_count = valence + (2 * mCN[metal]) - oxidation_state
        ONEC_inner[metal] = [oxidation_state, electron_count]

    # redistribute ONEC within metal networks based on ionization energy
    noint_balance = distribute_ONEC(
        ONEC_inner,
        connected_metals,
        KnownIE,
        ONProb,
        HighestKnownON,
        mCN,
        HighestProbON,
    )

    # determine and distribute outer sphere charges
    OSD = outer_sphere_domain(uniquesites, binding_sphere)
    OSC = outer_sphere_contrib(OSD, rAON)
    noint_outer = distribute_OuterSphere(
        noint_balance, OSC, KnownIE, ONProb, HighestKnownON, mCN
    )
    ONEC_inout = distribute_OuterSphere(
        ONEC_inner, OSC, KnownIE, ONProb, HighestKnownON, mCN
    )
    globaldis = global_charge_distribution(
        ONEC_inout, KnownIE, ONProb, HighestKnownON, mCN, HighestProbON
    )

    # GOAL: return dictionary of {AtomLabel:OS/FC}
    # First > dictionary with non-metal oxidation state contribution ("Formal Charge")
    fc_dict = {}
    # must reorder dict to match cif order & pymatgen
    modAON = {k.label: v for k, v in AON.items()}
    # print(cif_order)
    # print(aon_labels)
    # print(AON)
    # Reorder to match original cif atom ordering (match other descriptos)
    for x in cif_order:
        fc_dict[x] = float(-1 * modAON[x]) if x in modAON.keys() else 0
    inner_dict = fc_dict
    # Update with calculated metal OS
    inner_metal_os = {x.label: y[0] for x, y in ONEC_inner.items()}
    inner_dict.update(inner_metal_os)
    # Repeat with inner-outer sphere OS
    outer_dict = fc_dict
    outer_metal_os = {x.label: y[0] for x, y in ONEC_inout.items()}
    outer_dict.update(outer_metal_os)
    # Repeat with global sphere OS
    global_dict = fc_dict
    global_metal_os = {x.label: y[0] for x, y in globaldis.items()}
    global_dict.update(global_metal_os)
    return inner_dict, outer_dict, global_dict
