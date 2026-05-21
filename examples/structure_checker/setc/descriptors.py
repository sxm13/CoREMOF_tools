"""
Module handling the calculation of the local chemical environment descriptors
(i.e. RDFs, ADFs, R/A-ACSFs, and raw & smoothed R/A-WAPs).

"""

import numpy as np
import pandas as pd

from itertools import combinations, product

from pymatgen.core import Structure

from elements import en_ghosh


def generate_descriptors(
    cif_struct: Structure, symbols: list[str], labels: list[str]
) -> np.array:
    """
    Calculate the local chemical environment node features for all atoms
    in the structure.

    Parameters:
        cif_struct (pymatgen.core.Structure): Structure object containing
                                   atomic positions, labels, lattice, etc.
        symbols (list[str]):    list of element symbols in the Structure.
        labels  (list[str]):    list of atomic labels in the Structure.

    Returns:
        feature_list (np.array): array of all the local chemical environment
                                 features (RDF, ACSF, WAP, etc).
    """

    n_atoms = len(cif_struct)
    property_en = [en_ghosh[symbol] for symbol in symbols]

    max_frac = cif_struct.lattice.get_fractional_coords([8.0, 8.0, 8.0])
    super_scale = np.ceil(max_frac).astype(int)
    list_options = list(product(*[range(2 * i + 1) for i in super_scale]))
    super_options = np.array(list_options, dtype=float) - super_scale

    frac_xyz_og = np.array([atom.frac_coords for atom in cif_struct])
    frac_xyz_diff = frac_xyz_og.repeat(n_atoms, axis=0)
    frac_xyz_diff -= np.tile(frac_xyz_og, (n_atoms, 1))

    frac2cart = cif_struct.lattice.get_cartesian_coords

    cart_headers = ["cart_x", "cart_y", "cart_z"]

    all_img_list = []
    for i_opt, option in enumerate(super_options):
        df_tmp1 = pd.DataFrame(
            {"img_num": i_opt, "atom_label": labels, "property_en": property_en}
        )
        cart_xyz_mirror = frac2cart(frac_xyz_og + option)
        df_tmp2 = pd.DataFrame(data=cart_xyz_mirror, columns=cart_headers)
        cart_xyz_opt_diff = frac2cart(frac_xyz_diff + option)
        distance_full = np.linalg.norm(cart_xyz_opt_diff, axis=1).reshape(
            n_atoms, n_atoms
        )
        df_tmp3 = pd.DataFrame(data=distance_full, columns=labels)
        all_img_list.append(pd.concat([df_tmp1, df_tmp2, df_tmp3], axis=1))
    all_img_df = pd.concat(all_img_list)

    # parameters
    r_0 = 1.0
    rad_cut = 8.0
    ang_cut = 6.0
    rdf_bin_size = 0.25
    rdf_alpha = 60.0
    acsf_rad_bin_size = 0.5
    acsf_rad_eta = 6.0
    adf_bin_size_degree = 10
    adf_beta = 60.0
    acsf_ang_n_eta = 8

    # rdf
    rdf_n_bins = int((rad_cut - r_0) / rdf_bin_size)
    rdf_edges = np.linspace(r_0 + rdf_bin_size, rad_cut, num=rdf_n_bins)

    # radial ACSF
    acsf_rad_n_bins = int((rad_cut - r_0) / acsf_rad_bin_size)
    acsf_rad_edges = np.linspace(r_0 + acsf_rad_bin_size, rad_cut, num=acsf_rad_n_bins)

    # adf
    adf_bin_size = np.deg2rad(adf_bin_size_degree)
    adf_n_bins = int(np.pi / adf_bin_size)
    adf_edges = np.linspace(adf_bin_size, np.pi, num=adf_n_bins)

    # angular ACSF
    acsf_ang_edges = np.linspace(r_0, ang_cut, num=acsf_ang_n_eta)
    acsf_ang_eta = 1 / (2 * (acsf_ang_edges**2))
    acsf_ang_lambda = [-1, 1]
    acsf_ang_n_bins = acsf_ang_n_eta * 2
    acsf_ang_lambda_bins, acsf_ang_eta_bins = np.array(
        list(product(acsf_ang_lambda, acsf_ang_eta))
    ).T

    features_list = []
    for atom_label in labels:
        # radial environment
        rad_range = (all_img_df[atom_label] >= r_0) & (
            all_img_df[atom_label] <= rad_cut
        )
        rad_prop = all_img_df.loc[rad_range, "property_en"].to_numpy()
        rad_sphere = all_img_df.loc[rad_range, atom_label].to_numpy()

        # rdf
        rdf_diff_bins = np.tile(rad_sphere, (rdf_n_bins, 1)).T - rdf_edges
        rdf_gauss_bins = np.exp(-rdf_alpha * (rdf_diff_bins**2))
        rdf = (rdf_gauss_bins.T * rad_prop).sum(axis=1)

        # radial WAP
        rad_abs_bins = np.abs(np.ma.masked_outside(rdf_diff_bins, -rdf_bin_size, 0.0))
        wrh_top = (rad_abs_bins.T * rad_prop).sum(axis=1).data
        wrh_btm = rad_abs_bins.sum(axis=0).data
        wap_rad_harsh = np.divide(
            wrh_top, wrh_btm, out=np.zeros(rdf_n_bins), where=(wrh_btm != 0)
        )
        wrs_btm = rdf_gauss_bins.sum(axis=0)
        wap_rad_smooth = np.divide(
            rdf, wrs_btm, out=np.zeros(rdf_n_bins), where=(wrs_btm != 0)
        )

        # radial ACSF
        acsf_rad_fcut = (np.cos(rad_sphere * np.pi / rad_cut) + 1) * 0.5
        acsf_rad_diff_bins = (
            np.tile(rad_sphere, (acsf_rad_n_bins, 1)).T - acsf_rad_edges
        )
        acsf_rad_gauss_bins = np.exp(-acsf_rad_eta * (acsf_rad_diff_bins**2))
        acsf_rad = (rad_prop * acsf_rad_fcut * acsf_rad_gauss_bins.T).sum(axis=1)

        # angular environment
        ang_range = (all_img_df[atom_label] >= r_0) & (
            all_img_df[atom_label] <= ang_cut
        )
        ang_prop = all_img_df.loc[ang_range, "property_en"].to_numpy()
        ang_sphere = all_img_df.loc[ang_range, atom_label].to_numpy()

        if len(ang_sphere) > 1:
            # angle calculation
            cart_xyz = all_img_df.loc[ang_range, cart_headers].to_numpy()
            atoms_j, atoms_k = np.array(list(combinations(range(len(ang_sphere)), 2))).T
            property_jk = ang_prop[atoms_j] * ang_prop[atoms_k]
            d_ijk = np.vstack(
                [
                    ang_sphere[atoms_j],
                    ang_sphere[atoms_k],
                    np.linalg.norm(cart_xyz[atoms_j] - cart_xyz[atoms_k], axis=1),
                ]
            )
            d_ijk_sq = d_ijk**2
            cos_theta_ijk = (d_ijk_sq[0] + d_ijk_sq[1] - d_ijk_sq[2]) / (
                2 * d_ijk[0] * d_ijk[1]
            )
            cos_theta_ijk[cos_theta_ijk < -1.0] = -1.0
            cos_theta_ijk[cos_theta_ijk > 1.0] = 1.0
            theta_ijk = np.arccos(cos_theta_ijk)

            # adf
            adf_diff_bins = np.tile(theta_ijk, (adf_n_bins, 1)).T - adf_edges
            adf_gauss_bins = np.exp(-adf_beta * (adf_diff_bins**2))
            adf = (property_jk * adf_gauss_bins.T).sum(axis=1)

            # angular WAP
            ang_abs_bins = np.ma.masked_outside(adf_diff_bins, -adf_bin_size, 0.0)
            wah_top = (ang_abs_bins.T * property_jk).sum(axis=1).data
            wah_btm = ang_abs_bins.sum(axis=0).data
            wap_ang_harsh = np.divide(
                wah_top, wah_btm, out=np.zeros(adf_n_bins), where=(wah_btm != 0)
            )
            was_btm = adf_gauss_bins.sum(axis=0)
            wap_ang_smooth = np.divide(
                adf, was_btm, out=np.zeros(adf_n_bins), where=(was_btm != 0)
            )

            # angular ACSF
            acsf_ang_fcut = ((np.cos(d_ijk * np.pi / ang_cut) + 1) * 0.5).prod(axis=0)
            acsf_ang_pre = np.tile(cos_theta_ijk, (acsf_ang_n_bins, 1)).T
            acsf_ang_cos_bins = 1 + acsf_ang_lambda_bins * acsf_ang_pre
            acsf_ang_dist_bins = np.tile(d_ijk_sq.sum(axis=0), (acsf_ang_n_bins, 1)).T
            acsf_ang_gauss_bins = np.exp(-acsf_ang_eta_bins * acsf_ang_dist_bins)
            acsf_ang_bins = (acsf_ang_cos_bins * acsf_ang_gauss_bins).T
            acsf_ang_bins *= property_jk * acsf_ang_fcut
            acsf_ang = acsf_ang_bins.sum(axis=1)
        else:
            adf, wap_ang_harsh, wap_ang_smooth = np.zeros((3, adf_n_bins))
            acsf_ang = np.zeros(acsf_ang_n_bins)

        features_list.append(
            np.hstack(
                [
                    rdf,
                    adf,
                    acsf_rad,
                    acsf_ang,
                    wap_rad_harsh,
                    wap_rad_smooth,
                    wap_ang_harsh,
                    wap_ang_smooth,
                ]
            ),
        )

    return np.vstack(features_list)
