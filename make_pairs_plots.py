#!/usr/bin env python
# -*- coding: utf-8 -*-
"""Make some plots of the velocity widths from the merging pairs"""

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt

import vw_plotspectra as ps
import myname
import os.path as path
import numpy as np
import subfindhdf
from save_figure import save_figure

outdir = path.join(myname.base, "plots/")
print "Plots at: ",outdir
zrange = {1:(7,3.5), 3:(3.5,2.5), 5:(2.5,0)}
zzz = {1:4, 3:3, 5:2}

def pr_num(num,rnd=2):
    """Return a string rep of a number"""
    return str(np.round(num,rnd))

sim = 5
snap = 3
halo = myname.get_name(sim, True, box=10)
#Load from a save file only
hspec = ps.VWPlotSpectra(snap, halo, label="Pairs 1-4", savefile="subhalo_spectra_14.hdf5")
hspec.plot_vel_width("Si", 2, color="red", ls = "--")
hspec = ps.VWPlotSpectra(snap, halo, label="Pairs 835-7", savefile="subhalo_spectra_835837.hdf5")
hspec.plot_vel_width("Si", 2, color="blue", ls = "--")
hspec = ps.VWPlotSpectra(snap, halo, label="Pairs 1556-8", savefile="subhalo_spectra_15561558.hdf5")
hspec.plot_vel_width("Si", 2, color="green", ls = "--")
hspec = ps.VWPlotSpectra(snap, halo, label="Total")
hspec.plot_vel_width("Si", 2, color="black", ls = "-")

#Get the subhalo list
subs=subfindhdf.SubFindHDF5(halo, snap)
#In solar masses
sub_mass=np.array(subs.get_sub("SubhaloMass"))
loc= 2.4
plt.text(10, loc, "Halo mass: (1e10 Msun)")
for n in (1,4,835, 837, 1556, 1558):
    loc-=0.1
    plt.text(10, loc,str(n)+": "+pr_num(sub_mass[n]))
# plt.text(10, 2.45, strrr)
snap = 4
hspec = ps.VWPlotSpectra(snap, halo, label="Pairs 2531-2", savefile="subhalo_spectra_25312532.hdf5")
hspec.plot_vel_width("Si", 2, color="grey", ls = "--")
hspec = ps.VWPlotSpectra(snap, halo, label="Pairs 3240-1", savefile="subhalo_spectra_32403241.hdf5")
hspec.plot_vel_width("Si", 2, color="brown", ls = "--")

#Get the subhalo list
subs=subfindhdf.SubFindHDF5(halo, snap)
sub_mass=np.array(subs.get_sub("SubhaloMass"))
for n in (2531,2532,3240,3241):
    loc-=0.1
    plt.text(10, loc,str(n)+": "+pr_num(sub_mass[n]))
snap = 5
hspec = ps.VWPlotSpectra(snap, halo, label="Pairs 3775-6", savefile="subhalo_spectra_37753776.hdf5")
hspec.plot_vel_width("Si", 2, color="pink", ls = "--")
#Get the subhalo list
subs=subfindhdf.SubFindHDF5(halo, snap)
sub_mass=np.array(subs.get_sub("SubhaloMass"))
for n in (3775,3776):
    loc-=0.1
    plt.text(10, loc,str(n)+": "+pr_num(sub_mass[n]))
plt.xlim(10,4000)
plt.legend()
save_figure(outdir+"pairs_vw")
