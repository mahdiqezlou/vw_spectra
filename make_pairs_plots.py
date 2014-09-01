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

colors = ["black", "red", "blue", "green", "grey", "brown", "pink", "orange", "cyan"]

def plot_median_pair(haloname, snapnum, subhalopair, idnum=1):
    """Plot the median and quartiles for a pair"""
    label = "Pairs "+str(subhalopair[0])+"-"+str(subhalopair[1])
    savefile = "subhalo_spectra_"+"".join(map(str,subhalopair))+".hdf5"
    hspec = ps.VWPlotSpectra(snapnum, haloname, label=label, savefile=savefile)
    plt.figure(1)
    hspec.plot_vel_width("Si", 2, color=colors[idnum], ls = "--")
    vw = hspec.vel_width("Si", 2)
    plt.figure(2)
    median = np.median(vw)
    plt.errorbar([median,], idnum, xerr = [[median - np.percentile(vw, 25),], [np.percentile(vw, 75)- median,]], fmt='o')
    plt.figure(1)

sim = 5
snap = 3
halo = myname.get_name(sim, True, box=10)
#Load from a save file only
total = ps.VWPlotSpectra(snap, halo, label="Total")
total.plot_vel_width("Si", 2, color="black", ls = "-")

#Get the subhalo list
subs=subfindhdf.SubFindHDF5(halo, snap)
#In solar masses
sub_mass=np.array(subs.get_sub("SubhaloMass"))
loc= 2.4
plt.text(10, loc, "Halo mass: (1e10 Msun)")
height=1
for pair in ([1,4], [835, 837], [1556, 1558]):
    plot_median_pair(halo, snap, pair, height)
    height+=1
    for n in pair:
        loc-=0.1
        plt.text(10, loc,str(n)+": "+pr_num(sub_mass[n]))

snap = 4
#Get the subhalo list
subs=subfindhdf.SubFindHDF5(halo, snap)
sub_mass=np.array(subs.get_sub("SubhaloMass"))
for pair in ([2531,2532], [3240, 3241]):
    plot_median_pair(halo, snap, pair, height)
    height+=1
    for n in pair:
        loc-=0.1
        plt.text(10, loc,str(n)+": "+pr_num(sub_mass[n]))

snap = 5
#Get the subhalo list
subs=subfindhdf.SubFindHDF5(halo, snap)
sub_mass=np.array(subs.get_sub("SubhaloMass"))
for pair in ([3775, 3776],):
    plot_median_pair(halo, snap, pair, height)
    height+=1
    for n in pair:
        loc-=0.1
        plt.text(10, loc,str(n)+": "+pr_num(sub_mass[n]))

plt.xlim(10,4000)
plt.legend()
save_figure(outdir+"pairs_vw")

plt.figure(2)
plt.legend()
save_figure(outdir+"pairs_scatter")
