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

colors = ["black", "red", "blue", "green", "grey", "brown", "pink", "orange", "cyan", "magenta"]

def plot_median_bar(hspecc, idnum, label):
    """Plot median with error bars for velocity widths"""
    plt.figure(2)
    vw = hspecc.vel_width("Si", 2)
    median = np.median(vw)
    plt.errorbar([median,], idnum, xerr = [[median - np.percentile(vw, 25),], [np.percentile(vw, 75)- median,]], fmt='o', label=label)
    plt.figure(1)
    return median

def plot_median_pair(haloname, snapnum, subhalopair, idnum=1):
    """Plot the median and quartiles for a pair"""
    label = "Pairs "+str(subhalopair[0])+"-"+str(subhalopair[1])
    savefile = "subhalo_spectra_"+"".join(map(str,subhalopair))+".hdf5"
    hspec = ps.VWPlotSpectra(snapnum, haloname, label=label, savefile=savefile)
    plt.figure(1)
    hspec.plot_vel_width("Si", 2, color=colors[idnum % np.size(colors)], ls = "--")
    return plot_median_bar(hspec, idnum, label)

#Do big box
# base = "/home/spb/data/Illustris/"
# savedir = "/home/spb/data/Illustris/"
# pairs63 = np.loadtxt("pairs063.txt")
# pairs = zip(pairs63[:,0], pairs63[:,1])
#
# medians = []
# height = 1
# for pair in pairs:
#     medians.append(plot_median_pair(base, 63, pair, height))
#     height+=1
#
# pairs68 = np.loadtxt("pairs068.txt")
# pairs = zip(pairs68[:,0], pairs68[:,1])
#
# for pair in pairs:
#     medians.append(plot_median_pair(base, 68, pair, height))
#     height+=1
#

#My simulation
sim = 5
snap = 3
halo = myname.get_name(sim, True, box=10)
#Load from a save file only
total = ps.VWPlotSpectra(snap, halo, label="Total")
# total.plot_vel_width("Si", 2, color="black", ls = "-")
#
#plt.xlim(10,4000)
## plt.legend()
#save_figure(outdir+"illus_pairs_vw")
#
#tot_median = plot_median_bar(total, 0, "Total")
## plt.legend()
#plt.figure(2)
#plt.ylim(-0.1, height+1)
#save_figure(outdir+"illus_pairs_scatter")
#plt.clf()
#
#table=np.logspace(np.log10(np.min(medians)),np.log10(np.max(medians)),10)
#center = np.array([(table[i]+table[i+1])/2. for i in range(0,np.size(table)-1)])
#nn = np.histogram(medians,table)[0]
#plt.semilogx(center,nn, ls="-", color="red")
#total.plot_vel_width("Si", 2, color="black", ls = "-")
#plt.xlim(10,100)
#save_figure(outdir+"illus_pairs_pdf")
#plt.clf()

#My simulation
plt.figure(2)
total.plot_vel_width("Si", 2, color="black", ls = "-")
plt.figure(1)
tot_median = plot_median_bar(total, 0, "Total")

medians = []
height=1

zzz = {1:"4",2:"3.5",3:"3",4:"2.5",5:"2"}
#Get the subhalo list
for snap in [3,4,5,1,2]:
    subs=subfindhdf.SubFindHDF5(halo, snap)
    #In solar masses
    # sub_mass=np.array(subs.get_sub("SubhaloMass"))
    # loc= 2.4
    # plt.text(10, loc, "Halo mass: (1e10 Msun)")
    pairs = np.loadtxt("pairs10-"+str(snap)+"2.txt")
    if len(np.shape(pairs)) == 1:
        pairs = np.reshape(pairs, (1, np.shape(pairs)[0]))
    for pair in zip(pairs[:,0], pairs[:,1]):
        medians.append(plot_median_pair(halo, snap, pair, height))
        print height, " snap: ", zzz[snap], " pair: ",str(pair[0]),"-",str(pair[1])
        height+=1
    #     for n in pair:
    #         loc-=0.1
    #         plt.text(10, loc,str(n)+": "+pr_num(sub_mass[n]))

plt.xlim(10,4000)
# plt.legend()
save_figure(outdir+"pairs_vw_2")
plt.clf()

plt.figure(2)
# plt.legend()
plt.ylim(-0.1, height+1)
save_figure(outdir+"pairs_scatter_2")
plt.clf()

#My simulation
snap = 3
plt.figure(2)
total.plot_vel_width("Si", 2, color="black", ls = "-")
plt.figure(1)
tot_median = plot_median_bar(total, 0, "Total")

medians = []
height=1

print "Larger sample"

zzz = {1:"4",2:"3.5",3:"3",4:"2.5",5:"2"}
#Get the subhalo list
for snap in [3,4,1,2]:
    subs=subfindhdf.SubFindHDF5(halo, snap)
    #In solar masses
    # sub_mass=np.array(subs.get_sub("SubhaloMass"))
    # loc= 2.4
    # plt.text(10, loc, "Halo mass: (1e10 Msun)")
    pairs = np.loadtxt("pairs10-"+str(snap)+".txt")
    if len(np.shape(pairs)) == 1:
        pairs = np.reshape(pairs, (1, np.shape(pairs)[0]))
    for pair in zip(pairs[:,0], pairs[:,1]):
#         medians.append(plot_median_pair(halo, snap, pair, height))
        print height, " snap: ", zzz[snap], " pair: ",str(pair[0]),"-",str(pair[1])
        height+=1
    #     for n in pair:
    #         loc-=0.1
    #         plt.text(10, loc,str(n)+": "+pr_num(sub_mass[n]))

plt.xlim(10,4000)
# plt.legend()
save_figure(outdir+"pairs_vw_3")
plt.clf()

plt.figure(2)
# plt.legend()
plt.ylim(-0.1, height+1)
save_figure(outdir+"pairs_scatter_3")
plt.clf()

table=np.logspace(np.log10(np.min(medians)),np.log10(np.max(medians)),5)
center = np.array([(table[i]+table[i+1])/2. for i in range(0,np.size(table)-1)])
nn = np.histogram(medians,table)[0]
plt.plot(center, nn, ls="-", color="black")
plt.errorbar(center,nn,xerr=[center-table[:-1],table[1:]-center],fmt='.', color="black")
save_figure(outdir+"pairs_pdf_2")
plt.clf()
