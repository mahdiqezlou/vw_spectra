# -*- coding: utf-8 -*-
"""Class to gather and analyse various metal line statistics"""

import numpy as np
import hdfsim
import h5py
import subfindhdf
import os.path as path
import vw_spectra
import math

class SubHaloSpectra(vw_spectra.VWSpectra):
    """Generate metal line spectra from simulation snapshot"""
    def __init__(self,num, base, subhalopair, repeat = 1, res = 1., savefile=None, savedir=None, cdir=None):
        if savedir == None:
            savedir = path.join(base,"snapdir_"+str(num).rjust(3,'0'))
        if savefile == None:
            savefile = "subhalo_spectra_pairs.hdf5"
        shape = np.shape(subhalopair)
        assert shape[1] == 2
        nhalo = shape[0]
        self.savefile = path.join(savedir,savefile)
        #Load halos to push lines through them
        f = hdfsim.get_file(num, base, 0)
        self.OmegaM = f["Header"].attrs["Omega0"]
        self.box = f["Header"].attrs["BoxSize"]
        self.npart=f["Header"].attrs["NumPart_Total"]+2**32*f["Header"].attrs["NumPart_Total_HighWord"]
        f.close()
        #Get the subhalo list
        subs=subfindhdf.SubFindHDF5(base, num)
        self.sub_cofm=np.array(subs.get_sub("SubhaloPos"), dtype=np.float64)
        #In solar masses
        self.sub_mass=np.array(subs.get_sub("SubhaloMass"))*1e10
        #r in kpc/h (comoving).
        self.sub_radii = np.array(subs.get_sub("SubhaloHalfmassRad"))
        self.NumLos = nhalo*repeat
        self.subhalopair=subhalopair
        #Now we have the sightlines
        dist=np.abs(self.sub_cofm[subhalopair[0]]- self.sub_cofm[subhalopair[1]])
        axis = np.repeat(np.where(dist == np.min(dist))[0] + 1, nhalo*repeat)
        self.repeat = repeat
        #Re-seed for repeatability
        np.random.seed(23)
        cofm = np.empty([nhalo*repeat,3])
        total = 0
        for subp in subhalopair:
            cofm[total:total+repeat] = self.get_cofm_single_pair(subp, repeat)
            total+=repeat

        assert np.shape(cofm) == (np.shape(axis)[0],3)

        vw_spectra.VWSpectra.__init__(self,num, base, cofm, axis, res, savefile=self.savefile, savedir=savedir,reload_file=True, cdir=cdir)

    def get_cofm_single_pair(self, subhalopair, nlines):
        """Find a bunch more sightlines"""
        cofm = self.sub_cofm[subhalopair]
        #Perturb the sightlines within a sphere of the overlap between the halos.
        #Centered on midpoint between central regions
        center = (self.sub_cofm[subhalopair[0]] + self.sub_cofm[subhalopair[1]])/2.
        #Of radius such that it touches the two central halo regions
        #Maybe also encompass the smaller of the halos?
        radius = np.sqrt(np.sum((self.sub_cofm[subhalopair[0]] - self.sub_cofm[subhalopair[1]])**2/2.))
        #Generate random sphericals
        theta = 2*math.pi*np.random.random_sample(nlines)-math.pi
        phi = 2*math.pi*np.random.random_sample(nlines)
        rr = radius*np.random.random_sample(nlines)
        #Add them to halo centers
        cofm = np.repeat([center], nlines,axis=0)
        cofm[:,0]+=rr*np.sin(theta)*np.cos(phi)
        cofm[:,1]+=rr*np.sin(theta)*np.sin(phi)
        cofm[:,2]+=rr*np.cos(theta)
        return cofm

    def save_file(self):
        """
        Save additional halo data to the savefile
        """
        try:
            f=h5py.File(self.savefile,'a')
        except IOError:
            raise IOError("Could not open ",self.savefile," for writing")
        grp = f.create_group("halos")
        grp["radii"] = self.sub_radii
        grp["cofm"] = self.sub_cofm
        grp["mass"] = self.sub_mass
        grp["subhalopair"] = self.subhalopair
        grp.attrs["repeat"] = self.repeat
        grp.attrs["NumLos"] = self.NumLos
        f.close()
        vw_spectra.VWSpectra.save_file(self)

    def load_savefile(self,savefile=None):
        """Load data from a file"""
        #Name of savefile
        f=h5py.File(savefile,'r')
        grp = f["halos"]
        self.sub_radii = np.array(grp["radii"])
        self.sub_cofm = np.array(grp["cofm"])
        self.sub_mass = np.array(grp["mass"])
        self.subhalopair = list(grp["subhalopair"])
        self.repeat = grp.attrs["repeat"]
        self.NumLos = grp.attrs["NumLos"]
        f.close()
        vw_spectra.VWSpectra.load_savefile(self, savefile)


    def find_associated_halo(self, num):
        """Find the halo sightline num is associated with"""
        nh = num /self.repeat
        return (nh, self.sub_mass[nh], self.sub_cofm[nh,:], self.sub_radii[nh])

    def line_offsets(self):
        """Find the minimum distance between each line and its parent halo"""
        offsets = np.zeros(self.NumLos)
        hcofm = np.array([self.find_associated_halo(ii)[2] for ii in xrange(self.NumLos)])
        hrad = np.array([self.find_associated_halo(ii)[3] for ii in xrange(self.NumLos)])
        hpos = self.get_spectra_proj_pos(cofm=hcofm)
        lpos = self.get_spectra_proj_pos()
        offsets = np.sqrt(np.sum((hpos-lpos)**2,axis=1))/hrad
        return offsets

    def load_halo(self):
        """Do nothing - halos already loaded"""
        return

    def replace_not_DLA(self, thresh=10**20.3):
        """Do nothing"""
        print thresh
        return

def get_lines(halo):
    """Get the useful quantitites"""
    halo.get_density("H",1)
    #SiII 1260
    halo.get_tau("Si",2,1260, force_recompute=True)
    halo.get_tau("Si",2,1526, force_recompute=True)
    halo.get_tau("Si",2,1808, force_recompute=True)
    halo.get_observer_tau("Si",2, force_recompute=True)
    halo.get_tau("H",1,1215 , force_recompute=True)
    halo.get_density("Si",2)
    halo.get_density("Z",-1)
    halo.get_density("H",-1)
    halo.get_velocity("H",1)

if __name__ == "__main__":
    base = "/home/spb/data/Cosmo/Cosmo5_V6/L10n512/output/"
    pair = ([1,4], [835, 837], [1556, 1558])
    ahs = SubHaloSpectra(3,base,pair, repeat = 1000)
    get_lines(ahs)
    ahs.save_file()

    pair = ([2531,2532], [3240, 3241])
    ahs = SubHaloSpectra(4,base,pair, repeat = 1000)
    get_lines(ahs)
    ahs.save_file()

    pair = ([3775, 3776],)
    ahs = SubHaloSpectra(5,base,pair, repeat = 1000)
    get_lines(ahs)
    ahs.save_file()

    # big box z=2.5
    base = "/n/ghernquist/Illustris/Runs/Illustris-1"
    pairs63 = np.loadtxt("pairs063.txt")
    pair = zip(pairs63[:,0], pairs63[:,1])
    ahs = SubHaloSpectra(63,base,pair, repeat = 500)
    get_lines(ahs)
    ahs.save_file()
    # big box z=2
    pairs68 = np.loadtxt("pairs068.txt")
    pair = zip(pairs68[:,0], pairs68[:,1])
    ahs = SubHaloSpectra(68,base,pair, repeat = 500)
    get_lines(ahs)
    ahs.save_file()
