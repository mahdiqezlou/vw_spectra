# -*- coding: utf-8 -*-
"""Contains the plotting-specific functions specific to the velocity width analysis."""

from __future__ import print_function
import plot_spectra as ps
import vw_spectra as vw
import numpy as np
import kstest as ks
import matplotlib.pyplot as plt
try:
    xrange(1)
except NameError:
    xrange = range

def _bootstrap_sample(vel_data, v_table, samples, error):
    """Generate a Monte Carlo error sample of the differential distribution."""
    # Generate some Monte Carlo samples where each element is perturbed by
    # a Gaussian, sigma given by error.
    index = np.random.random_integers(0, np.size(vel_data)-1, samples)
    bootstrap = vel_data[index]
    if error > 0.:
        bootstrap += np.random.normal(0,error,size=samples)
    nn = np.histogram(bootstrap,v_table)[0]
    return nn

class VWPlotSpectra(ps.PlottingSpectra, vw.VWSpectra):
    """Extends PlottingSpectra with velocity width specific code."""
    def plot_vel_width(self, elem, ion, dv=0.17, color="red", ls="-"):
        """Plot the velocity widths of this snapshot
        Parameters:
            elem - element to use
            ion - ionisation state: 1 is neutral.
            dv - bin spacing
        """
        (vbin, vels) = self.vel_width_hist(elem, ion, dv)
        plt.semilogx(vbin, vels, color=color, lw=3, ls=ls,label=self.label)

    def plot_cum_vel_width(self, elem, ion, norm, dv=0.1, color="red", ls="-"):
        """Plot the velocity widths of this snapshot
        Parameters:
            elem - element to use
            ion - ionisation state: 1 is neutral.
            dv - bin spacing
        """
        (vbin, vels) = self.vel_width_hist(elem, ion, dv)
        cvels = np.cumsum(vels)
        cvels = cvels*norm/cvels[-1]
        plt.semilogx(vbin, cvels, color=color, lw=3, ls=ls,label=self.label)

    def plot_cum_f_peak(self, elem, ion, norm, dv=0.01, color="red", ls="-"):
        """Plot the velocity widths of this snapshot
        Parameters:
            elem - element to use
            ion - ionisation state: 1 is neutral.
            dv - bin spacing
        """
        (vbin, vels) = self.f_peak_hist(elem, ion, dv)
        cvels = np.cumsum(vels)
        cvels = cvels*norm/cvels[-1]
        plt.plot(vbin, cvels, color=color, lw=3, ls=ls,label=self.label)
        plt.xlabel(r"$f_\mathrm{edg}$")

    def plot_f_meanmedian_errors(self, elem, ion, samples, cumulative=False, nv_table = 11, color="red"):
        """Plot 68% contour for error on the fmm distribution"""
        f_peak = self.vel_mean_median(elem, ion)
        ind = self.get_filt(elem, ion)
        f_peak = f_peak[ind]
        v_table=np.linspace(0,1,nv_table)
        self._plot_errors(f_peak, v_table, samples, 0., cumulative, False, color)

    def plot_f_peak_errors(self, elem, ion, samples, cumulative=False, nv_table=11, color="red"):
        """Plot 68% contour for error on the fpeak distribution"""
        f_peak = self.vel_peak(elem, ion)
        ind = self.get_filt(elem, ion)
        f_peak = f_peak[ind]
        v_table=np.linspace(0,1,nv_table)
        self._plot_errors(f_peak, v_table, samples, 0., cumulative, False, color)

    def plot_eq_width_errors(self, elem, ion, line, samples, cumulative=False, min_width = -1.6, nv_table=11, color="red"):
        """Plot 68% contour for error on the fpeak distribution"""
        eq_width = self.equivalent_width(elem, ion, line)
        ind = self.get_filt(elem, ion)
        eq_width = eq_width[ind]
        v_table = np.logspace(min_width, np.log10(np.max(eq_width)), nv_table)
        self._plot_errors(np.log10(eq_width), np.log10(v_table), samples, 0.05, cumulative, False, color)

    def plot_vw_errors(self, elem, ion, samples, cumulative=False, nv_table=11, color="red"):
        """Plot 68% contour for error on the velocity width distribution"""
        vel_width = self.vel_width(elem, ion)
        ind = self.get_filt(elem, ion)
        vel_width = vel_width[ind]
        v_table=np.logspace(1,np.log10(np.max(vel_width)+10),nv_table)
        self._plot_errors(vel_width, v_table, samples, 5, cumulative, True, color)

    def _plot_errors(self, vel_data, v_table, samples, error, cumulative=False, lognorm=True, color="red"):
        """Find and plot a 68% contour for a subsample of size samples, by Monte Carlo."""
        vbin = np.array([(v_table[i]+v_table[i+1])/2. for i in range(0,np.size(v_table)-1)])
        #Get a subsample
        cdfs = np.array([_bootstrap_sample(vel_data, v_table, samples, error) for _ in xrange(10000)])
        if cumulative:
            cdfs = np.cumsum(cdfs, axis=1)
            norm = 1
        else:
            if lognorm:
                v_table = np.log10(v_table)
            norm = samples * np.array([(-v_table[i]+v_table[i+1]) for i in xrange(np.size(v_table)-1)])

        lower = np.percentile(cdfs, 16, axis=0)/norm
        upper = np.percentile(cdfs, 84, axis=0)/norm
        plt.fill_between(vbin, lower, upper, color=color, alpha=0.3)

    def plot_f_meanmedian(self, elem, ion, dv=0.06, color="red", ls="-"):
        """
        Plot an f_mean_median histogram
        For args see plot_vel_width
        """
        (vbin, vels) = self.f_meanmedian_hist(elem, ion, dv)
        plt.plot(vbin, vels, color=color, lw=3, ls=ls,label=self.label)
        plt.xlabel(r"$f_\mathrm{mm}$")

    def plot_f_peak(self, elem, ion, dv=0.06, color="red", ls="-"):
        """
        Plot an f_peak histogram
        For args see plot_vel_width
        """
        (vbin, vels) = self.f_peak_hist(elem, ion, dv)
        plt.plot(vbin, vels, color=color, lw=3, ls=ls,label=self.label)
        plt.xlabel(r"$f_\mathrm{edg}$")

    def plot_sep_frac(self,elem = "Si", ion = 2, thresh = 1e-1, mindist = 15, dv = 0.2, color="blue", ls="-"):
        """
        Plots the fraction of spectra in each velocity width bin which are separated.
        Threshold is as a percentage of the maximum value.
        mindist is in km/s
        """
        sep = self.get_separated(elem, ion, thresh,mindist)
        vels = self.vel_width(elem, ion)
        ind = self.get_filt(elem, ion)
        v_table = 10**np.arange(1, 3, dv)
        vbin = np.array([(v_table[i]+v_table[i+1])/2. for i in range(0,np.size(v_table)-1)])
        hist1 = np.histogram(vels[ind], v_table)
        hist2 = np.histogram(vels[ind][sep],v_table)
        hist1[0][np.where(hist1[0] == 0)] = 1
        plt.semilogx(vbin, hist2[0]/(1.*hist1[0]), color=color, ls=ls, label=self.label)

    def plot_vel_width_breakdown(self, elem = "Si", ion = 2, dv = 0.1):
        """
        Plots the fraction of the total velocity width histogram in a series of virial velocity bins
        """
        #Find velocity width
        vels = self.vel_width(elem, ion)
        ii = self.get_filt(elem, ion)
        self._plot_breakdown(vels,ii,(0, 60, 120), (60, 120, 900), ("< 60", "60-120", "> 120"),dv)
        plt.xlabel(r"$v_\mathrm{90}$ (km s$^{-1}$)")
        plt.ylim(0,1)


    def plot_f_peak_breakdown(self, elem = "Si", ion = 2, dv = 0.05):
        """
        Plots the fraction of the total fedge histogram in a series of virial velocity bins
        """
        #Find velocity width
        vels = self.vel_peak(elem, ion)
        ii = self.get_filt(elem, ion)
        self._plot_breakdown(vels,ii,(0, 50), (50, 900), ("< 50", "> 50"),dv, False)
        plt.xlabel(r"$f_\mathrm{edg}$")
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.legend(loc=1,ncol=2)

    def plot_mult_halo_frac(self,elem = "Si", ion = 2, dv = 0.2, color="blue", ls="-"):
        """
        Plots the fraction of spectra in each velocity width bin which are separated.
        Threshold is as a percentage of the maximum value.
        mindist is in km/s
        """
        #Find velocity width
        (halos, subhalos) = self.find_nearby_halos()
        vels = self.vel_width(elem, ion)
        ii = self.get_filt(elem, ion)
        #Find virial velocity
        (halo, _) = self.find_nearest_halo()
        ind = np.where(halo[ii] > 0)
#         virial = np.ones_like(halo, dtype=np.double)
#         virial[ind] = self.virial_vel(halo[ind])
        vwvir = vels[ii][ind]  #/virial[ind]
        #Make bins
        v_table = 10**np.arange(np.min(np.log10(vwvir)),np.max(np.log10(vwvir)) , dv)
        vbin = np.array([(v_table[i]+v_table[i+1])/2. for i in range(0,np.size(v_table)-1)])
        #Histogram of vel width / virial vel
        hist1 = np.histogram(vwvir, v_table)
        hist1[0][np.where(hist1[0] == 0)] = 1
        #Find places with multiple halos
        subhalo_parent = [list(self.sub_sub_index[ss]) for ss in subhalos]
        allh = np.array([list(set(subhalo_parent[ii] + halos[ii])) for ii in xrange(self.NumLos)])
        indmult = np.where([len(aa) > 1 for aa in allh[ind]])
        histmult = np.histogram(vwvir[indmult],v_table)
        plt.semilogx(vbin, histmult[0]/(1.*hist1[0]), color=color, ls=ls, label=self.label)

    def plot_Z_vs_vel_width(self,elem="Si", ion=2, color="blue",color2="darkblue"):
        """Plot the correlation between metallicity and velocity width"""
        vel = self.vel_width(elem, ion)
        met = self.get_metallicity()
        #Ignore objects too faint to be seen
        ind2 = np.where(met > 1e-4)
        met = met[ind2]
        vel = vel[ind2]
        self._plot_2d_contour(vel, met, 10, "Z vel sim", color, color2, fit=True)
        plt.xlim(10,2e3)
        plt.ylabel(r"$\mathrm{Z} / \mathrm{Z}_\odot$")
        plt.xlabel(r"$v_\mathrm{90}$ (km s$^{-1}$)")

    def plot_vel_vs_mass(self,elem, ion, color="blue",color2="darkblue"):
        """Plot the correlation between mass and metallicity, with a fit"""
        vel = self.vel_width(elem, ion)
        self._plot_xx_vs_mass(vel, "vel",color,color2)

    def kstest(self, Zdata, veldata, elem="Si", ion=2):
        """Find the 2D KS test value of the vel width and log metallicity
           with respect to an external dataset, veldata and Z data"""
        met = self.get_metallicity()
        ind = self.get_filt(elem, ion)
        met = np.log10(met[ind])
        vel = np.log10(self.vel_width(elem, ion)[ind])
        data2 = np.array([met,vel]).T
        data = np.array([np.log10(Zdata), np.log10(veldata)]).T
        return ks.ks_2d_2samp(data,data2)

    def plot_virial_vel_vs_vel_width(self,elem, ion,color="red", ls="-", label="", dm=0.1):
        """Plot a histogram of the velocity widths vs the halo virial velocity"""
        (halos, _) = self.find_nearest_halo()
        ind = self.get_filt(elem,ion)
        f_ind = np.where(halos[ind] != -1)
        vel = self.vel_width(elem, ion)[ind][f_ind]
        virial = self.virial_vel(halos[ind][f_ind])+0.1
        vvvir = vel/virial
        m_table = 10**np.arange(np.log10(np.min(vvvir)), np.log10(np.max(vvvir)), dm)
        mbin = np.array([(m_table[i]+m_table[i+1])/2. for i in range(0,np.size(m_table)-1)])
        pdf = np.histogram(np.log10(vvvir),np.log10(m_table), density=True)[0]
        print("median v/vir: ",np.median(vvvir))
        plt.semilogx(mbin, pdf, color=color, ls=ls, label=label)
        return (mbin, pdf)

    def plot_vbars(self, tau):
        """Plot the vertical bars marking the velocity widths"""
        (low, high) = self._vel_width_bound(tau)
        xaxis = np.arange(0,np.size(tau))*self.dvbin - (high+low)/2
        if high - low > 0:
            plt.plot([xaxis[0]+low,xaxis[0]+low],[-1,20], color="green")
            plt.plot([xaxis[0]+high,xaxis[0]+high],[-1,20],color="red")
        if high - low > 30:
            tpos = xaxis[0]+low+5
        else:
            tpos = xaxis[0]+high+5
        if high - low > 60:
            tpos = xaxis[0]+low+25
        if high - low > 150:
            tpos = xaxis[0]+low+35
        ypos = np.max(tau) -0.2
        if np.max(tau) < 0.8:
            ypos = 0.7
        elif np.max(tau) > 4.:
            ypos = 3.5
        plt.text(tpos,ypos,r"$\Delta v_{90} = "+str(np.round(high-low,1))+r"$", size=14)
        xlims = (np.max((xaxis[0],xaxis[0]+low-20)),np.min((xaxis[-1],xaxis[0]+high+20)))
        return (xaxis, xlims)

    def plot_spectrum(self, elem, ion, line, num, flux=True):
        """Plot an spectrum, centered on the maximum tau,
           and marking the 90% velocity width.
           offset: offset in km/s for the x-axis labels"""
        if line == -1:
            tau_no = self.get_observer_tau(elem, ion, num, noise=False)
            tau = self.get_observer_tau(elem, ion, num, noise=True)
        else:
            tau_no = self.get_tau(elem, ion, line, num, noise=False)
            tau = self.get_tau(elem, ion, line, num, noise=True)
        (low, high, offset) = self.find_absorber_width(elem, ion)
        tau_l = np.roll(tau_no, offset[num])[low[num]:high[num]]
        (xaxis, xlims) = self.plot_vbars(tau_l)
        tau_l = np.roll(tau, offset[num])[low[num]:high[num]]
        return self.plot_spectrum_raw(tau_l,xaxis, xlims, flux)

    def get_filt(self, elem, ion, thresh = 100):
        """
        Get an index list to exclude spectra where the ion is too small, or velocity width < 20
        thresh - observable density threshold
        """
        return vw.VWSpectra.get_filt(self, elem, ion, thresh)
