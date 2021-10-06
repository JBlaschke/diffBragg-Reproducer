from __future__ import absolute_import, division, print_function
import sys
from mpi4py import MPI
COMM = MPI.COMM_WORLD
import numpy as np
import os
import time
import signal
import pandas
import logging
from collections import Counter
import resource
import platform

SEED = 11
np.random.seed(SEED)
if COMM.rank==0:
    print("USING %d MPI RANKS" % (COMM.size))


from cctbx import miller, sgtbx
from libtbx.phil import parse
from cctbx.array_family import flex

MAKEFAIL = "--makeFail" in "".join(sys.argv)
def get_memory_usage():
    '''Return memory used by the process in MB'''
    if MAKEFAIL:
        # getrusage returns kb on linux, bytes on mac
        units_per_mb = 1024
        if platform.system() == "Darwin":
            units_per_mb = 1024*1024
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / units_per_mb
    else:
        return np.random.random()*1000


class DataModeler:
    def __init__(self):
        pass


class StageTwoRefiner2:

    def __init__(self, shot_modelers, sgsymbol, params):

        self.params = params
        self.min_multiplicity = self.params.refiner.stage_two.min_multiplicity
        self.trad_conv_eps = self.params.refiner.tradeps
        self.calc_curvatures = self.params.refiner.curvatures
        self.break_signal = self.params.refiner.break_signal
        self.output_dir = self.params.refiner.io.output_dir  # directory to dump progress files, these can be used to restart simulation later
        self.save_model_freq = self.params.refiner.stage_two.save_model_freq
        self.use_nominal_h = self.params.refiner.stage_two.use_nominal_hkl

        self.saveZ_freq = self.params.refiner.stage_two.save_Z_freq  # save Z-score data every N iterations
        self.break_signal = None  # check for this signal during refinement, and break refinement if signal is received (see python signal module) TODO: make work with MPI
        self.save_model = False  # whether to save the model
        self.idx_from_asu = {}  # maps global fcell index to asu hkl
        self.asu_from_idx = {}  # maps asu hkl to global fcell index
        self.rescale_params = True  # whether to rescale parameters during refinement  # TODO this will always be true, so remove the ability to disable
        self.request_diag_once = False  # LBFGS refiner property
        self.min_multiplicity = 1  # only refine a spots Fhkl if multiplicity greater than this number
        self.restart_file = None  # output file from previous run refinement
        self.trial_id = 0  # trial id in case multiple trials are run in sequence
        self.x_init = None  # used to restart the refiner (e.g. self.x gets updated with this)
        self.log_fcells = True  # to refine Fcell using logarithms to avoid negative Fcells
        self.refine_crystal_scale = False  # whether to refine the crystal scale factor
        self.refine_Fcell = False  # whether to refine Fhkl for each shoebox ROI
        self.use_curvatures_threshold = 7  # how many positive curvature iterations required before breaking, after which simulation can be restart with use_curvatures=True
        self.verbose = True  # whether to print during iterations
        self.iterations = 0  # iteration counter , used internally
        self.shot_ids = None  # for global refinement ,
        self.log2pi = np.log(np.pi*2)

        self._sig_hand = None  # method for handling the break_signal, e.g. SIGHAND.handle defined above (theres an MPI version in global_refiner that overwrites this in stage 2)
        self._is_trusted = None  # used during refinement, 1-D array or trusted pixels corresponding to the pixels in the ROI

        self.rank = COMM.rank

        self.Modelers = shot_modelers
        self.shot_ids = sorted(self.Modelers.keys())
        self.n_shots = len(shot_modelers)
        self.n_shots_total = COMM.bcast(COMM.reduce(self.n_shots))
        LOGGER.debug("Loaded %d shots across all ranks" % self.n_shots_total)
        self.f_vals = []  # store the functional over time

        self._ncells_id = 9  # diffBragg internal index for Ncells derivative manager
        self._detector_distance_id = 10  # diffBragg internal index for detector_distance derivative manager
        self._panelRotO_id = 14  # diffBragg internal index for derivative manager
        self._panelRotF_id = 17  # diffBragg internal index for derivative manager
        self._panelRotS_id = 18  # diffBragg internal index for derivative manager
        self._panelX_id = 15  # diffBragg internal index for  derivative manager
        self._panelY_id = 16  # diffBragg internal index for  derivative manager
        self._fcell_id = 11  # diffBragg internal index for Fcell derivative manager
        self._eta_id = 19  # diffBragg internal index for eta derivative manager
        self._lambda0_id = 12  # diffBragg interneal index for lambda derivatives
        self._lambda1_id = 13  # diffBragg interneal index for lambda derivatives
        self._ncells_def_id = 21

        self.symbol = "P6522" #sgsymbol
        self.space_group = sgtbx.space_group(sgtbx.space_group_info(symbol=self.symbol).type().hall_symbol())
        self.I_AM_ROOT = COMM.rank==0

    def __call__(self, *args, **kwargs):
        _, _ = self.compute_functional_and_gradients()
        return self.x, self._f, self._g, self.d

    @property
    def n(self):
        """LBFGS property"""
        return len(self.x)

    @property
    def n_global_fcell(self):
        return len(self.idx_from_asu)

    @property
    def image_shape(self):
        panelXdim, panelYdim = self.S.detector[0].get_image_size()
        Npanels = len(self.S.detector)
        return Npanels, panelYdim, panelXdim

    @property
    def x(self):
        """LBFGS parameter array"""
        return self._x

    @x.setter
    def x(self, val):
        self._x = val

    def _check_keys(self, shot_dict):
        """checks that the dictionary keys are the same"""
        if not sorted(shot_dict.keys()) == self.shot_ids:
            raise KeyError("input data funky, check GlobalRefiner inputs")
        return shot_dict

    def _evaluate_averageI(self):
        """model_Lambda means expected intensity in the pixel"""
        self.model_Lambda = self.Modelers[self._i_shot].all_background + self.model_bragg_spots

    def make_output_dir(self):
        if self.I_AM_ROOT and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.Zdir = os.path.join(self.output_dir, "Z")
        self.model_dir = os.path.join(self.output_dir, "model")
        for dirname in (self.Zdir, self.model_dir):
            if self.I_AM_ROOT and not os.path.exists(dirname):
                os.makedirs(dirname)
        COMM.barrier()

    def _setup(self):
        # Here we go!  https://youtu.be/7VvkXA6xpqI
        LOGGER.info("Setup begins!")
        if self.refine_Fcell and not self.asu_from_idx:
            raise ValueError("Need to supply a non empty asu from idx map")
        if self.refine_Fcell and not self.idx_from_asu:  # # TODO just derive from its inverse
            raise ValueError("Need to supply a non empty idx from asu map")

        self.make_output_dir()

        self.shot_mapping = self._get_shot_mapping()
        self.n_total_shots = len(self.shot_mapping)

        test_shot = self.shot_ids[0]
        N_PARAM_PER_SHOT = 2
        self.n_ucell_param =  2 #len(self.Modelers[test_shot].PAR.ucell_man.variables)
        self.n_total_params = self.n_total_shots*N_PARAM_PER_SHOT + self.n_global_fcell

        self.spot_scale_xpos = {}
        self.Bfactor_xpos = {}
        for shot_id in self.shot_ids:
            self.spot_scale_xpos[shot_id] = self.shot_mapping[shot_id]*N_PARAM_PER_SHOT
            self.Bfactor_xpos[shot_id] = self.shot_mapping[shot_id]*N_PARAM_PER_SHOT + 1

        LOGGER.info("--0 create an Fcell mapping")
        if self.refine_Fcell:
            self._make_p1_equiv_mapping()

        # Make a mapping of panel id to parameter index and backwards
        self.pid_from_idx = {}
        self.idx_from_pid = {}

        self.x = flex.double(np.ones(self.n_total_params))
        LOGGER.info("--Setting up per shot parameters")

        self.fcell_xstart = self.n_total_shots*N_PARAM_PER_SHOT

        self.hkl_totals = []
        if self.refine_Fcell:
            for i_shot in self.shot_ids:
                for i_h, h in enumerate(self.Modelers[i_shot].Hi_asu):
                    self.hkl_totals.append(self.idx_from_asu[h])
            self.hkl_totals = self._MPI_reduce_broadcast(self.hkl_totals)

        self._MPI_setup_global_params()
        self._MPI_sync_fcell_parameters()
        # reduce then broadcast fcell
        LOGGER.info("--combining parameters across ranks")
        self._MPI_sync_hkl_freq()

        if self.x_init is not None:
            LOGGER.info("Initializing with provided x_init array")
            self.x = self.x_init
        elif self.restart_file is not None:
            LOGGER.info("Restarting from parameter file %s" % self.restart_file)
            self.x = flex.double(np.load(self.restart_file)["x"])

        self._MPI_barrier()
        LOGGER.info("Setup ends!")

    def _get_shot_mapping(self):
        """each modeled shot maps to an integer along interval [0,Nshots) """
        all_shot_ids = COMM.gather(self.shot_ids)
        shot_mapping = None
        if COMM.rank == 0:
            unique_shot_ids = set([sid for shot_ids in all_shot_ids for sid in shot_ids])
            shot_mapping = {shot_id: i_shot for i_shot, shot_id in enumerate(unique_shot_ids)}
        shot_mapping = COMM.bcast(shot_mapping)
        return shot_mapping

    def _make_p1_equiv_mapping(self):
        self.num_equivs_for_i_fcell = {}
        self.update_indices = []
        for i_fcell in range(self.n_global_fcell):
            hkl_asu = self.asu_from_idx[i_fcell]

            equivs = [i.h() for i in miller.sym_equiv_indices(self.space_group, hkl_asu).indices()]
            self.num_equivs_for_i_fcell[i_fcell] = len(equivs)
            self.update_indices += equivs
        self.update_indices = flex.miller_index(self.update_indices)

    def _MPI_setup_global_params(self):
        if self.I_AM_ROOT:
            LOGGER.info("--2 Setting up global parameters")
            if self.output_dir is not None:
                np.save(os.path.join(self.output_dir, "f_asu_map"), self.asu_from_idx)

            self._setup_fcell_params()

    def _setup_fcell_params(self):
        if self.refine_Fcell:
            LOGGER.info("----loading fcell data")
            # this is the number of observations of hkl (accessed like a dictionary via global_fcell_index)
            LOGGER.info("---- -- counting hkl totes")
            LOGGER.info("compute HKL multiplicity")
            self.hkl_frequency = Counter(self.hkl_totals)
            LOGGER.info("save HKL multiplicity")
            np.save(os.path.join(self.output_dir, "f_asu_multi"), self.hkl_frequency)
            LOGGER.info("Done ")

            LOGGER.info("make an Fhkl map")
            LOGGER.info("make fcell_init")
            self.fcell_init_from_i_fcell = np.random.random(100)#np.array([ma_map[self.asu_from_idx[i_fcell]] for i_fcell in range(self.n_global_fcell)])
            self.fcell_sigmas_from_i_fcell = 1 #self.params.sigmas.Fhkl
            LOGGER.info("DONE make fcell_init")

    def _get_detector_distance_val(self, i_shot):
        return self.Modelers[i_shot].PAR.detz_shift.init

    def _get_m_val(self, i_shot):
        vals = [self.Modelers[i_shot].PAR.Nabc[i_N].init for i_N in range(3)]
        return vals

    def _get_spot_scale(self, i_shot):
        xval = self.x[self.spot_scale_xpos[i_shot]]
        PAR = self.Modelers[i_shot].PAR
        sig = PAR.Scale.sigma
        init = PAR.Scale.init
        val = sig*(xval-1) + init
        return val

    def _get_bfactor(self, i_shot):
        xval = self.x[self.Bfactor_xpos[i_shot]]
        PAR = self.Modelers[i_shot].PAR
        sig = PAR.B.sigma
        init = PAR.B.init
        val = sig*(xval-1) + init
        return val

    def _run_diffBragg_current(self):
        LOGGER.info("run diffBragg for shot %d" % self._i_shot)
        pfs = self.Modelers[self._i_shot].pan_fast_slow
        if self.use_nominal_h:
            nom_h = self.Modelers[self._i_shot].all_nominal_hkl
            self.D.add_diffBragg_spots(pfs, nom_h)
        else:
            self.D.add_diffBragg_spots(pfs)
        LOGGER.info("finished diffBragg for shot %d" % self._i_shot)

    def _store_updated_Fcell(self):
        if not self.refine_Fcell:
            return
        xvals = self.x[self.fcell_xstart: self.fcell_xstart+self.n_global_fcell]
        if self.rescale_params and self.log_fcells:
            sigs = self.fcell_sigmas_from_i_fcell
            inits = self.fcell_init_from_i_fcell
            if self.log_fcells:
                vals = np.exp(sigs*(xvals - 1))*inits
            else:
                vals = sigs*(xvals - 1) + inits
                vals[vals < 0] = 0
        else:
            if self.log_fcells:
                vals = np.exp(xvals)
            else:
                vals = xvals
                vals [vals < 0] = 0
        self._fcell_at_i_fcell = vals

    def _update_Fcell(self):
        if not self.refine_Fcell:
            return
        update_amps = []
        for i_fcell in range(self.n_global_fcell):
            new_Fcell_amplitude = self._fcell_at_i_fcell[i_fcell]
            update_amps += [new_Fcell_amplitude] * self.num_equivs_for_i_fcell[i_fcell]

        update_amps = flex.double(update_amps)
        self.S.D.quick_Fhkl_update((self.update_indices, update_amps))

    def _update_spectra_coefficients(self):
        pass

    def _update_eta(self):
        pass

    def _update_sausages(self):
        pass

    def _update_rotXYZ(self):
        pass

    def _update_ncells(self):
        vals = self._get_m_val(self._i_shot)
        self.D.set_ncells_values(tuple(vals))

    def _update_ncells_def(self):
        pass

    def _update_dxtbx_detector(self):
        shiftZ = self._get_detector_distance_val(self._i_shot)
        self.S.D.shift_origin_z(self.S.detector,  shiftZ)

    def _pre_extract_deriv_arrays(self):
        npix = len(self.Modelers[self._i_shot].all_data)
        self._model_pix = self.D.raw_pixels_roi[:npix].as_numpy_array()

        if self.refine_Fcell:
            dF = self.D.get_derivative_pixels(self._fcell_id)
            self._extracted_fcell_deriv = dF[:npix].as_numpy_array()
            if self.calc_curvatures:
                d2F = self.D.get_second_derivative_pixels(self._fcell_id)
                self._extracted_fcell_second_deriv = d2F[:npix].as_numpy_array()

    def _extract_Fcell_derivative_pixels(self):
        # TODO pre-extract
        self.fcell_deriv = self.fcell_second_deriv = 0
        if self.refine_Fcell:
            SG = self.scale_fac
            self.fcell_deriv = SG*(self._extracted_fcell_deriv)
            # handles Nan's when Fcell is 0 for whatever reason
            if self.calc_curvatures:
                self.fcell_second_deriv = SG*self._extracted_fcell_second_deriv

    def _extract_pixel_data(self):
        self.model_bragg_spots = self.scale_fac*self._model_pix
        self._extract_Fcell_derivative_pixels()

    def _update_ucell(self):
        self.D.Bmatrix = self.Modelers[self._i_shot].PAR.Bmatrix

    def _update_umatrix(self):
        self.D.Umatrix = self.Modelers[self._i_shot].PAR.Umatrix

    def _update_beams(self):
        # sim_data instance has a nanoBragg beam object, which takes spectra and converts to nanoBragg xray_beams
        self.S.beam.spectrum = self.Modelers[self._i_shot].spectra
        self.D.xray_beams = self.S.beam.xray_beams

    def compute_functional_and_gradients(self):
        t = time.time()
        out = self._compute_functional_and_gradients()
        t = time.time()-t
        LOGGER.info("TOok %.4f sec to compute functional and grad" % t)
        return out

    def _compute_functional_and_gradients(self):
        LOGGER.info("BEGIN FUNC GRAD ; iteration %d" % self.iterations)

        self.target_functional = 0

        self.grad = flex.double(self.n_total_params)
        if self.calc_curvatures:
            self.curv = flex.double(self.n_total_params)

        LOGGER.info("start update Fcell")
        self._store_updated_Fcell()
        self._update_Fcell()  # update the structure factor with the new x
        LOGGER.info("done update Fcell")
        self._MPI_save_state_of_refiner()
        self._update_spectra_coefficients()  # updates the diffBragg lambda coefficients if refinining spectra

        tshots = time.time()

        LOGGER.info("Iterate over %d shots" % len(self.shot_ids))
        self._shot_Zscores = []
        save_model = self.save_model_freq is not None and self.iterations % self.save_model_freq == 0
        if save_model:
            self._save_model_dir = os.path.join(self.model_dir, "iter%d" % self.iterations)

            if COMM.rank == 0 and not os.path.exists(self._save_model_dir):
                os.makedirs(self._save_model_dir)
            COMM.barrier()

        for self._i_shot in self.shot_ids:
            self.scale_fac = self._get_spot_scale(self._i_shot)**2
            self.b_fac = self._get_bfactor(self._i_shot)

            # TODO: Omatrix update? All crystal models here should have the same to_primitive operation, ideally
            self._update_beams()
            self._update_umatrix()
            self._update_ucell()
            self._update_ncells()
            self._update_ncells_def()
            self._update_rotXYZ()
            self._update_eta()  # mosaic spread
            self._update_dxtbx_detector()
            self._update_sausages()

            self._run_diffBragg_current()

            # CHECK FOR SIGNAL INTERRUPT HERE
            if self.break_signal is not None:
                signal.signal(self.break_signal, self._sig_hand.handle)
                self._MPI_check_for_break_signal()

            # TODO pre-extractions for all parameters
            self._pre_extract_deriv_arrays()
            self._extract_pixel_data()
            self._evaluate_averageI()
            self._evaluate_log_averageI_plus_sigma_readout()

            self._derivative_convenience_factors()

            if self.iterations % self.saveZ_freq == 0:
                MOD = self.Modelers[self._i_shot]
                self._spot_Zscores = []
                for i_fcell in MOD.unique_i_fcell:
                    for slc in MOD.i_fcell_slices[i_fcell]:
                        sigZ = self._Zscore[slc]
                        trus = MOD.all_trusted[slc]
                        sigZ = sigZ[trus].std()
                        self._spot_Zscores.append((i_fcell, sigZ))
                self._shot_Zscores.append(self._spot_Zscores)

            if save_model:
                MOD = self.Modelers[self._i_shot]
                P = MOD.all_pid
                F = MOD.all_fast
                S = MOD.all_slow
                M = self.model_Lambda
                B = MOD.all_background
                D = MOD.all_data
                C = self.model_bragg_spots
                Z = self._Zscore
                iF = MOD.all_fcell_global_idx
                iROI = MOD.roi_id
                trust = MOD.all_trusted

                model_info = {"p": P, "f": F, "s": S, "model": M,
                              "background": B, "data": D, "bragg": C,
                              "Zscore": Z, "i_fcell": iF, "trust": trust,
                              "i_roi": iROI}
                self._save_model(model_info)
            self._is_trusted = self.Modelers[self._i_shot].all_trusted
            self.target_functional += self._target_accumulate()
            self._spot_scale_derivatives()
            self._Fcell_derivatives()
        tshots = time.time()-tshots
        LOGGER.info("Time rank worked on shots=%.4f" % tshots)
        tmpi = time.time()
        COMM.barrier()
        LOGGER.info("MPI aggregation of func and grad")
        sys.exit()
        self._mpi_aggregation()
        print("DONEZO!")
        COMM.barrier()
        sys.exit()
        tmpi = time.time() - tmpi
        LOGGER.info("Time for MPIaggregation=%.4f" % tmpi)

        LOGGER.info("Aliases")
        self._f = self.target_functional
        self._g = self.g = self.grad
        self.d = self.curv
        LOGGER.info("curvature analysis")
        self._curvature_analysis()

        # reset ROI pixels TODO: is this necessary
        LOGGER.info("Zero pixels")
        self.D.raw_pixels_roi *= 0
        self.gnorm = -1

        tsave = time.time()
        LOGGER.info("DUMP param and Zscore data")
        self._save_Zscore_data()
        tsave = time.time()-tsave
        LOGGER.info("Time to dump param and Zscore data: %.4f" % tsave)

        self.iterations += 1
        self.f_vals.append(self.target_functional)

        if self.calc_curvatures and not self.use_curvatures:
            if self.num_positive_curvatures == self.use_curvatures_threshold:
                raise BreakToUseCurvatures

        LOGGER.info("DONE WITH FUNC GRAD")
        return self._f, self._g

    def _save_model(self, model_info):
        LOGGER.info("SAVING MODEL FOR SHOT %d" % self._i_shot)
        df = pandas.DataFrame(model_info)
        df["shot_id"] = self._i_shot
        outdir = self._save_model_dir
        outname = os.path.join(outdir, "rank%d_shot%d_ITER%d.pkl" % (COMM.rank, self._i_shot, self.iterations))
        df.to_pickle(outname)

    def _save_Zscore_data(self):
        if not self.iterations % self.saveZ_freq == 0:
            return
        outdir = os.path.join(self.Zdir, "rank%d_Zscore" % self.rank)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fname = os.path.join(outdir, "sigZ_iter%d_rank%d" % (self.iterations, self.rank))
        np.save(fname, self._shot_Zscores)

    def _Fcell_derivatives(self):
        if not self.refine_Fcell:
            return
        MOD = self.Modelers[self._i_shot]
        for i_fcell in MOD.unique_i_fcell:

            multi = self.hkl_frequency[i_fcell]
            if multi < self.min_multiplicity:
                continue

            xpos = self.fcell_xstart + i_fcell
            Famp = self._fcell_at_i_fcell[i_fcell]
            sig = 1
            for slc in MOD.i_fcell_slices[i_fcell]:
                self.fcell_dI_dtheta = self.fcell_deriv[slc]

                if self.log_fcells:
                    # case 2 rescaling
                    sig_times_fcell = sig*Famp
                    d = sig_times_fcell*self.fcell_dI_dtheta
                else:
                    # case 1 rescaling
                    d = sig*self.fcell_dI_dtheta

                gterm = self.common_grad_term[slc]
                g_accum = d*gterm
                trust = MOD.all_trusted[slc]
                self.grad[xpos] += (g_accum[trust].sum())*.5
                if self.calc_curvatures:
                    raise NotImplementedError("No curvature for Fcell refinement")

    def _spot_scale_derivatives(self, return_derivatives=False):
        if not self.refine_crystal_scale:
            return
        S = np.sqrt(self.scale_fac)
        dI_dtheta = (2./S)*self.model_bragg_spots
        d2I_dtheta2 = (2./S/S)*self.model_bragg_spots
        # second derivative is 0 with respect to scale factor
        sig = self.Modelers[self._i_shot].PAR.Scale.sigma
        d = dI_dtheta*sig
        d2 = d2I_dtheta2 *(sig**2)

        xpos = self.spot_scale_xpos[self._i_shot]
        self.grad[xpos] += self._grad_accumulate(d)
        if self.calc_curvatures:
            self.curv[xpos] += self._curv_accumulate(d, d2)

        if return_derivatives:
            return d, d2

    def _mpi_aggregation(self):
        # reduce the broadcast summed results:
        LOGGER.info("aggregate barrier")
        self._MPI_barrier()
        LOGGER.info("Functional")
        self.target_functional = self._MPI_reduce_broadcast(self.target_functional)
        LOGGER.info("gradients")
        self.grad = self._MPI_reduce_broadcast(self.grad)
        if self.calc_curvatures:
            self.curv = self._MPI_reduce_broadcast(self.curv)

    def _curvature_analysis(self):
        self.tot_neg_curv = 0
        self.neg_curv_shots = []
        if self.calc_curvatures:
            self.is_negative_curvature = self.curv.as_numpy_array() < 0
            self.tot_neg_curv = sum(self.is_negative_curvature)

        if self.calc_curvatures and not self.use_curvatures:
            if self.tot_neg_curv == 0:
                self.num_positive_curvatures += 1
                self.d = self.curv
                self._verify_diag()
            else:
                self.num_positive_curvatures = 0
                self.d = None

        if self.use_curvatures:
            assert self.tot_neg_curv == 0
            self.request_diag_once = False
            self.diag_mode = "always"  # TODO is this proper place to set ?
            self.d = self.curv
            self._verify_diag()
        else:
            self.d = None

    def _MPI_save_state_of_refiner(self):
        if self.I_AM_ROOT and self.output_dir is not None and self.refine_Fcell:
            outf = os.path.join(self.output_dir, "_fcell_trial%d_iter%d" % (self.trial_id, self.iterations))
            np.savez(outf, fvals=self._fcell_at_i_fcell)

    def _target_accumulate(self):
        fterm = self.log2pi + self.log_v + self.u*self.u*self.one_over_v
        if self._is_trusted is not None:
            fterm = fterm[self._is_trusted]
        fterm = 0.5*fterm.sum()
        return fterm

    def _grad_accumulate(self, d):
        gterm = d * self.one_over_v * self.one_minus_2u_minus_u_squared_over_v
        if self._is_trusted is not None:
            gterm = gterm[self._is_trusted]
        gterm = 0.5*gterm.sum()
        return gterm

    def _curv_accumulate(self, d, d2):
        cterm = self.one_over_v * (d2*self.one_minus_2u_minus_u_squared_over_v -
                                   d*d*(self.one_over_v_times_one_minus_2u_minus_u_squared_over_v -
                                        (2 + 2*self.u_times_one_over_v + self.u_u_one_over_v*self.one_over_v)))
        if self._is_trusted is not None:
            cterm = cterm[self._is_trusted]
        cterm = .5 * (cterm.sum())
        return cterm

    def _derivative_convenience_factors(self):
        Mod = self.Modelers[self._i_shot]
        self.Imeas = Mod.all_data
        self.u = self.Imeas - self.model_Lambda
        self.one_over_v = 1. / (self.model_Lambda + Mod.sigma_rdout ** 2)
        self.one_minus_2u_minus_u_squared_over_v = 1 - 2 * self.u - self.u * self.u * self.one_over_v
        if self.calc_curvatures:
            self.u_times_one_over_v = self.u*self.one_over_v
            self.u_u_one_over_v = self.u*self.u_times_one_over_v
            self.one_over_v_times_one_minus_2u_minus_u_squared_over_v = self.one_over_v*self.one_minus_2u_minus_u_squared_over_v
        self.common_grad_term = self.one_over_v * self.one_minus_2u_minus_u_squared_over_v
        self._Zscore = self.u*np.sqrt(self.one_over_v)

    def _evaluate_log_averageI_plus_sigma_readout(self):
        Mod = self.Modelers[self._i_shot]
        v = self.model_Lambda + Mod.sigma_rdout ** 2
        v_is_neg = (v <= 0).ravel()
        if any(v_is_neg):
            LOGGER.info("\n<><><><><><><><>\n\tWARNING: NEGATIVE INTENSITY IN MODEL!!!!!!!!!\n<><><><><><><><><>\n")
        self.log_v = np.log(v)
        self.log_v[v <= 0] = 0  # but will I ever negative_model ?

    def curvatures(self):
        return self.curv

    def _MPI_sync_hkl_freq(self):
        if self.refine_Fcell:
            if self.rank != 0:
                self.hkl_frequency = None
            self.hkl_frequency = COMM.bcast(self.hkl_frequency)

    def _MPI_sync_fcell_parameters(self):
        if not self.I_AM_ROOT:
            self.sigma_for_res_id = None
            self.res_group_id_from_fcell_index = None
            self.resolution_ids_from_i_fcell = self.fcell_sigmas_from_i_fcell = self.fcell_init_from_i_fcell = None

        if self.rescale_params:
            if self.refine_Fcell:
                self.fcell_sigmas_from_i_fcell = COMM.bcast(self.fcell_sigmas_from_i_fcell)
                self.fcell_init_from_i_fcell = COMM.bcast(self.fcell_init_from_i_fcell)

    def _MPI_reduce_broadcast(self, var):
        var = COMM.reduce(var, MPI.SUM, root=0)
        var = COMM.bcast(var, root=0)
        return var

    def _MPI_barrier(self):
        COMM.barrier()

LOGGER = logging.getLogger("main")
class UC:
    def __init__(self):
        self.a = 78
        self.b = 78
        self.c = 264
        self.al = 90
        self.be = 90
        self.ga = 120

def global_refiner_from_parameters(params):
    launcher = RefineLauncher(params)
    # TODO read on each rank, or read and broadcast ?
    LOGGER.info("EVENT: read input pickle")
    pandas_table = None # pandas.read_pickle(params.pandas_table)
    LOGGER.info("EVENT: BEGIN prep dataframe")
    LOGGER.info("EVENT: DONE prep dataframe")
    return launcher.launch_refiner(pandas_table)

class RefineLauncher:

    def __init__(self, params):
        self.params = self.check_parameter_integrity(params)
        self.n_shots_on_rank = None
        self.df = None
        self.Modelers = {}
        self.Hi = {}
        self.Hi_asu = {}
        self.symbol = "P6522"
        self.DEVICE_ID = 0

    @property
    def NPIX_TO_ALLOC(self):
        return self._NPIX_TO_ALLOC

    @NPIX_TO_ALLOC.setter
    def NPIX_TO_ALLOC(self, val):
        assert val> 0 or val == -1
        self._NPIX_TO_ALLOC = int(val)

    @staticmethod
    def check_parameter_integrity(params):
        if params.refiner.max_calls is None or len(params.refiner.max_calls) == 0:
            raise ValueError("Cannot refine because params.refiner.max_calls is empty")

        if os.environ.get("DIFFBRAGG_CUDA") is not None:
            params.refiner.use_cuda = True

        return params

    @property
    def num_shots_on_rank(self):
        return len(self.Modelers)

    def launch_refiner(self, pandas_table):

        COMM.barrier()
        num_exp = 7500 #len(pandas_table)

        if COMM.size > num_exp:
            raise ValueError("Requested %d MPI ranks to process %d shots. Reduce number of ranks to %d"
                             % (COMM.size, num_exp, num_exp))

        self.verbose = False
        if COMM.rank == 0:
            self.verbose = self.params.refiner.verbose > 0
            if self.params.refiner.gather_dir is not None and not os.path.exists(self.params.refiner.gather_dir):
              os.makedirs(self.params.refiner.gather_dir)
              LOGGER.info("MADE GATHER DIR %s" % self.params.refiner.gather_dir)
        COMM.barrier()
        shot_idx = 0  # each rank keeps index of the shots local to it
        rank_panel_groups_refined = set()
        exper_names = ["dummie%d" % i_exp for i_exp in range(num_exp)] #*pandas_table.exp_name
        assert len(exper_names) == len(set(exper_names))
        LOGGER.info("EVENT: begin loading inputs")
        for i_exp, exper_name in enumerate(exper_names):
            if i_exp % COMM.size != COMM.rank:
                continue
            LOGGER.info("EVENT: BEGIN loading experiment list")

            if shot_idx == 0:  # each rank initializes a simulator only once
                if self.params.simulator.init_scale != 1:
                    print("WARNING: For stage_two , it is assumed that total scale is stored in the pandas dataframe")
                    print("WARNING: resetting params.simulator.init_scale to 1!")
                    self.params.simulator.init_scale = 1

            LOGGER.info("EVENT: LOADING ROI DATA")
            shot_modeler = DataModeler()
            time.sleep(np.random.random()*0.05)

            Nh = np.random.randint(10,250)
            shot_modeler.Hi = [tuple(hkl) for hkl in np.random.randint(-100,100,(Nh,3)).astype(np.int32)]
            shot_modeler.Hi_asu = [tuple(hkl) for hkl in np.random.randint(-100,100,(Nh,3)).astype(np.int32)]
            self.Hi[shot_idx] = shot_modeler.Hi
            self.Hi_asu[shot_idx] =shot_modeler.Hi_asu

            shot_modeler.originZ_init = 0
            shot_modeler.exper_name = exper_name

            shot_idx += 1
            if COMM.rank == 0:
                print("Finished loading image %d / %d" % (i_exp+1, len(exper_names)), flush=True)

            self.Modelers[i_exp] = shot_modeler

        LOGGER.info("DONE LOADING DATA; ENTER BARRIER")
        COMM.barrier()
        LOGGER.info("DONE LOADING DATA; EXIT BARRIER")
        #if not self.shot_roi_darkRMS:
        self.shot_roi_darkRMS = None

        # TODO warn that per_spot_scale refinement not intended for ensemble mode
        all_refined_groups = COMM.gather(rank_panel_groups_refined)
        panel_groups_refined = None
        if COMM.rank == 0:
            panel_groups_refined = set()
            for set_of_panels in all_refined_groups:
                panel_groups_refined = panel_groups_refined.union(set_of_panels)
        self.panel_groups_refined = list(COMM.bcast(panel_groups_refined))

        LOGGER.info("EVENT: Gathering global HKL information")
        self._gather_Hi_information()
        LOGGER.info("EVENT: FINISHED gather global HKL information")
        if self.params.roi.cache_dir_only:
            print("Done creating cache directory and cache_dir_only=True, so goodbye.")
            sys.exit()

        # in case of GPU
        LOGGER.info("BEGIN DETERMINE MAX PIX")
        self.NPIX_TO_ALLOC = self._determine_per_rank_max_num_pix()
        # TODO in case of randomize devices, shouldnt this be total max across all ranks?
        n = COMM.gather(self.NPIX_TO_ALLOC)
        if COMM.rank == 0:
            n = max(n)
        self.NPIX_TO_ALLOC = COMM.bcast(n)
        LOGGER.info("DONE DETERMINE MAX PIX")

        self.DEVICE_ID = COMM.rank % self.params.refiner.num_devices

        self._mem_usage()

        LOGGER.info("EVENT: launch refiner")
        self._launch()

        return self.RUC

    def _mem_usage(self):
        memMB = get_memory_usage()
        import socket
        host = socket.gethostname()
        print("reporting memory usage: %f GB on node %s" % (memMB / 1e3, host), flush=True)

    def _determine_per_rank_max_num_pix(self):
        max_npix = 0
        for i_shot in self.Modelers:
            modeler = self.Modelers[i_shot]
            npix = np.random.randint(1000,10000) # #np.sum((x2-x1)*(y2-y1))
            max_npix = max(npix, max_npix)
        return max_npix

    def _gather_Hi_information(self):
        nshots_on_this_rank = len(self.Hi)
        # aggregate all miller indices
        self.Hi_all_ranks, self.Hi_asu_all_ranks = [], []
        # TODO assert list types are stored in Hi and Hi_asu
        for i_shot in range(nshots_on_this_rank):
            self.Hi_all_ranks += self.Hi[i_shot]
            self.Hi_asu_all_ranks += self.Hi_asu[i_shot]
        self.Hi_all_ranks = COMM.reduce(self.Hi_all_ranks)
        self.Hi_all_ranks = COMM.bcast(self.Hi_all_ranks)

        self.Hi_asu_all_ranks = COMM.reduce(self.Hi_asu_all_ranks)
        self.Hi_asu_all_ranks = COMM.bcast(self.Hi_asu_all_ranks)

        marr_unique_h = self._get_unique_Hi()
        self.ma = marr_unique_h

        # this will map the measured miller indices to their index in the LBFGS parameter array self.x
        self.idx_from_asu = {h: i for i, h in enumerate(set(self.Hi_asu_all_ranks))}
        # we will need the inverse map during refinement to update the miller array in diffBragg, so we cache it here
        self.asu_from_idx = {i: h for i, h in enumerate(set(self.Hi_asu_all_ranks))}

        self.num_hkl_global = len(self.idx_from_asu)

    def _get_unique_Hi(self):
        COMM.barrier()
        if COMM.rank == 0:
            from cctbx.crystal import symmetry
            from cctbx import miller
            from cctbx.array_family import flex as cctbx_flex

            ii = list(self.Modelers.keys())[0]
            uc = UC()
            params = uc.a, uc.b, uc.c, uc.al , uc.be , uc.ga
            #if self.params.refiner.force_unit_cell is not None:
            #    params = self.params.refiner.force_unit_cell
            symm = symmetry(unit_cell=params, space_group_symbol="P6522")
            hi_asu_flex = cctbx_flex.miller_index(self.Hi_asu_all_ranks)
            mset = miller.set(symm, hi_asu_flex, anomalous_flag=True)
            marr = miller.array(mset)
            binner = marr.setup_binner(d_max=self.params.refiner.stage_two.d_max, d_min=self.params.refiner.stage_two.d_min,
                                       n_bins=self.params.refiner.stage_two.n_bin)
            from collections import Counter
            print("Average multiplicities:")
            print("<><><><><><><><><><><><>")
            for i_bin in range(self.params.refiner.stage_two.n_bin - 1):
                dmax, dmin = binner.bin_d_range(i_bin + 1)
                F_in_bin = marr.resolution_filter(d_max=dmax, d_min=dmin)
                multi_in_bin = np.array(list(Counter(F_in_bin.indices()).values()))
                print("%2.5g-%2.5g : Multiplicity=%.4f" % (dmax, dmin, multi_in_bin.mean()))
                for ii in range(1, 100, 8):
                    print("\t %d refls with multi %d" % (sum(multi_in_bin == ii), ii))

            print("Overall completeness\n<><><><><><><><>")
            symm = symmetry(unit_cell=params, space_group_symbol="P6522")
            hi_flex_unique = cctbx_flex.miller_index(list(set(self.Hi_asu_all_ranks)))
            mset = miller.set(symm, hi_flex_unique, anomalous_flag=True)
            self.binner = mset.setup_binner(d_min=self.params.refiner.stage_two.d_min,
                                            d_max=self.params.refiner.stage_two.d_max,
                                            n_bins=self.params.refiner.stage_two.n_bin)
            mset.completeness(use_binning=True).show()
            marr_unique_h = miller.array(mset)
            print("Rank %d: total miller vars=%d" % (COMM.rank, len(set(self.Hi_asu_all_ranks))))
        else:
            marr_unique_h = None

        marr_unique_h = COMM.bcast(marr_unique_h)
        return marr_unique_h

    def _launch(self):
        """
        Usually this method should be modified when new features are added to refinement
        """
        # TODO return None or refiner instance
        LOGGER.info("begin _launch")
        x_init = None
        nmacro = self.params.refiner.num_macro_cycles
        n_trials = len(self.params.refiner.max_calls)
        for i_trial in range(n_trials*nmacro):

            self.RUC = StageTwoRefiner2(self.Modelers, self.symbol, self.params)

            if self.will_refine(self.params.refiner.refine_spot_scale):
                self.RUC.refine_crystal_scale = (self.params.refiner.refine_spot_scale*nmacro)[i_trial]

            if self.will_refine(self.params.refiner.refine_Fcell):
                self.RUC.refine_Fcell = (self.params.refiner.refine_Fcell*nmacro)[i_trial]

            self.RUC.panel_group_from_id = None #self.panel_group_from_id
            self.RUC.panel_reference_from_id = None #self.panel_reference_from_id
            self.RUC.panel_groups_being_refined = None #self.panel_groups_refined

            # TODO verify not refining Fcell in case of local refiner
            self.RUC.max_calls = (self.params.refiner.max_calls*nmacro)[i_trial]
            self.RUC.x_init = x_init
            self.RUC.ignore_line_search_failed_step_at_lower_bound = True  # TODO: why was this necessary?

            # plot things
            self.RUC.trial_id = i_trial

            self.RUC.log_fcells = True
            self.RUC.request_diag_once = False
            self.RUC.trad_conv = True
            self.RUC.idx_from_asu = self.idx_from_asu
            self.RUC.asu_from_idx = self.asu_from_idx

            self.RUC.ma = self.ma
            self.RUC.restart_file = self.params.refiner.io.restart_file

            LOGGER.info("_launch run setup")
            self.RUC._setup()
            COMM.barrier()
            sys.exit()
            LOGGER.info("_launch done run setup")

            self.RUC.num_positive_curvatures = 0
            self.RUC.use_curvatures = self.params.refiner.start_with_curvatures
            self.RUC.hit_break_to_use_curvatures = False

            LOGGER.info("_launcher runno setup")
            self.RUC.run(setup=False)
            LOGGER.info("_launcher done runno setup")
            if self.RUC.hit_break_to_use_curvatures:
                self.RUC.fix_params_with_negative_curvature = False
                self.RUC.num_positive_curvatures = 0
                self.RUC.use_curvatures = True
                self.RUC.run(setup=False)

            if self.RUC.hit_break_signal:
                if self.params.profile:
                    self.RUC.S.D.show_timings(self.RUC.rank)
                self.RUC._MPI_barrier()
                break

            x_init = self.RUC.x

            if self.params.profile:
                self.RUC.S.D.show_timings(self.RUC.rank)
            if os.environ.get("DIFFBRAGG_USE_CUDA") is not None:
                self.RUC.S.D.gpu_free()

    def will_refine(self, param):
        return param is not None and any(param)

if __name__ == "__main__":
    script_phil = """
    pandas_table = None
      .type = str
      .help = path to an input pandas table (usually output by simtbx.diffBragg.predictions)
    prep_time = 60
      .type = float
      .help = Time spent optimizing order of input dataframe to better divide shots across ranks
      .help = Unit is seconds, 1-2 minutes of prep might save a lot of time during refinement!
    """

    hopper_phil = """
    use_float32 = False
      .type = bool
      .help = store pixel data and background models in 32bit arrays
      .expert_level=10
    test_gathered_file = False
      .type = bool
      .help = run a quick test to ensure the gathered data file preserves information
      .expert_level=10
    load_data_from_refls = False
      .type = bool
      .help = load image data, background etc from reflection tables
      .expert_level=10
    gathered_output_file = None
      .type = str
      .help = optional file for storing a new hopper input file which points to the gathered data dumps
      .expert_level=10
    only_dump_gathers = False
      .type = bool
      .help = only reads in image data, fits background planes, and dumps
      .help = results to disk, writes a new exper refl file at the end
      .expert_level=10
    gathers_dir = None
      .type = str
      .help = folder where gathered data reflection tables
      .help = will be writen (if dump_gathers=True)
      .expert_level=10
    dump_gathers = False
      .type = bool
      .help = optionally dump the loaded experimental data to reflection tables
      .help = for portability
      .expert_level=10
    spectrum_from_imageset = False
      .type = bool
      .help = if True, load the spectrum from the imageset in the experiment, then probably downsample it
      .expert_level=0
    isotropic {
      diffuse_gamma = False
        .type = bool
        .help = refine a single diffuse gamma parameter as opposed to 3
      diffuse_sigma = False
        .type = bool
        .help = refine a single diffuse gamma parameter as opposed to 3
    }
    downsamp_spec {
      skip = False
        .type = bool
        .help = if reading spectra from imageset, optionally skip the downsample portion
        .help = Note, if skip=True, then total flux will be determined by whats in the imageset spectrum (sum of the weights)
        .expert_level=10
      filt_freq = 0.07
        .type = float
        .help = low pass filter frequency in units of inverse spectrometer pixels (??)
        .expert_level=10
      filt_order = 3
        .type = int
        .help = order for bandpass butter filter
        .expert_level=10
      tail = 50
        .type = int
        .help = endpoints of the spectrum that are used in background estimation
        .expert_level=10
      delta_en = 0.5
        .type = float
        .help = final resolution of downsampled spectrum in eV
        .expert_level=0
    }
    apply_best_crystal_model = False
      .type = bool
      .help = depending on what experiments in the exper refl file, one may want
      .help = to apply the optimal crystal transformations (this parameter only matters
      .help = if params.best_pickle is not None)
      .expert_level=10
    filter_unpredicted_refls_in_output = True
      .type = bool
      .help = filter reflections in the output refl table for which there was no model bragg peak
      .help = after stage 1 termination
      .expert_level=10
    tag = stage1
      .type = str
      .help = output name tag
      .expert_level=0
    ignore_existing = False
      .type = bool
      .help = experimental, ignore expts that already have optimized models in the output dir
      .expert_level=0
    global_method = *basinhopping annealing
      .type = choice
      .help = the method of global optimization to use
      .expert_level=10
    nelder_mead_maxfev = 60
      .type = int
      .help = multiplied by total number of modeled pixels to get max number of iterations
      .expert_level=10
    nelder_mead_fatol = 0.0001
      .type = float
      .help = nelder mead functional error tolerance
    niter_per_J = 1
      .type = int
      .help = if using gradient descent, compute gradients
      .help = every niter_per_J iterations .
      .expert_level=10
    rescale_params = True
      .type = bool
      .help = use rescaled range parameters
      .expert_level=10
    best_pickle = None
      .type = str
      .help = path to a pandas pickle containing the best models for the experiments
      .expert_level=0
    betas
      .help = variances for the restraint targets
      .expert_level=0
    {
      Nvol = 1e8
        .type = float
        .help = tightness of the Nabc volume contraint
      detz_shift = 1e8
        .type = float
        .help = restraint variance for detector shift target
      ucell = [1e8,1e8,1e8,1e8,1e8,1e8]
        .type = floats
        .help = beta values for unit cell constants
      RotXYZ = 1e8
        .type = float
        .help = restraint factor for the rotXYZ restraint
      Nabc = [1e8,1e8,1e8]
        .type = floats(size=3)
        .help = restraint factor for the ncells abc
      Ndef = [1e8,1e8,1e8]
        .type = floats(size=3)
        .help = restraint factor for the ncells def
      diffuse_sigma = 1e8,1e8,1e8
        .type = floats(size=3)
        .help = restraint factor for diffuse sigma
      diffuse_gamma = 1e8,1e8,1e8
        .type = floats(size=3)
        .help = restraint factor for diffuse gamma
      G = 1e8
        .type = float
        .help = restraint factor for the scale G
      B = 1e8
        .type = float
        .help = restraint factor for Bfactor
    }
    dual
      .help = configuration parameters for dual annealing
      .expert_level=10
    {
      initial_temp = 5230
        .type = float
        .help = init temp for dual annealing
      no_local_search = False
        .type = bool
        .help = whether to try local search procedure with dual annealing
        .help = if False, then falls back on classical simulated annealing
      visit = 2.62
        .type = float
        .help = dual_annealing visit param, see scipy optimize docs
      accept = -5
        .type = float
        .help = dual_annealing accept param, see scipy optimize docs
    }
    centers
      .help = restraint targets
      .expert_level=0
    {
      Nvol = None
        .type = float
        .help = if provided, constrain the product Na*Nb*Nc to this value
      detz_shift = 0
        .type = float
        .help = restraint target for detector shift along z-direction
      ucell = [63.66, 28.87, 35.86, 1.8425]
        .type = floats
        .help = centers for unit cell constants
      RotXYZ = [0,0,0]
        .type = floats(size=3)
        .help = restraint target for Umat rotations
      Nabc = [100,100,100]
        .type = floats(size=3)
        .help = restraint target for Nabc
      Ndef = [0,0,0]
        .type = floats(size=3)
        .help = restraint target for Ndef
      diffuse_sigma = [1,1,1]
        .type = floats(size=3)
        .help = restraint target for diffuse sigma
      diffuse_gamma = [1,1,1]
        .type = floats(size=3)
        .help = restraint target for diffuse gamma
      G = 100
        .type = float
        .help = restraint target for scale G
      B = 0
        .type = float
        .help = restraint target for Bfactor
    }
    skip = None
      .type = int
      .help = skip this many exp
      .expert_level=0
    hess = None
      .type = str
      .help = scipy minimize hessian argument, 2-point, 3-point, cs, or None
      .expert_level=10
    stepsize = 0.5
      .type = float
      .help = basinhopping stepsize
      .expert_level=10
    temp = 1
      .type = float
      .help = temperature for basin hopping algo
      .expert_level=10
    niter = 0
      .type = int
      .help = number of basin hopping iterations (0 just does a gradient descent and stops at the first minima encountered)
      .expert_level=0
    exp_ref_spec_file = None
      .type = str
      .help = path to 3 col txt file containing file names for exper, refl, spectrum (.lam)
      .expert_level=0
    method = None
      .type = str
      .help = minimizer method, usually this is L-BFGS-B (gradients) or Nelder-Mead (simplex)
      .help = other methods are experimental (see details in hopper_utils.py)
      .expert_level=0
    opt_det = None
      .type = str
      .help = path to experiment with optimized detector model
      .expert_level=0
    opt_beam = None
      .type = str
      .help = path to experiment with optimized beam model
      .expert_level=0
    number_of_xtals = 1
      .type = int
      .help = number of crystal domains to model per shot
      .expert_level=10
    sanity_test_input = True
      .type = bool
      .help = sanity test input
      .expert_level=10
    outdir = None
      .type = str
      .help = output folder
      .expert_level=0
    max_process = -1
      .type = int
      .help = max exp to process
      .expert_level=0
    sigmas
      .help = sensitivity of target to parameter (experimental)
      .expert_level=10
    {
      detz_shift = 1
        .type = float
        .help = sensitivity shift for the overall detector shift along z-direction
      Nabc = [1,1,1]
        .type = floats(size=3)
        .help = sensitivity for Nabc
      Ndef = [1,1,1]
        .type = floats(size=3)
        .help = sensitivity for Ndef
      diffuse_sigma = [1,1,1]
        .type = floats(size=3)
        .help = sensitivity for diffuse sigma
      diffuse_gamma = [1,1,1]
        .type = floats(size=3)
        .help = sensitivity for diffuse gamma
      RotXYZ = [1,1,1]
        .type = floats(size=3)
        .help = sensitivity for RotXYZ
      G = 1
        .type = float
        .help = sensitivity for scale factor
      B = 1
        .type = float
        .help = sensitivity for Bfactor
      ucell = [1,1,1,1,1,1]
        .type = floats
        .help = sensitivity for unit cell params
      Fhkl = 1
        .type = float
        .help = sensitivity for structure factors
    }
    init
      .help = initial value of model parameter (will be overrided if best pickle is provided)
      .expert_level=0
    {
      detz_shift = 0
        .type = float
        .help = initial value for the detector position overall shift along z-direction in millimeters
      Nabc = [100,100,100]
        .type = floats(size=3)
        .help = init for Nabc
      Ndef = [0,0,0]
        .type = floats(size=3)
        .help = init for Ndef
      diffuse_sigma = [.01,.01,.01]
        .type = floats(size=3)
        .help = init diffuse sigma
      diffuse_gamma = [1,1,1]
        .type = floats(size=3)
        .help = init for diffuse gamma
      RotXYZ = [0,0,0]
        .type = floats(size=3)
        .help = init for RotXYZ
      G = 1
        .type = float
        .help = init for scale factor
      B = 0
        .type = float
        .help = init for B factor
    }
    mins
      .help = min value allowed for parameter
      .expert_level = 0
    {
      detz_shift = -10
        .type = float
        .help = min value for detector z-shift in millimeters
      Nabc = [3,3,3]
        .type = floats(size=3)
        .help = min for Nabc
      Ndef = [-200,-200,-200]
        .type = floats(size=3)
        .help = min for Ndef
      diffuse_sigma = [0,0,0]
        .type = floats(size=3)
        .help = min diffuse sigma
      diffuse_gamma = [0,0,0]
        .type = floats(size=3)
        .help = min for diffuse gamma
      RotXYZ = [-1,-1,-1]
        .type = floats(size=3)
        .help = min for rotXYZ in degrees
      G = 0
        .type = float
        .help = min for scale G
      B = 0
        .type = float
        .help = min for Bfactor
      Fhkl = 0
        .type = float
        .help = min for structure factors
    }
    maxs
      .help = max value allowed for parameter
      .expert_level = 0
    {
      detz_shift = 10
        .type = float
        .help = max value for detector z-shift in millimeters
      eta = 0.1
        .type = float
        .help = maximum mosaic spread in degrees
      Nabc = [300,300,300]
        .type = floats(size=3)
        .help = max for Nabc
      Ndef = [200,200,200]
        .type = floats(size=3)
        .help = max for Ndef
      diffuse_sigma = [20,20,20]
        .type = floats(size=3)
        .help = max diffuse sigma
      diffuse_gamma = [1000,1000,1000]
        .type = floats(size=3)
        .help = max for diffuse gamma
      RotXYZ = [1,1,1]
        .type = floats(size=3)
        .help = max for rotXYZ in degrees
      G = 1e12
        .type = float
        .help = max for scale G
      B = 1e3
        .type = float
        .help = max for Bfactor
      Fhkl = 1e6
        .type = float
        .help = max for structure factors
    }
    fix
      .help = flags for fixing parameters during refinement
      .expert_level = 0
    {
      G = False
        .type = bool
        .help = fix the Bragg spot scale during refinement
      B = True
        .type = bool
        .help = fix the Bfactor during refinement
      RotXYZ = False
        .type = bool
        .help = fix the misorientation matrix during refinement
      Nabc = False
        .type = bool
        .help = fix the diagonal mosaic domain size parameters during refinement
      Ndef = False
        .type = bool
        .help = fix the diagonal mosaic domain size parameters during refinement
      diffuse_sigma = True
        .type = bool
        .help = fix diffuse sigma
      diffuse_gamma = True
        .type = bool
        .help = fix diffuse gamma
      ucell = False
        .type = bool
        .help = fix the unit cell during refinement
      detz_shift = False
        .type = bool
        .help = fix the detector distance shift during refinement
    }
    relative_tilt = False
      .type = bool
      .help = fit tilt coef relative to roi corner
      .expert_level = 10
    num_mosaic_blocks = 1
      .type = int
      .help = number of mosaic blocks making up mosaic spread dist (not implemented)
      .expert_level = 10
    ucell_edge_perc = 10
      .type = float
      .help = precentage for allowing ucell to fluctuate during refinement
      .expert_level = 10
    ucell_ang_abs = 5
      .type = float
      .help = absolute angle deviation in degrees for unit cell angles to vary during refinement
      .expert_level = 10
    no_Nabc_scale = False
      .type = bool
      .help = toggle Nabc scaling of the intensity
      .expert_level = 10
    use_diffuse_models = False
      .type = bool
      .help = if True, let the values of init.diffuse_sigma and init.diffuse_gamma
      .help = be used to define the diffuse scattering. Set e.g. fix.diffuse_sigma=True in order to refine them
      .expert_level = 10
    sigma_frac = None
      .type = float
      .help = sigma for Fhkl restraints will be some fraction of the starting value
      .expert_level = 10
    sanity_test_hkl_variation = False
      .type = bool
      .help = measure the variation of each HKL within the shoebox
      .expert_level = 10
    sanity_test_models = False
      .type = bool
      .help = make sure best models from stage 1 are reproduced at the start
      .expert_level = 10
    sanity_test_amplitudes = False
      .type = bool
      .help = if True, then quickly run a sanity check ensuring that all h,k,l are predicted
      .help = and/or in the starting miller array
      .expert_level = 10
    x_write_freq = 25
      .type = int
      .help = save x arrays every x_write_freq iterations
      .expert_level = 10
    percentile_cut = None
      .type = float
      .help = percentile below which pixels are masked
      .expert_level = 10
    space_group = None
      .type = str
      .help = space group to refine structure factors in
      .expert_level = 0
    first_n = None
      .type = int
      .help = refine the first n shots only
      .expert_level = 0
    maxiter = 15000
      .type = int
      .help = stop refiner after this many iters
      .expert_level = 10
    ftol = 1e-10
      .type = float
      .help = ftol convergence threshold for scipys L-BFGS-B
      .expert_level = 10
    disp = False
      .type = bool
      .help = scipy minimize convergence printouts
      .expert_level = 10
    use_restraints = True
      .type = bool
      .help = disable the parameter restraints
      .expert_level = 0
    min_multi = 2
      .type = int
      .help = minimum ASU multiplicity, obs that fall below this threshold
      .help = are removed from analysis
      .expert_level = 10
    min_spot = 5
      .type = int
      .help = minimum spots on a shot in order to optimize that shot
      .expert_level = 10
    logging
      .help = controls the logging module for hopper and stage_two
      .expert_level = 10
    {
      disable = True 
        .type = bool
        .help = turn off logging
      logfiles_level = low *normal high
        .type = choice
        .help = level of the main log when writing logfiles
      logfiles = False
        .type = bool
        .help = write log files in the outputdir
      rank0_level = low *normal high
        .type = choice
        .help = console log level for rank 0, ignored if logfiles=True
      other_ranks_level = *low normal high
        .type = choice
        .help = console log level for all ranks > 0, ignored if logfiles=True
      overwrite = True
        .type = bool
        .help = overwrite the existing logfiles
      logname = None
        .type = str
        .help = if logfiles=True, then write the log to this file, stored in the folder specified by outdir
        .help = if None, then defaults to main_stage1.log for hopper, main_pred.log for prediction, main_stage2.log for stage_two
    }
    profile = False
      .type = bool
      .help = profile the workhorse functions
      .expert_level = 0
    profile_name = None
      .type = str
      .help = name of the output file that stores the line-by-line profile (written to folder specified by outdir)
      .help = if None, defaults to prof_stage1.log, prof_pred.log, prof_stage2.log for hopper, prediction, stage_two respectively
      .expert_level = 10
    """

    simulator_phil = """
    simulator {
      oversample = 0
        .type = int
        .help = pixel oversample rate (0 means auto-select)
      device_id = 0
        .type = int
        .help = device id for GPU simulation
      init_scale = 1
        .type = float
        .help = initial scale factor for this crystal simulation
      total_flux = 1e12
        .type = float
        .help = total photon flux for all energies
      crystal {
        ncells_abc = (10,10,10)
          .type = floats(size=3)
          .help = number of unit cells along each crystal axis making up a mosaic domain
        ncells_def = (0,0,0)
          .type = floats(size=3)
          .help = off-diagonal components for mosaic domain model (experimental)
        has_isotropic_ncells = False
          .type = bool
          .help = if True, ncells_abc are constrained to be the same values during refinement
        mosaicity = 0
          .type = float
          .help = mosaic spread in degrees
        anisotropic_mosaicity = None
          .type = floats
          .help = mosaic spread 3-tuple or 6-tuple specifying anisotropic mosaicity
        num_mosaicity_samples = 1
          .type = int
          .help = the number of mosaic domains to use when simulating mosaic spread
        mos_angles_per_axis = 10
          .type = int
          .help = if doing a uniform mosaicity sampling, use this many angles per rotation axis
        num_mos_axes = 10
          .type = int
          .help = number of sampled rot axes if doing a uniform mosaicity sampling
        mosaicity_method = 2
          .type = int
          .help = 1 or 2. 1 is random sampling, 2 is even sampling
        rotXYZ_ucell = None
          .type = floats(size=9)
          .help = three missetting angles (about X,Y,Z axes), followed by
          .help = unit cell parameters. The crystal will be rotated according to
          .help = the matrix RotX*RotY*RotZ, and then the unit cell will be updated
      }
      structure_factors {
        mtz_name = None
          .type = str
          .help = path to an MTZ file
        mtz_column = None
          .type = str
          .help = column in an MTZ file
        dmin = 1.5
          .type = float
          .help = minimum resolution for structure factor array
        dmax = 30
          .type = float
          .help = maximum resolution for structure factor array
        default_F = 0
          .type = float
          .help = default value for structure factor amps
      }
      spectrum {
        filename = None
          .type = str
          .help = a .lam file (precognition) for inputting wavelength spectra
        stride = 1
          .type = int
          .help = stride of the spectrum (e.g. set to 10 to keep every 10th value in the spectrum file data)
        filename_list = None
          .type = str
          .help = path to a file containing 1 .lam filename per line
      }
      beam {
        size_mm = 1
          .type = float
          .help = diameter of the beam in mm
      }
      detector {
        force_zero_thickness = False
          .type = bool
          .help = if True, then set sensor thickness to 0
      }
    }
    """

    refiner_phil = """
    refiner {
      load_data_from_refl = False
        .type = bool
      test_gathered_file = False
        .type = bool
      gather_dir = None
        .type = str
        .help = optional dir for stashing loaded input data in refl files (mainly for tests/portability)
      break_signal = None
        .type = int
        .help = intended to be used to break out of a long refinement job prior to a timeout on a super computer
        .help = On summit, set this to 12 (SIGUSR2), at least thats what it was last I checked (July 2021)
      debug_pixel_panelfastslow = None
        .type = ints(size=3)
        .help = 3-tuple of panel ID, fast coord, slow coord. If set, show the state of diffBragg
        .help = for this pixel once refinement has finished
      res_ranges = None
        .type = str
        .help = resolution-defining strings, where each string is
        .help = is comma-separated substrings, formatted according to "%f-%f,%f-%f" where the first float
        .help = in each substr specifies the high-resolution for the refinement trial, and the second float
        .help = specifies the low-resolution for the refinement trial. Should be same length as max_calls
      mask = None
        .type = str
        .help = path to a dials mask flagging the trusted pixels
      force_symbol = None
        .type = str
        .help = a space group lookup symbol used to map input miller indices to ASU
      force_unit_cell = None
        .type = ints(size=6)
        .help = a unit cell tuple to use
      randomize_devices = True
        .type = bool
        .help = randomly select a device Id
      num_devices = 1
        .type = int
        .help = number of cuda devices on current node
      refine_Fcell = None
        .type = ints(size_min=1)
        .help = whether to refine the structure factor amplitudes
      refine_spot_scale = None
        .type = ints(size_min=1)
        .help = whether to refine the crystal scale factor
      max_calls = [100]
        .type = ints(size_min=1)
        .help = maximum number of calls for the refinement trial
      panel_group_file = None
        .type = str
        .help = a text file with 2 columns, the first column is the panel_id and the second
        .help = column is the panel_group_id. Panels geometries in the same group are refined together
      update_oversample_during_refinement = False
        .type = bool
        .help = whether to update the oversample parameter as ncells changes
      sigma_r = 3
        .type = float
        .help = standard deviation of the dark signal fluctuation
      adu_per_photon = 1
        .type = float
        .help = how many ADUs (detector units) equal 1 photon
      use_curvatures_threshold = 10
        .type = int
        .help = how many consecutiv positive curvature results before switching to curvature mode
      curvatures = False
        .type = bool
        .help = whether to try using curvatures
      start_with_curvatures = False
        .type = bool
        .help = whether to try using curvatures in the first iteration
      tradeps = 1e-2
        .type = float
        .help = LBFGS termination parameter  (smaller means minimize for longer)
      io {
        restart_file = None
          .type = str
          .help = output file for re-starting a simulation
        output_dir = 'TEST'
          .type = str
          .help = optional output directory
      }
      quiet = False
        .type = bool
        .help = silence the refiner
      verbose = 0
        .type = int
        .help = verbosity level (0-10) for nanoBragg
      num_macro_cycles = 1
        .type = int
        .help = keep repeating the same refinement scheme over and over, this many times
      ncells_mask = *000 110 101 011 111
        .type = choice
        .help = a mask specifying which ncells parameters should be the same
        .help = e.g. 110 specifies Na and Nb are refined together as one parameter
      reference_geom = None
        .type = str
        .help = path to expt list file containing a detector model
      stage_two {
        use_nominal_hkl = True
          .type = bool
          .help = use the nominal hkl as a filter for Fhkl gradients
        save_model_freq = 50
          .type = int
          .help = save the model  after this many iterations
        save_Z_freq = 25
          .type = int
          .help = save Z-scores for all pixels after this many iterations
        min_multiplicity = 1
          .type = int
          .help = structure factors whose multiplicity falls below this value
          .help = will not be refined
        Fref_mtzname = None
          .type = str
          .help = path to a reference MTZ file. if passed, this is used solely to
          .help = observe the R-factor and CC between it and the Fobs being optimized
        Fref_mtzcol = "Famp(+),Famp(-)"
          .type = str
          .help = column in the mtz file containing the data
        d_min = 2
          .type = float
          .help = high res lim for binner
        d_max = 999
          .type = float
          .help = low res lim for binner
        n_bin = 10
          .type = int
          .help = number of binner bins
      }
    }
    """

    roi_phil = """
    roi {
      cache_dir_only = False
        .type = bool
        .help = if True, create the cache folder , populate it with the roi data, then exit
      fit_tilt = False
        .type = bool
        .help = fit tilt plane, or else background is simply an offset
      force_negative_background_to_zero = False
        .type = bool
        .help = if True and the background model evaluates to a negative number
        .help = within an ROI, then force the background to be 0 for all pixels in that ROI
      background_threshold = 3.5
        .type = float
        .help = for determining background pixels
      pad_shoebox_for_background_estimation = None
        .type = int
        .help = shoebox_size specifies the dimenstion of the shoebox used during refinement
        .help = and this parameter is used to increase that shoebox_size only during the background
        .help = estimation stage
      shoebox_size = 10
        .type = int
        .help = roi box dimension
      deltaQ = None
        .type = float
        .help = roi dimension in inverse Angstrom, such that shoeboxes at wider angles are larger.
        .help = If this parameter is supplied, shoebox_size will be ignored.
      reject_edge_reflections = True
        .type = bool
        .help = whether to reject ROIs if they occur near the detector panel edge
      reject_roi_with_hotpix = True
        .type = bool
        .help = whether to reject an ROI if it has a bad pixel
      hotpixel_mask = None
        .type = str
        .help = path to a hotpixel mask (hot pixels set to True)
      panels = None
        .type = str
        .help = panel list for refinement as a string, e.g. "0-8,10,32-40" . The ranges are inclusive,
        .help = e.g. 0-8 means panels 0,1,2,3,4,5,6,7,8
      fit_tilt_using_weights = True
        .type = bool
        .help = if not using robust estimation for background, and instead using tilt plane fit,
        .help = then this parameter will toggle the use of weights. Weights are the estimated
        .help = pixel variance, incuding readout and shot noises.
      allow_overlapping_spots = False
        .type = bool
        .help = if True, then model overlapping spots
    }
    """

    preditions_phil = """
    predictions {
      weak_fraction = 0.5
        .type = float
        .help = fraction of weak predictions to integrate
      threshold = 1e-3
        .type = float
        .help = value determining the cutoff for the forward model intensity. Bragg peaks will then be determined
        .help = as regions of connected values greater than the threshold
      oversample_override = None
        .type = int
        .help = force the pixel oversample rate to this value during the forward model simulation
        .help = for maximum speed gains, set this to 1, but inspect the output!
        .expert_level=10
      use_diffBragg_mtz = False
        .type = bool
        .help = whether to use the mtz supplied to diffBragg for prediction
      Nabc_override = None
        .type = ints(size=3)
        .help = use this value of mosaic block size for every shot, useful to get more predicted spots
        .expert_level=10
      pink_stride_override = None
        .type = int
        .help = if specified, stride through the spectrum according to this interval
      default_Famplitude = 1e3
        .type = float
        .help = default structure factor amplitude for every miller index
        .help = this creates a flat prediction model, where the magnitude is dependent on the distance to the Ewald sphere
      resolution_range = [1,999]
        .type = floats(size=2)
        .help = high-res to low-res limit for prediction model
      symbol_override = None
        .type = str
        .help = specify the space group symbol to use in diffBragg (e,g, P43212),
        .help = if None, then it will be pulled from the crystal model
      method = *diffbragg exascale
        .type = choice
        .help = engine used for computing the forward model
        .help = diffbragg offers CUDA support via the DIFFBRAGG_USE_CUDA=1 environment variable specification
        .help = or openmp support using the OMP_NUM_THREADS flag
        .help = The exascale only uses CUDA (will raise error if CUDA is not confugured)
    }
    """

    philz = script_phil + hopper_phil + simulator_phil + refiner_phil + roi_phil + preditions_phil
    phil_scope = parse(philz)
    arg_interp = phil_scope.command_line_argument_interpreter(home_scope="")

    phil_file = """
    roi.shoebox_size = 13
    roi.fit_tilt = True
    roi.reject_edge_reflections = False
    roi.pad_shoebox_for_background_estimation=10
    relative_tilt = False
    refiner.force_symbol=P6522
    refiner.refine_Fcell = [1]
    refiner.refine_spot_scale = [1]
    refiner.max_calls = [501]
    refiner.ncells_mask = 000
    refiner.tradeps = 1e-20
    refiner.verbose = 0
    refiner.sigma_r = 1.5
    refiner.adu_per_photon = 9.481
    refiner.stage_two.min_multiplicity = 2
    simulator.crystal.has_isotropic_ncells = False
    simulator.beam.size_mm = 0.001
    spectrum_from_imageset=True
    downsamp_spec {
      delta_en = 1
    }
    """
    user_phil = parse(phil_file)
    phil_sources = [user_phil]

    working_phil, unused = phil_scope.fetch(sources=phil_sources, track_unused_definitions=True)
    for loc in unused:
        print("WARNING: unused phil:", loc)

    params = working_phil.extract()
    refiner = global_refiner_from_parameters(params)
