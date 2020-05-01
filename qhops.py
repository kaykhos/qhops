#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Mar 5 17:27:19 2020

@author: Kiran
Setting up the hilbert space structire for site hopping simulations
    + Spesify dimentions and delta beta magnitudes
    + Constructs random Hamiltonians and can unitarilly simulate them for some time
    + All energy scales in EV
    + Site energies are 1/cm energy scales, 
    + t = 1 is 1cm of propogation = 3.3333e-11 s
"""

import pdb
import copy
import time
import numpy as np
import qutip as qt
import networkx as nx
import matplotlib.pyplot as plt


# number defaults
NB_DIMS = 7
NB_DELTA_BETA = 10 # unts of 1/cm
NB_BETA = 0 # only relative changes really matter
NB_SEED = 10
NB_INITIAL_STATE = 0
NB_TIME_STEP = 0.1
NB_SINK_RATE = 10
NB_WAVEGUIDE_SEGMENTS = 16
NB_TOTAL_TIME = 1

# boolean defaults
BL_RECALC_COUPLINGS = False
BL_VERBOSE = False

# List defaults
LS_COUPLINGS =[[0, 1, 96], # e.g. site 0 and 1 are coupeled at 96 /cm
               [1, 2, 33],
               [2, 3, 51.1],
               [3, 4, 76.6],
               [4, 5, 78.3],
               [5, 6, 38.3],
               [3, 6, 67]]
LS_SINK_COUPELED_SITES = [3, 5]

# Misc defaults
QT_OPTIONS = qt.Options()
QT_OPTIONS.nsteps = 150000


# ==============================================
# Simulator base class, updating and propogating
# ==============================================
class simulator():
    """ Class that deals with simulations of the FMO wave guide experiments
        + Can be called with default argumets and gives results
        + Generate or import spesific coupling Hamiltonians
        + Stores initial and curent quantum states
        + Can evolve under unitary / dissipative evolutions 
        + Freedom to choose where sources and sinks are
        + Properly simulates different lengths of wave guides 
        + Ability to simulate a small bit of wave guide, then update the Hamiltonian
        + The Free Hamiltonian is all that needs to be spesified, but under the
            hood, I append extra sites for sinks - this is the extended_hamiltonian
            + If you use class.expect(operator) it will auto ensure the 
              operator is the right dimentions
        + 
        """
    def __init__(self, 
                 hamiltonian=None,
                 dims=NB_DIMS, 
                 delta_beta=NB_DELTA_BETA,
                 beta=NB_BETA,
                 couplings=LS_COUPLINGS,
                 recalc_couplings=BL_RECALC_COUPLINGS,
                 seed=NB_SEED,
                 initial_state=NB_INITIAL_STATE,
                 sink_sites=LS_SINK_COUPELED_SITES,
                 sink_rate=NB_SINK_RATE,
                 verbose=BL_VERBOSE):
        """ hamiltonian = qt.Qobj or np array of the full hamiltonian, 
            dims = number of sites
            delta_beta = delta beta fluctuations (flat distribution for now)
            couplings = magnitude, or sparse/full matrix of couplings between sites, 
            recalc_couplings = bool to update couplngs by loking at beta_i - beta_j
            seed = random seed for random circuit gen
            initial state = qt.Obj or single site, or list of sites for coherent initial state 
            sink_sits = single or list of sites that have sinces
            sink_rate = single or list of coupling rates of sink sites """
        # Basic propties
        self.verbose = verbose
        self.dims = dims
        self._sink_sites = np.atleast_1d(sink_sites)
        self._sink_rate = np.atleast_1d(sink_rate)
        self._recalc_couplings = recalc_couplings
        self.hamiltonian = self._gen_hamiltonian(hamiltonian=hamiltonian,
                                                 delta_beta=delta_beta,
                                                 beta=beta,
                                                 couplings=couplings,
                                                 seed=seed)
        # To be implimented in subclass
        self._extended_hamiltonian = self._gen_extended_hamiltonian()
        self._dissipative_ops = self._gen_dissipative_ops()
        
        # Implimentation requires subclaass generators
        self.initial_state = self._gen_initial_state(initial_state)
        self.current_state = self.initial_state
        self.basis = self._gen_basis_ops()
        # self.time_evolution_list = copy.deepcopy([])
        self.output = copy.deepcopy({})
    
    def _gen_extended_hamiltonian(self):
        """ Generates extendedn Hamiltonian (inc. dissipative operators)"""
        raise NotImplementedError
        
    def _gen_dissipative_ops(self):
        """ Generates dissipative operators (inc. incoherent hopping terms)"""
        raise NotImplementedError

    def _gen_hamiltonian(self,
                         hamiltonian=None,
                         delta_beta=NB_DELTA_BETA,
                         beta=NB_BETA,
                         couplings=LS_COUPLINGS,
                         seed=None):
        """ Generates random Hamiltonian or check input Hamiltonian
            If a Hamiltonian is spesified, non of the parameters are used 
                except recalc_couplings
            Random Hamiltonian assumed NN couplings only (can change this)
            By default not seeded, so updates will follow from the initial seed
            Helper functions deal with different input types and coupling options"""
        hamiltonian_gen_dict = {'delta_beta':delta_beta,
                                'beta':beta,
                                'couplings':couplings,
                                'seed':seed}
        self._hamiltonian_gen_dict = hamiltonian_gen_dict
        if hamiltonian != None:
            hamiltonian = qt.Qobj(hamiltonian)
            assert hamiltonian.isherm, "This Hamiltonian is not hermetian"
            self.dims = hamiltonian.dims[0][0]
            if self._recalc_couplings:
                hamiltonian = self._x_recalc_couplings(hamiltonian)
                if self.verbose: print("Warning couplings from spesified Hamiltonians have been re-normalized using delta beta: see obj._recalc_couplings")
            return hamiltonian
        
        if seed != None: np.random.seed(seed)
        dims = self.dims
        if type(beta) == float or type(beta) == int:
            site_energy = delta_beta*(2*np.random.random(dims)-1) + beta
        else:
            site_energy = beta
            if self.verbose: print(" Using spesified values for beta, adding delta_beta in addition")
        hamiltonian = 1j*np.zeros((dims, dims))
        for row in range(dims):
            for col in range(dims):
                if row == col:
                    hamiltonian[row, col] = site_energy[row]
        hamiltonian = self._x_add_coupling_terms(hamiltonian, couplings)
        hamiltonian = (hamiltonian + hamiltonian.conjugate().transpose())/2
        hamiltonian = qt.Qobj(hamiltonian)
        if self._recalc_couplings:
            hamiltonian = self._x_recalc_couplings(hamiltonian)
            if self.verbose: print("Warning couplings from spesified Hamiltonians have been re-normalized using delta beta: see obj._recalc_couplings")
        return hamiltonian
    
    def _x_add_coupling_terms(self, np_hamiltonian, couplings):
        """ Adds coupling terms to the Hamiltonian, either by full/sparse matri, 
            or by single values"""
        dims = self.dims
        if type(couplings) == list or type(couplings) == np.ndarray:
            if self.verbose: print("Warning when spesifying couplingss, only the upper triangular (row<col) elements are used")
            is_sparse = (len(couplings[0]) == 3)
            is_full = (len(couplings[0]) == dims)
            if is_sparse and is_full: assert False, "For now code cannot deal with 3 side model (coupling matrix is ambiguous)"
            if is_sparse:
                for triple in couplings:
                    if triple[0] < triple[1]:
                        np_hamiltonian[triple[0], triple[1]] = 2*triple[2]
            if is_full:
                assert type(couplings) == np.ndarray, "Please input np.ndarray for a full coupling matrix"
                for row in range(dims):
                    for col in range(dims):
                        if row < col:
                            np_hamiltonian[row, col] = 2*couplings[row,col]
        else:
            for row in range(dims):
                for col in range(dims):
                    if col == row+1:
                        np_hamiltonian[row, col] = 2*couplings
        return np_hamiltonian
    
    def _x_recalc_couplings(self, hamiltonian):
        """ Updates coupling terms based of beta_i-beta_j due to change in RI"""
        hamiltonian = np.array(hamiltonian)
        dims = self.dims
        for row in range(dims):
            for col in range(dims):
                if row != col and hamiltonian[row,col] !=0:
                    db = hamiltonian[row,row] - hamiltonian[col,col]
                    cij = hamiltonian[row,col]
                    ceff = np.sqrt( (db / 2)**2 + cij**2 )
                    hamiltonian[row,col]  = ceff
        return qt.Qobj(hamiltonian)

    def _gen_initial_state(self, initial_state=NB_INITIAL_STATE):
        """ Resutns intial state, eiter site number/numbers or qt.Obj"""
        extended_dims = self._extended_dims
        if type(initial_state) == int:
            psi = qt.basis(extended_dims, initial_state)
            rho = psi * psi.dag()
            return rho
        elif type(initial_state) == list:
            psi = sum([qt.basis(extended_dims, ii) for ii in initial_state])
            rho = psi * psi.dag()
            return rho / rho.norm()
        elif type(initial_state) == qt.qobj.Qobj:
            return initial_state
        else:
            raise NotImplementedError
    
    def _gen_basis_ops(self, extended=False):
        """ Generates set of basis operators to visualize where the excitation is"""
        ls = []
        for ii in range(self._extended_dims):
            state = qt.basis(self._extended_dims, ii)
            op = state * state.dag()
            ls.append(op)
        return ls
    
    def _extend_operator(self, operator):
        """ Adds extra dimentions to the operator so total 
            dimention = num_sinks + site dimention, works on arbitrary operator
            acting on the sites"""
        if list(np.squeeze(operator.dims)) == [self._extended_dims, self._extended_dims]:
            return operator
        elif list(np.squeeze(operator.dims)) == [self.dims, self.dims]:
            extended = np.zeros((self._extended_dims, self._extended_dims))*1j
            extended[:self.dims,:self.dims] = np.array(operator)
            return qt.Qobj(extended)
        else:
            assert False, " Unknown dimentions of input operator"
            
    def unitary_timestep(self, 
                         time=NB_TIME_STEP,
                         from_start=False,
                         options=QT_OPTIONS):
        """ Propogates current_state forward by time, updates the current state
            and save the unitary operator.
            - Option from_start = True allows researt from initial state"""
        if from_start:
            if self.verbose: print("unitary list and current state have been reset")
            rho = self.initial_state
            # self.time_evolution_list = []
        else:
            rho = self.current_state
        unitary = (-1j * time * self._extended_hamiltonian).expm()
        # self.time_evolution_list.append(unitary)
        rho_t = unitary * rho * unitary.dag()
        self.current_state = rho_t
    
    def me_timestep(self, 
                    time=NB_TIME_STEP, 
                    from_start=False,
                    options=QT_OPTIONS):
        # pdb.set_trace()
        """ Propogates forward by non-unitary evolution including the sinks, 
            to update the current state, 
            Uses qt.mesolve and saves the whole result in the time_evolution_list
            Same from_start ability as unitary_timestep"""
        if from_start:
            if self.verbose: print("unitary list and current state have been reset")
            rho = self.initial_state
            # self.time_evolution_list = []
        else:
            rho = self.current_state
        H = self._extended_hamiltonian
        cops = self._dissipative_ops
        if type(time) == list or type(time) == np.ndarray:
            times = time
        else:
            times = np.array([0, time])
        result = qt.mesolve(H, rho, times, c_ops = cops, e_ops = [], options=options)
        self.current_state = result.states[-1]
        # self.time_evolution_list.append(result)
    
    def expect(self, operator=None):
        """calculates the expectation value of an operator, user only need spesify
            a site basis operator, extention is handeled automatically"""
        if operator == None:
            operator = self._extended_hamiltonian
        operator = self._extend_operator(operator)
        return qt.expect(operator, self.current_state)
    
    def site_occupation(self, plot=False):
        """ Returns the site occupation (including sinks) with the ability to plot"""
        ls = []
        for bb in self.basis:
            ls.append(self.expect(bb))
        if plot: 
            sites = ls[:self.dims]
            sinks = ls[self.dims:]
            plt.plot(sites, 'bd', label='sites')
            if '_irreversible_' in str(self.__class__):
                plt.plot(self._sink_sites, sinks, 'r*', label='sinks')
                plt.axis([-1, self._extended_dims, 0, 1])
            elif '_reversible_' in str(self.__class__):
                plt.plot(self.dims, sum(sinks), 'r*', label='all sinks')
                plt.axis([-1, self.dims+1, 0, 1])
            plt.xlabel('site/sink')
            plt.ylabel('occupation')
            plt.title('Occupation plot')
            plt.legend()
            plt.show()
        return ls
    
    def transport_efficiency(self):
        """ Returns transport efficiency by looking at the probabilities of 
            what is NOT on the list of sites """
        occupations = self.site_occupation()
        relevant = [occupations[ss] for ss in range(self.dims)]
        return 1 - sum(relevant)
        # if sites == None:
        # elif is_iter(sites) and max(sites) < self.dims:
        #     sites = sites
        #     relevant = [occupations[ii] for ss in sites]
        #     print("Warning: double check how this efficiency is defined")
        # elif sites < self.dims:
        #     sites = [sites]
        #     print("Warning: double check how this efficiency is defined")
        # elif sites >= self.dims:
        #     assert False, "transport efficiency to fixed sink site is not implimented yet"
        # return 1 - sum(relevant)

    def update_hamiltonian(self,
                           hamiltonian=None,
                           delta_beta=None,
                           beta=None,
                           couplings=None,
                           seed=None):
        """ Updates Hamiltonian e.g. next slice of the wave guide
            BEHAVIOUR: if delta_beta, beta or couplings are not spesified, 
                these values are the same as construction. 
              E.g. if none are given a new Hamiltonian is generated ONLY if the 
              seed is different 
              Again, if hamiltonian IS given, none of the other parameters do anything"""
        di = self._hamiltonian_gen_dict
        if delta_beta == None: delta_beta = di['delta_beta']
        if beta == None: beta = di['beta']
        if couplings == None: couplings = di['couplings']
        
        hamiltonian = self._gen_hamiltonian(hamiltonian=hamiltonian,
                                            delta_beta=delta_beta,
                                            beta=beta, 
                                            couplings=couplings,
                                            seed=seed)
        self.hamiltonian = hamiltonian
        self._extended_hamiltonian = self._gen_extended_hamiltonian()
        if self.verbose: print('Hamiltonian updated')
    
    def simulate_waveguide(self, 
                           nb_segments=NB_WAVEGUIDE_SEGMENTS, 
                           total_time=NB_TOTAL_TIME,
                           include_sinks=True, 
                           hamiltonian_list=None, 
                           options=QT_OPTIONS,
                           plot_mid=False,
                           plot_end=False):
        if self.verbose and len(self.output) != 0:
            print("This will overwrite the stored output")
                
        """Simulates the FMO wave guie for mutiple segments returs a useful results dictionary"""
        if hamiltonian_list != None:
            assert len(hamiltonian_list) == nb_segments, "to spesify Hamiltonians, must spesify one pre segment"
        else:
            hamiltonian_list = [None for ii in range(nb_segments)]
        time_per_segment = total_time / nb_segments
        output = {'rho':[],
                  'transport_efficiency':[],
                  'occupation':[],
                  'purity':[],
                  'dims':[self.dims, self._extended_dims]}
        for ii in range(nb_segments):
            if include_sinks:
                self.me_timestep(time=time_per_segment,
                                 from_start=False,
                                 options=options)
            else:
                self.unitary_timestep(time=time_per_segment,
                                      from_start=False)
            output['rho'].append(self.current_state)
            output['transport_efficiency'].append(self.transport_efficiency())
            output['occupation'].append(self.site_occupation())
            output['purity'].append((self.current_state**2).norm())
            if plot_mid: self.site_occupation(plot=True)
            self.update_hamiltonian(hamiltonian=hamiltonian_list[ii])
        if plot_end: self.site_occupation(plot=True)
        output['occupation']  = np.array(output['occupation'])
        output['purity']  = np.array(output['purity'])
        output['transport_efficiency']  = np.array(output['transport_efficiency'])
        self.output = output
        return output

    def plot_operator(self, operator=None, log_scale=True):
        if operator == None:
            operator = self._extended_hamiltonian
        operator = np.array(operator)
        if log_scale:
            operatorr = np.log(np.abs(operator.real) + 1e-16)
            operatori = np.log(np.abs(operator.imag) + 1e-16)
        else:
            operatorr = operator.real
            operatori = operator.imag
        plt.subplot(2, 1, 1)
        plt.pcolor(operatorr)
        plt.colorbar()
        plt.title('real')
        plt.subplot(2, 1, 2)
        plt.pcolor(operatori)
        plt.colorbar()
        plt.title('imag')
        plt.show()
    
    def plot_waveguide(self, 
                         include_sinks=False,
                         binary_colors=False,
                         max_sinks=3):
        graph, node_cols, edge_cols = self._return_graph(include_sinks=include_sinks,
                                                         binary_colors=binary_colors,
                                                         max_sinks=max_sinks)
        pos = nx.spring_layout(graph, iterations=500)
        vmin = min(node_cols + edge_cols)
        vmax = max(node_cols + edge_cols)
        cmap=plt.cm.Blues
        nx.draw(graph, pos, 
                node_color=node_cols, 
                edge_color=edge_cols,
                vmin=vmin,
                vmax=vmax)
        #nx.draw_networkx_edges(graph,pos)
        nx.draw_networkx_labels(graph,pos,font_size=10, font_color='w')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
        # sm._A = []
        plt.colorbar(sm)
        plt.title(str(self.__class__))
        plt.show()
    
    
    def quick_summary(self):    
        quick_summary(self.output)
    
    
    def _return_graph(self, 
                      include_sinks=False, 
                      binary_colors=False,
                      max_sinks=3):
        """ Returns node and edge colors spesified by the matrix, and and lets
            the user ignore"""
        if include_sinks:
            hamiltonian = np.abs(np.array(self._extended_hamiltonian))
            if binary_colors:
                site_cols = [np.max(hamiltonian)]*self.dims
                sink_cols = [0]*(self._extended_dims - self.dims)
                node_cols = site_cols + sink_cols
            else:
                node_cols = np.diag(hamiltonian)
        else:
            hamiltonian = np.abs(np.array(self.hamiltonian))
            node_cols = np.diag(hamiltonian)
        graph = nx.from_numpy_array(hamiltonian)
        edges = graph.edges()
        edge_cols = []
        for ee in edges:
            edge_cols.append(hamiltonian[ee])
        node_cols = list(node_cols)
        return graph, node_cols, edge_cols
       

def quick_summary(output):
    nb_all_sites = output['dims'][1]
    nb_sites = output['dims'][0]
    """ Quick summary of results form single simulatio"""
    plt.figure(figsize=(10,10))
    grid = plt.GridSpec(2, nb_all_sites, wspace=0.4, hspace=0.3)
    half_way = int(np.ceil(nb_all_sites/2))
    transport = plt.subplot(grid[0, :half_way])
    purity = plt.subplot(grid[0,half_way:])
    
    transport.plot(output['transport_efficiency'])
    transport.set_title('transport efficiency')
    purity.plot(output['purity'])
    purity.set_title('purity (inc. sinks)')
    
    
    for ss in range(nb_all_sites):
        ax = plt.subplot(grid[1,ss])
        if ss >= nb_sites:
            ax.plot(output['occupation'][:,ss], 'r')
        else:
            ax.plot(output['occupation'][:,ss], 'b')
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        if ss == 0:
            ax.set_axis_on()
            ax.set_ylabel('site occ. over time [0,1]')
            


class waveguide_irreversible_sink(simulator):
    """ Simulate an actual wave guie, with irreversible couplings
        * sink_sites is a list of where irreversible couplings are attached, 
        * sink_rates is a single number giving the coupling rate to all sinks
          OR a list of a different sink rate for each site"""
            
    def _gen_extended_hamiltonian(self):
        """ Free Hamiltonian acts only on sites, but the evolution requires sinks, 
            this extension just ensures the right dimentions for time evolution"""
        self._extended_dims = self.dims + len(self._sink_sites)
        extended_ham = self._extend_operator(self.hamiltonian)
        return extended_ham
    
    def _gen_dissipative_ops(self):
        """ Generates sink operators attachd to sink sites, with spesified sink rate"""
        sink_sites = self._sink_sites
        sink_rates = self._sink_rate
        nb_sink_sites = len(sink_sites)
        if nb_sink_sites != len(sink_rates):
            sink_rates = [sink_rates[0]] * nb_sink_sites
        dims = self.dims
        sink_ops = []
        for ss in range(nb_sink_sites):
            site = qt.basis(dims + nb_sink_sites, sink_sites[ss])
            sink = qt.basis(dims + nb_sink_sites, dims + ss)
            sink_ops.append(sink * site.dag() * np.sqrt(2*sink_rates[ss]))
        return sink_ops


class waveguide_reversible_sink(simulator):
    """ Simulate the reversible sink wave guide, adding many coupled reversible 
        sites, strongly coupeled to a single site
        * sink_sites is a LIST of where the site is coupeled AND how many sites to add,
          E.g. sink_sites = [3, 50] will add 50 extra LINEARLY coupeled sites
               attached to site 3, the coupling rates will be spesified by 
               sink_rates
        * Also supports mutiple sinks, e.g. [[3, 20], [6, 25]] will add 
          20 sinks to site 3 and 25 sinks to site 6, 
        * If sink_rates = is a list, couplngs for sites 3 and 6 can be chosen
          independently"""
    def _gen_extended_hamiltonian(self):
        sink_sites = np.atleast_2d(self._sink_sites)
        sink_rates = self._sink_rate
        if len(sink_rates) != len(sink_sites):
            sink_rates = [sink_rates[0]] * len(sink_sites)
        

        total_extra_sites = sink_sites[:,1].sum()
        self._extended_dims = self.dims + total_extra_sites
        extended_ham = np.array(self._extend_operator(self.hamiltonian))
        sink_starts = sink_sites[:,1].cumsum() - sink_sites[:,1] + self.dims
        for ss in range(len(sink_sites)):
            sink = sink_sites[ss]
            extended_ham[sink[0], sink_starts[ss]] = sink_rates[ss]
            extended_ham[sink_starts[ss], sink[0]] = sink_rates[ss]
            for jj in range(sink[1] - 1):
                jj+=sink_starts[ss]
                extended_ham[jj, jj+1] = sink_rates[ss]
                extended_ham[jj+1, jj] = sink_rates[ss]
        extended_ham = qt.Qobj(extended_ham)
        return extended_ham
    
    def _gen_dissipative_ops(self):
        return []

        
    
def is_iter(obj):
    """ Returns check for __iter__-able objects"""
    try:
        obj.__iter__
        return True
    except:
        return False


            
if __name__ == '__main__':
    test = waveguide_reversible_sink(sink_sites=[[2,5],[5, 3]], 
                                      couplings = LS_COUPLINGS, 
                                      seed=10, 
                                      delta_beta=100)
    
    test.plot_waveguide(include_sinks=True)