import numpy as np
from scipy.linalg import sqrtm

from qibo import gates, models
from qibo.quantum_info import hellinger_distance, hellinger_fidelity


def noisy_circuit(circuit, params):
    """Creates a noisy circuit from the circuit given as argument.

    The function applies a :class:`qibo.gates.ThermalRelaxationChannel` after each step of the circuit
    and, after each gate, a :class:`qibo.gates.DepolarizingChannel`, whose parameter depends on whether the
    gate applies on one or two qubits. In the end are applied  asymmetric bitflips on measurement gates.


    Args:
        circuit (qibo.models.Circuit): Circuit on which noise will be applied. Since in the end are
        applied bitflips, measurement gates are required.
        params (dictionary): object which contains the parameters of the channels organized as follow
        params = {"t1" : (t1, t2,..., tn),
          "t2" : (t1, t2,..., tn),
          "gate time" : (time1, time2),
          "excited population" : 0,
          "depolarizing error" : (lambda1, lambda2),
          "bitflips error" : ([p1, p2,..., pm], [p1, p2,..., pm]),
          "idle_qubits" : 1
         }
        Where n is the number of qubits, and m the number of measurement gates.
        The first four parameters are used by the thermal relaxation error. The first two  elements are the
        tuple containing the $T_1$ and $T_2$ parameters; the third one is a tuple which contain the gate times,
        for single and two qubit gates; then we have the excited population parameter.
        The fifth parameter is a tuple containing the depolaraziong errors for single and 2 qubit gate.
        The sisxth parameter is a tuple containg the two arrays for bitflips probability errors: the first one implements 0->1 errors, the other one 1->0.
        The last parameter is a boolean variable: if True the noise model takes into account idle qubits.

    Returns:
        The new noisy circuit (qibo.models.Circuit).


    """
    # parameters of the model
    t1 = params["t1"]
    t2 = params["t2"]
    time1 = params["gate_time"][0]
    time2 = params["gate_time"][1]
    excited_population = params["excited_population"]
    depolarizing_error_1 = params["depolarizing_error"][0]
    depolarizing_error_2 = params["depolarizing_error"][1]
    bitflips_01 = params["bitflips_error"][0]
    bitflips_10 = params["bitflips_error"][1]
    idle_qubits = params["idle_qubits"]

    # new circuit
    noisy_circ = models.Circuit(circuit.nqubits, density_matrix=True)

    # time steps of the circuit
    time_steps = max(circuit.queue.moment_index)

    # current_time keeps track of the time spent by the qubits
    # being manipulated by the gates of the circuit
    current_time = np.zeros(circuit.nqubits)

    # the idea behind ths loop is to build the old circuit adding the noise channels and
    # keeping track of the time qubits spend being manipulated by the gates, in order
    # to correct the thermal relaxation time of each qubit, even if they are idle.
    for t in range(time_steps):
        # for each time step, I look for each qubit what gate are applied
        for qubit in range(circuit.nqubits):
            # if there's no gate, move on!
            if circuit.queue.moments[t][qubit] == None:
                pass
            # measurement gates
            elif isinstance(circuit.queue.moments[t][qubit], gates.measurements.M):
                for key in list(circuit.measurement_tuples):
                    # if there is a 2-qubits measurement gate we must check that both qubit intercated
                    # with the environment for the same amount of time. If not, before applying
                    # the 2-qubits gate we apply the therm-rel channel for the time difference
                    if len(circuit.measurement_tuples[key]) > 1:
                        q1 = circuit.measurement_tuples[key][0]
                        q2 = circuit.measurement_tuples[key][1]
                        if current_time[q1] != current_time[q2] and idle_qubits == True:
                            q_min = q1
                            q_max = q2
                            if current_time[q1] > current_time[q2]:
                                q_min = q2
                                q_max = q1
                            time_difference = current_time[q_max] - current_time[q_min]
                            # this is the thermal relaxation channel which model the intercation
                            # of the idle qubit with the environment
                            noisy_circ.add(
                                gates.ThermalRelaxationChannel(
                                    q_min,
                                    t1[q_min],
                                    t2[q_min],
                                    time_difference,
                                    excited_population,
                                )
                            )
                            # update the qubit time
                            current_time[q_min] += time_difference
                q = circuit.queue.moments[t][qubit].qubits
                # adding measurements gates
                if len(circuit.queue.moments[t][qubit].qubits) == 1:
                    q = q[0]
                    noisy_circ.add(gates.M(q, p0=bitflips_01[q], p1=bitflips_10[q]))
                else:
                    p0q = []
                    p1q = []
                    for j in q:
                        p0q.append(bitflips_01[j])
                        p1q.append(bitflips_10[j])
                    noisy_circ.add(gates.M(*q, p0=p0q, p1=p1q))
                    circuit.queue.moments[t][
                        max(circuit.queue.moments[t][qubit].qubits)
                    ] = None
            # if there is a 1-qubit gate I add the old gate, the dep and therm-rel channels
            elif len(circuit.queue.moments[t][qubit].qubits) == 1:
                noisy_circ.add(circuit.queue.moments[t][qubit])
                noisy_circ.add(
                    gates.DepolarizingChannel(
                        circuit.queue.moments[t][qubit].qubits, depolarizing_error_1
                    )
                )
                noisy_circ.add(
                    gates.ThermalRelaxationChannel(
                        qubit,
                        t1[qubit],
                        t2[qubit],
                        time1,
                        excited_population,
                    )
                )
                # I update the qubit time
                current_time[qubit] += time1
            # if there is a 2-qubits gate we must check that both qubit intercated
            # with the environment for the same amount of time. If not, before applying
            # the 2-qubits gate we apply the therm-rel channel for the time difference
            else:
                q1 = circuit.queue.moments[t][qubit].qubits[0]
                q2 = circuit.queue.moments[t][qubit].qubits[1]
                if current_time[q1] != current_time[q2] and idle_qubits == True:
                    q_min = q1
                    q_max = q2
                    if current_time[q1] > current_time[q2]:
                        q_min = q2
                        q_max = q1
                    time_difference = current_time[q_max] - current_time[q_min]
                    # this is the thermal relaxation channel which model the intercation
                    # of the idle qubit with the environment
                    noisy_circ.add(
                        gates.ThermalRelaxationChannel(
                            q_min,
                            t1[q_min],
                            t2[q_min],
                            time_difference,
                            excited_population,
                        )
                    )
                    # I update the qubit time
                    current_time[q_min] += time_difference
                # I add the 2-qubit gate, dep and therm-rel channels
                noisy_circ.add(circuit.queue.moments[t][qubit])
                noisy_circ.add(
                    gates.DepolarizingChannel(
                        tuple(set(circuit.queue.moments[t][qubit].qubits)),
                        depolarizing_error_2,
                    )
                )
                noisy_circ.add(
                    gates.ThermalRelaxationChannel(
                        q1, t1[q1], t2[q1], time2, excited_population
                    )
                )
                noisy_circ.add(
                    gates.ThermalRelaxationChannel(
                        q2, t1[q2], t2[q2], time2, excited_population
                    )
                )
                # I update the qubit time
                current_time[circuit.queue.moments[t][qubit].qubits[0]] += time2
                current_time[circuit.queue.moments[t][qubit].qubits[1]] += time2
                circuit.queue.moments[t][
                    max(circuit.queue.moments[t][qubit].qubits)
                ] = None

    # setting noisy_circ.measurements
    measurements = []
    for m in circuit.measurements:
        q = m.qubits
        if len(q) == 1:
            q = q[0]
            measurements.append(gates.M(q, p0=bitflips_01[q], p1=bitflips_10[q]))
        else:
            p0q = []
            p1q = []
            for j in q:
                p0q.append(bitflips_01[j])
                p1q.append(bitflips_10[j])
            measurements.append(gates.M(*q, p0=p0q, p1=p1q))
    noisy_circ.measurements = measurements

    return noisy_circ


def freq_to_prob(freq):
    """Transforms a dictionary of frequencies in an array of probabilities.

    Args:
        freq (CircuitResult.frequencies): frequencies you want to transform.

    Returns:
        The new array (numpy.ndarray).
    """
    norm = sum(freq.values())
    nqubits = len(list(freq.keys())[0])
    prob = np.zeros(2**nqubits)
    for k in range(2**nqubits):
        index = "{:b}".format(k).zfill(nqubits)
        prob[k] = freq[index] / norm
    return prob


def hellinger_shot_error(p, q, nshots):
    """Hellinger fidelity error caused by using two probability distributions estimated using a finite number of shots.
    It is calculated propagating the probability error of each state of the system. The complete formula is:
    :math:`4 * H(p, q) * (1 - H^{2}(p, q)) * (\\sum_{i=1}^{n}(|1 - \\sqrt{\frac{q_i}{p_i}}| * \\sqrt{\frac{p_i - {p_i}^2}{nshots}} + |1 - \\sqrt{\frac{p_i}{q_i}}| * \\sqrt{\frac{q_i - {q_i}^2}{nshots}}) / (4 * H(p,q))`
    where the sum is made all over the possible states and :math:`H(p, q)` is the Hellinger distance.

       Args:
           p (numpy.ndarray): (discrete) probability distribution :math:`p`.
           q (numpy.ndarray): (discrete) probability distribution :math:`q`.
           nshots (int): the number of shots we used to run the circuit to obtain :math:`p` and :math:`q`.

       Returns:
           (float): The Hellinger fidelity error.

    """
    prob_p = np.sqrt((p - p**2) / nshots)
    prob_q = np.sqrt((q - q**2) / nshots)
    hellinger_dist = hellinger_distance(p, q)
    hellinger_dist_e = np.sum(
        (abs(1 - np.sqrt(q / p)) * prob_p + abs(1 - np.sqrt(p / q)) * prob_q)
        / (4 * hellinger_dist)
    )
    hellinger_fid_e = 4 * hellinger_dist * (1 - hellinger_dist**2) * hellinger_dist_e
    return hellinger_fid_e


def loss(parameters, grad, args):
    """The loss function used to be maximized in the fit method of the :class:`qibo.noise_model.CompositeNoiseModel`.
    It's the hellinger fidelity calculated between the probability distribution of the noise model and the experimental target distribution using the :func:`qibo.quantum_info.hellinger_fidelity`.
    It is possible to return also the finite shot error correction calculated with the :func:`qibo.noise_model.hellinger_shot_error`.

       Args:
           parameters (numpy.ndarray): parameters of the :func:`qibo.noise_model.noisy_circuit` which must be inferred.
           They must be given in form of array as
           array([params["t1"], params["t2"], params["gate_time"], params["depolarizing_error"], params["bitflips_error"]])
           q (numpy.ndarray): (discrete) probability distribution :math:`q`.
           nshots (int): the number of shots we used to run the circuit to obtain :math:`p` and :math:`q`.
           args (numpy.ndarray): other parameters which don't need to be inferred as
           array([circuit, nshots, target_prob, idle_qubits, backend, error]).
           The circuit you want to simulate; the number of shots of the simulatin; the target probability; the boolean variable idle_qubits,
           if you want the noise model to take into account idle qubits; the backend; the boolean variable error, if you want to take into account the hellinger fidelity error due to shot noise.

       Returns:
           (float): The Hellinger fidelity if error is False.
           (list): [Hellinger fidelity, Hellinger fidelity error] if error is True.
    """
    circuit = args[0]
    nshots = args[1]
    target_prob = args[2]
    idle_qubits = args[3]
    backend = args[4]
    error = args[5]
    qubits = circuit.nqubits
    parameters = np.array(parameters)

    if any(2 * parameters[0:qubits] - parameters[qubits : 2 * qubits] < 0):
        return -np.inf

    params = {
        "t1": tuple(parameters[0:qubits]),
        "t2": tuple(parameters[qubits : 2 * qubits]),
        "gate_time": tuple(parameters[2 * qubits : 2 * qubits + 2]),
        "excited_population": 0,
        "depolarizing_error": tuple(parameters[2 * qubits + 2 : 2 * qubits + 4]),
        "bitflips_error": (
            parameters[2 * qubits + 4 : 3 * qubits + 4],
            parameters[3 * qubits + 4 : 4 * qubits + 4],
        ),
        "idle_qubits": idle_qubits,
    }

    noisy_circ = noisy_circuit(circuit, params)
    freq = backend.execute_circuit(circuit=noisy_circ, nshots=nshots).frequencies()
    prob = freq_to_prob(freq)

    hellinger_fid = hellinger_fidelity(target_prob, prob)

    if error == True:
        return [hellinger_fid, hellinger_shot_error(target_prob, prob, nshots)]
    else:
        return hellinger_fid


class CompositeNoiseModel:
    """Class associated with a realistic representation of a noisy circuit modeled by the :func:`qibo.noise_model.noisy_circuit`.
    This class is able to fit the parameters of the noise model to reproduce an experimental realization of the circuit
    you want to simulate.

    Args:
        noisy_circuit (qibo.models.Circuit): the noisy circuit. See :func:`qibo.noise_model.noisy_circuit`.
        params (dictionary): the parameters of the noise model. See :func:`qibo.noise_model.noisy_circuit`.
        hellinger (float): current value of the hellinger fidelity between the noisy simulation and the given target result.
        hellinger0 (dictionary): the fidelity and the shot error fidelity  organized as {"fidelity": (float) f, "shot_error": (float) e}.
    """

    def __init__(self, params):
        self.noisy_circuit = {}
        self.params = params
        self.hellinger = {}
        self.hellinger0 = {}

    def apply(self, circuit):
        """Creates the noisy circuit from the circuit given as argument by using the :func:`qibo.noise_model.noisy_circuit`.
        Args:
            circuit (qibo.models.Circuit): the circuit you want to simulate.
        """
        self.noisy_circuit = noisy_circuit(circuit, self.params)

    def fit(
        self,
        target_result,
        bounds=True,
        f_min_rtol=None,
        backend=None,
    ):
        r"""Performes the fitting procedure of the noise model parameters, using the method nlopt.opt from the library nplot. The fitting procedure is implemented to maximize the hellinger fidelity calculated using the :func:`qibo.noise_model.loss` between the probability distribution function estimated by the noise model and the one measured experimentally. Since, we are using probability distribution functions estimated using a finite number of shots, the hellinger fidelity is going to have an error caused by an imperfect estimation of the probabilities. This method takes into account this effect and stops when the fidelity reaches a corrected maximum $1-\epsilon$, with $\epsilon$=:func:`qibo.noise_model.hellinger_shot_error`.

        Args:
            target_result (qibo.states.CircuitResult): the circuit result with frequencies you want to emulate.
            bounds: If True are given the default bounds for the depolarizing and thermal relaxation channels' parameters.
            Otherwise it's possible to pass a matrix of size (2, 4 * nqubits + 4), where bounds[0] and bounds[1]
            will be respectively the lower and the upper bounds for the parameters. The first 2 * nqubit columns are related
            to the $T_1$ and $T_2$ parameters; the subsequent 2 columns are related to the gate time parameters; the other subsequent 2 columns are related depolarizing error parameters; the last 2 * nqubit columns are related to bitflips errors.
            f_min_rtol (float): the tolerance of the optimization. The optimization will finish when the fidelity reaches the value
            $1-f_min_rtol$, by default f_min_rtol is set to be the fidelity error caused by the finite number of shots and calculated by the :func:`qibo.noise_model.hellinger_shot_error`.
            backend: you can specify your backend. If None qibo.backends.GlobalBackend is used.
        """

        from functools import partial

        import nlopt

        if backend == None:  # pragma: no cover
            from qibo.backends import GlobalBackend

            backend = GlobalBackend()

        circuit = target_result.circuit
        nshots = target_result.nshots
        target_prob = freq_to_prob(target_result.frequencies())

        idle_qubits = self.params["idle_qubits"]
        qubits = target_result.nqubits

        if bounds == True:
            qubits = target_result.nqubits
            lb = np.zeros(4 * qubits + 4)
            ub = [10000] * (2 * qubits + 2) + [4 / 3, 16 / 15] + [1] * 2 * qubits
        else:
            lb = bounds[0]
            ub = bounds[1]

        shot_error = True
        args = (circuit, nshots, target_prob, idle_qubits, backend, shot_error)
        result = -np.inf
        while result == -np.inf:
            initial_params = np.random.uniform(lb, ub)
            result = loss(initial_params, 0, args)

        if f_min_rtol == None:
            f_min_rtol = result[1]

        args = list(args)
        args[5] = False
        args = tuple(args)

        self.hellinger0 = {"fidelity": abs(result[0]), "shot_error": result[1]}

        opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND, len(initial_params))
        f = partial(loss, args=args)
        opt.set_max_objective(f)
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_stopval(1 - f_min_rtol)
        xopt = opt.optimize(list(initial_params))
        maxf = opt.last_optimum_value()
        result = opt.last_optimize_result()

        parameters = xopt
        params = {
            "t1": tuple(parameters[0:qubits]),
            "t2": tuple(parameters[qubits : 2 * qubits]),
            "gate_time": tuple(parameters[2 * qubits : 2 * qubits + 2]),
            "excited_population": 0,
            "depolarizing_error": tuple(parameters[2 * qubits + 2 : 2 * qubits + 4]),
            "bitflips_error": (
                parameters[2 * qubits + 4 : 3 * qubits + 4],
                parameters[3 * qubits + 4 : 4 * qubits + 4],
            ),
            "idle_qubits": idle_qubits,
        }
        self.hellinger = maxf
        self.params = params
        self.extra = {
            "message": result,
        }
