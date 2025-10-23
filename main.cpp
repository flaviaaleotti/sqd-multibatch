/*
# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
*/

#include <algorithm>
#include <bitset>
#include <cassert>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <unordered_map>
#include <chrono>

#include "boost/dynamic_bitset.hpp"
#include "ffsim/ucj.hpp"
#include "ffsim/ucjop_spinbalanced.hpp"
#include "load_parameters.hpp"
#include "qiskit/addon/sqd/configuration_recovery.hpp"
#include "qiskit/addon/sqd/postselection.hpp"
#include "qiskit/addon/sqd/subsampling.hpp"
#include "sbd_helper.hpp"
#include "sqd_helper.hpp"

#include "circuit/quantumcircuit.hpp"
#include "compiler/transpiler.hpp"
#include "primitives/backend_sampler_v2.hpp"
#include "service/qiskit_runtime_service.hpp"

using namespace Qiskit::circuit;
using namespace Qiskit::providers;
using namespace Qiskit::primitives;
using namespace Qiskit::service;
using namespace Qiskit::compiler;

using Sampler = BackendSamplerV2;

// Test stub: generate num_samples random bitstrings of length num_bits
// with Bernoulli(p=0.5) and aggregate into counts (bitstring -> occurrences).
// Use this when a real backend/simulator is unavailable (debugging).
std::unordered_map<std::string, uint64_t>
generate_counts_uniform(int num_samples, int num_bits,
                        std::optional<unsigned int> seed = std::nullopt)
{
    std::mt19937 rng(seed.value_or(std::random_device{}()));
    std::bernoulli_distribution dist(0.5);

    std::unordered_map<std::string, uint64_t> counts;

    for (int i = 0; i < num_samples; ++i)
    {
        std::string bitstring;
        bitstring.reserve(num_bits);
        for (int j = 0; j < num_bits; ++j)
        {
            bitstring += dist(rng) ? '1' : '0';
        }
        counts[bitstring]++;
    }
    return counts;
}

// Convert an array of boost::dynamic_bitset<> to string-based BitString objects.
static auto
bitsets_to_bitstrings(const std::vector<boost::dynamic_bitset<>>& bitsets) -> std::vector<BitString>
{
    std::vector<BitString> bitstrings;
    bitstrings.reserve(bitsets.size());
    std::string str;
    for (const auto& bitset : bitsets)
    {
        boost::to_string(bitset, str);
        bitstrings.emplace_back(str);
    }
    return bitstrings;
}
// Convert string-based BitString objects back to boost::dynamic_bitset<>.
// Internal representation for efficient bitwise operations in recovery/post-selection.
static auto bitsets_from_bitstrings(const std::vector<BitString>& bitstrings)
    -> std::vector<boost::dynamic_bitset<>>
{
    std::vector<boost::dynamic_bitset<>> bitsets;
    bitsets.reserve(bitsets.size());
    for (const auto& bitstring : bitstrings)
    {
        bitsets.emplace_back(bitstring.to_string());
    }
    return bitsets;
}

// Load initial alpha/beta occupancies from a JSON file.
// Format: { "init_occupancies": [ alpha..., beta... ] }  (even length)
// After splitting, reverse each to match the internal right-to-left convention.
std::array<std::vector<double>, 2> load_initial_occupancies(const std::string& filename)
{
    std::ifstream i(filename);
    nlohmann::json input;
    i >> input;

    // Validate input JSON: throw on missing key to fail fast on user error.
    if (!input.contains("init_occupancies"))
        throw std::invalid_argument("no init_params in initial parameter json: file=" + filename);
    std::vector<double> init_occupancy = input["init_occupancies"];

    if ((init_occupancy.size() & 1) != 0)
    {
        throw std::runtime_error("Initial occupancies list must have even number of elements");
    }
    const auto half_size = init_occupancy.size() / 2;

    std::vector<double> alpha_occupancy(init_occupancy.begin(), init_occupancy.begin() + half_size);
    std::vector<double> beta_occupancy(init_occupancy.begin() + half_size, init_occupancy.end());

    std::reverse(alpha_occupancy.begin(), alpha_occupancy.end());
    std::reverse(beta_occupancy.begin(), beta_occupancy.end());

    return {alpha_occupancy, beta_occupancy};
}

// Utility: normalize counts (occurrences) into probabilities in [0,1].
// Empty input returns an empty map (no exception).
std::unordered_map<std::string, double>
_normalize_counts_dict(const std::unordered_map<std::string, uint64_t>& counts)
{
    // Check if the input map is empty
    if (counts.empty())
    {
        return {}; // Return an empty map
    }

    // Calculate the total counts
    uint64_t total_counts =
        std::accumulate(counts.begin(), counts.end(), 0,
                        [](uint64_t sum, const auto& pair) { return sum + pair.second; });

    // Create a new map with normalized values
    std::unordered_map<std::string, double> probabilities;
    for (const auto& [key, value] : counts)
    {
        probabilities[key] = static_cast<double>(value) / total_counts;
    }

    return probabilities;
}

// Transform counts (bitstring -> count) into parallel arrays (bitstrings,
// probabilities).
std::pair<std::vector<BitString>, std::vector<double>>
counts_to_arrays(const std::unordered_map<std::string, uint64_t>& counts)
{
    std::vector<BitString> bs_mat;
    std::vector<double> freq_arr;

    if (counts.empty())
        return {bs_mat, freq_arr};

    // Normalize the counts to probabilities
    auto prob_dict = _normalize_counts_dict(counts);

    // Convert bitstrings to a 2D boolean matrix
    for (const auto& [bitstring, _] : prob_dict)
    {
        bs_mat.push_back(BitString(bitstring));
    }

    // Convert probabilities to a 1D array
    for (const auto& [_, probability] : prob_dict)
    {
        freq_arr.push_back(probability);
    }

    return {bs_mat, freq_arr};
}

using namespace Eigen;
using namespace ffsim;

int main(int argc, char* argv[])
{
    // Start time
    auto start = std::chrono::high_resolution_clock::now();   // start timer
    
    // ===== MPI initialization =====
    // This workflow assumes MPI. Request FUNNELED (only main thread calls MPI).
    int provided;
    int mpi_init_error = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (mpi_init_error != MPI_SUCCESS)
    {
        char err_msg[1024];
        int err_msg_len;
        MPI_Error_string(mpi_init_error, err_msg, &err_msg_len);
        // On failure, print a readable error and exit immediately (cluster-friendly
        // diagnostics).
        std::cerr << "MPI_Init failed: " << err_msg << std::endl;
        return mpi_init_error;
    }
    
    double start_intro = MPI_Wtime();

    // ===== SBD (diagonalization sub-workflow) configuration =====
    // Build SBD parameters from CLI args. Used by sbd_main for energy evaluation, etc.
    SBD diag_data = generate_sbd_data(argc, argv);

    // ===== SQD (sampling/recovery sub-workflow) configuration =====
    // Holds run_id, backend, shot count, and other metadata for sampling and recovery.
    SQD sqd_data = generate_sqd_data(argc, argv);
    
    // ===== MPI communicator setup =====
    // global communicator
    sqd_data.comm = MPI_COMM_WORLD;
    MPI_Comm_rank(sqd_data.comm, &sqd_data.mpi_rank);
    MPI_Comm_size(sqd_data.comm, &sqd_data.mpi_size);

    // sub-communicators for ranks on same node
    MPI_Comm node_comm;
    MPI_Comm_split_type(sqd_data.comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    int local_rank;
    MPI_Comm_rank(node_comm, &local_rank);

    // create a communicator for rank 0 on each node
    int is_node_leader = (local_rank == 0);
    int leaders_rank = -1;
    MPI_Comm leaders_comm;
    MPI_Comm_split(MPI_COMM_WORLD, is_node_leader ? 0 : MPI_UNDEFINED, sqd_data.mpi_rank, &leaders_comm);
    if (is_node_leader) {
        MPI_Comm_rank(leaders_comm, &leaders_rank);
    }

    // Get number of nodes from size of leaders_comm
    int num_nodes = 0;
    if (is_node_leader) {
        MPI_Comm_size(leaders_comm, &num_nodes);
    }

    // Broadcast num_nodes to everyone (also non-leaders)
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::cout << "Rank " << sqd_data.mpi_rank << " / " << sqd_data.mpi_size
              << " is on node rank " << local_rank
              << ", total nodes = " << num_nodes << std::endl;
    
    // ORIGINAL CODE
    int message_size = sqd_data.run_id.size();
    // Send the integer message_size from rank 0 to all others.
    // Others resize their run_id buffer to receive the content.
    MPI_Bcast(&message_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (sqd_data.mpi_rank != 0)
    {
        sqd_data.run_id.resize(message_size);
    }
    // Broadcast the textual run_id to all ranks to keep logs/artifacts consistent.
    // Rank 0 sends the size, others resize buffer, then receive content.
    MPI_Bcast(sqd_data.run_id.data(), message_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Batch sizing for SBD input (alpha-determinant groups).
    uint64_t samples_per_batch = sqd_data.samples_per_batch;
    uint64_t n_batches = sqd_data.n_batches;
    
    // ===== Read initial parameters (norb, nelec, params for lucj) from JSON =====
    const std::string input_file_path = "../data/parameters_fe4s4.json";
    double tol = 1e-8;
    uint64_t norb;
    size_t n_reps = 1;
    std::pair<uint64_t, uint64_t> nelec;
    std::vector<std::pair<uint64_t, uint64_t>> interaction_aa;
    std::vector<std::pair<uint64_t, uint64_t>> interaction_ab;
    std::vector<double> init_params;

    // Centralize I/O on rank 0. Abort the whole job on input failure.
    if (sqd_data.mpi_rank == 0)
    {
        try
        {
            load_initial_parameters(input_file_path, norb, nelec, interaction_aa, interaction_ab,
                                    init_params);
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error loading initial parameters: " << e.what() << std::endl;
            MPI_Abort(sqd_data.comm, 1);
            return 1;
        }
        //log(sqd_data,
        //    {"initial parameters are loaded. param_length=", std::to_string(init_params.size())});
    }

    auto num_elec_a = nelec.first;
    auto num_elec_b = nelec.second;
    // Broadcast norb num_elec to all ranks (these quantities will be used by all ranks later)
    MPI_Bcast(&norb, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_elec_a, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_elec_b, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    double end_intro = MPI_Wtime();
    if (sqd_data.mpi_rank == 0)
    {
        std::cout << "TIME: setup " << (end_intro - start_intro) << " seconds" << std::endl;
    }
    
    double start_quantum = MPI_Wtime();
    // ===== Sampling (circuit execution) =====
    
    // Measurement results: (bitstring -> counts). Produced on rank 0, then array-ified
    // later.
    std::unordered_map<std::string, uint64_t> counts;
    // Rank 0 performs sampling (real or mock), others wait to receive results.
    if (sqd_data.mpi_rank == 0)
    {
// ===== Sampling mode switch =====
// a) Mock: generate_counts_uniform (debugging)
// b) Real: build LUCJ circuit -> transpile -> run on backend with Sampler -> get counts
#if USE_RANDOM_SHOTS != 0
        counts = generate_counts_uniform(sqd_data.num_shots, 2 * norb, 1234);
        // counts is a key:value map (bitstring -> occurrences) e.g. {"0101", 23}
#else
        //////////////// LUCJ Circuit Generation ////////////////
        size_t params_size = init_params.size();

        Eigen::VectorXcd params(params_size);
        for (size_t i = 0; i < params_size; ++i)
        {
            params(i) = init_params[i];
        }
        std::optional<MatrixXcd> t1 = std::nullopt;
        // 'interaction_pairs' allows passing (alpha-alpha, alpha-beta/beta-beta)
        // coupling patterns.
        std::array<std::optional<std::vector<std::pair<uint64_t, uint64_t>>>, 2> interaction_pairs =
            {std::make_optional<std::vector<std::pair<uint64_t, uint64_t>>>(interaction_aa),
             std::make_optional<std::vector<std::pair<uint64_t, uint64_t>>>(interaction_ab)};

        // Construct the spin-balanced UCJ operator from parameter vector.

        UCJOpSpinBalanced ucj_op =
            UCJOpSpinBalanced::from_parameters(params, norb, n_reps, interaction_pairs, true);
        std::vector<uint32_t> qubits(2 * norb);
        std::iota(qubits.begin(), qubits.end(), 0);
        auto instructions = hf_and_ucj_op_spin_balanced_jw(qubits, nelec, ucj_op);

        // Quantum circuit with Qiskit C++
        auto qr = QuantumRegister(2 * norb);   // quantum registers
        auto cr = ClassicalRegister(2 * norb); // classical registers
        auto circ = QuantumCircuit(qr, cr);    // create a quantum circuits with registers

        // add gates from instruction list from hf_and_ucj_op_spin_balanced_jw
        //   for demo: calling Qiskit C++ circuit functions to make quantum circuit
        for (const auto& instr : instructions)
        {
            if (std::string("x") == instr.gate)
            {
                // X gate
                circ.x(instr.qubits[0]);
            }
            else if (std::string("rz") == instr.gate)
            {
                // RZ gate
                circ.rz(instr.params[0], instr.qubits[0]);
            }
            else if (std::string("cp") == instr.gate)
            {
                // controlled phase gate
                circ.cp(instr.params[0], instr.qubits[0], instr.qubits[1]);
            }
            else if (std::string("xx_plus_yy") == instr.gate)
            {
                // XX_plus_YY gate
                circ.xx_plus_yy(instr.params[0], instr.params[1], instr.qubits[0], instr.qubits[1]);
            }
        }
        // this is smarter way using standard gate mapping to convert gate name to op
        // auto map = get_standard_gate_name_mapping();
        // for (const auto &instr : instructions) {
        //    auto op = map[instr.gate];
        //    if (instr.params.size() > 0)
        //         op.set_params(instr.params);
        //    circ.append(op, instr.qubits);
        // }

        // sampling all the qubits
        for (size_t i = 0; i < circ.num_qubits(); ++i)
        {
            circ.measure(i, i);
        }

        // get backend from Quantum Runtime Service
        // set 2 environment variables before executing
        // QISKIT_IBM_TOKEN = "your API key"
        // QISKIT_IBM_INSTANCE = "your CRN"
        std::string backend_name = sqd_data.backend_name;
        auto service = QiskitRuntimeService();
        auto backend = service.backend(backend_name);

        // Transpile a quantum circuit for the target backend.
        auto transpiled = transpile(circ, backend);

        uint64_t num_shots = sqd_data.num_shots;

        // Configure the Sampler execution (num_shots from SQD configuration).
        auto sampler = Sampler(backend, num_shots);

        auto job = sampler.run({SamplerPub(transpiled)});
        if (job == nullptr)
            return -1;
        auto result = job->result();
        auto pub_result = result[0];

        // Extract classical counts from the execution result.
        // These form the classical distribution for downstream recovery/selection.
        counts = pub_result.data().get_counts();
#endif // USE_RANDOM_SHOTS
    } 
    
    // ===== Broadcast sampling results (counts) to all ranks =====
    // (1) compute and share number of elements in counts
    int n_counts = counts.size();
    MPI_Bcast(&n_counts, 1, MPI_INT, 0, sqd_data.comm);
    // (2) compute length of single bitstring and total amount of characters to broadcast
    int bitstring_length = 2 * norb; // length of each bitstring (same for all)
    int total_chars = n_counts * bitstring_length;

    // (3) flatten keys and values of "counts" on rank 0 (for broadcast)
    std::vector<char> flat_keys(total_chars);
    std::vector<uint64_t> flat_values(n_counts);
    if (sqd_data.mpi_rank == 0) 
    {
        int offset = 0;
        int idx = 0;
        for (const auto& [key, val] : counts) 
        {
            std::copy(key.begin(), key.end(), flat_keys.begin() + offset);
            offset += key.size();
            flat_values[idx++] = val;
        }
    }
    // (4) Broadcast flattened keys and values to all ranks
    MPI_Bcast(flat_keys.data(), total_chars, MPI_CHAR, 0, sqd_data.comm);
    MPI_Bcast(flat_values.data(), n_counts, MPI_UINT64_T, 0, sqd_data.comm);
    
    // (5) On non-root ranks, reconstruct counts map from received flat arrays
    if (sqd_data.mpi_rank != 0) 
    {
        counts.clear();
        int offset = 0;
        for (int i = 0; i < n_counts; ++i) 
        {
            std::string key(flat_keys.data() + offset, bitstring_length);
            uint64_t val = flat_values[i];
            counts.emplace(key, val);
            offset += bitstring_length;
        }
    }

    // Expand counts (map) into (bitstrings[], probs[]).
    auto [bitstring_matrix_full_, probs_arr_full] = counts_to_arrays(counts);
    // Convert BitString objects to boost::dynamic_bitset<> for internal processing.
    auto bitstring_matrix_full = bitsets_from_bitstrings(bitstring_matrix_full_);

    double end_quantum = MPI_Wtime();
    if (sqd_data.mpi_rank == 0)
    {
        std::cout << "TIME: quantum circuit + quantum data distribution " << (end_quantum - start_quantum) << " seconds" << std::endl;
    }
    
    // =====  Configuration Recovery, Post Selection, Diagonalization ===== 

    std::vector<boost::dynamic_bitset<>> bs_mat_tmp;
    std::vector<double> probs_arr_tmp;
    std::array<std::vector<double>, 2> latest_occupancies, initial_occupancies;
    int n_recovery = sqd_data.n_recovery;

    try
    {
        // Load prior alpha/beta occupancies used as the initial distribution for
        // recovery.
        initial_occupancies = load_initial_occupancies("../data/initial_occupancies_fe4s4.json");
    }
    catch (const std::invalid_argument& e)
    {
        std::cerr << "Error loading initial occupancies: " << e.what() << std::endl;
        return 1;
    }
    
    double best_E = 0.0;
    
    // ===== Configuration recovery loop (n_recovery iterations) =====
    // Each iter: recover_configurations → postselect → subsample → SBD (diagonalize) → update occupancies.
    for (uint64_t i_recovery = 0; i_recovery < n_recovery; ++i_recovery)
    {
        
        if (sqd_data.mpi_rank == 0)
        {
            log(sqd_data, {"\n=== Recovery iteration ", std::to_string(i_recovery), " ==="});
        }

        // Iteration 0: feed full bitstring/probability sets and seed recovery from
        // initial occupancies.
        if (i_recovery == 0)
        {
            bs_mat_tmp = bitstring_matrix_full;
            probs_arr_tmp = probs_arr_full;
            latest_occupancies = initial_occupancies;
        }

        // ===== RECOVERY & POST-SELECTION of BITSTRINGS and PROBABILITIES =====

        // Recover physically consistent configurations from observed probabilities + prior occupancies.
        // Random generator:
        //  - rc_rng : used for configuration recovery randomness (derived from rng).
        std::mt19937 rc_rng(sqd_data.mpi_rank * sqd_data.mpi_rank + 5678); // different seed per MPI rank

        // (split across MPI ranks for scalability)
        double start_recovery = MPI_Wtime();
        size_t N = bitstring_matrix_full.size(); // total number of bitstrings sampled from qpu
        size_t chunk = N / sqd_data.mpi_size;
        size_t remainder = N % sqd_data.mpi_size;
        // each rank gets a chunk, ranks whose ID is lower than remainder get an extra item
        size_t start = sqd_data.mpi_rank * chunk + std::min<size_t>(sqd_data.mpi_rank, remainder);
        size_t end   = start + chunk + (sqd_data.mpi_rank < remainder ? 1 : 0);

        // local slices of bitstrings and probabilities for this rank
        std::vector<boost::dynamic_bitset<>> local_bitstrings(bitstring_matrix_full.begin() + start, bitstring_matrix_full.begin() + end);
        std::vector<double> local_probs(probs_arr_full.begin() + start, probs_arr_full.begin() + end);

        auto recovered = Qiskit::addon::sqd::recover_configurations(
            local_bitstrings, local_probs, latest_occupancies, {num_elec_a, num_elec_b},
            rc_rng);
        bs_mat_tmp = std::move(recovered.first); // take recovered bitstrings from first element of "recovered"
        probs_arr_tmp = std::move(recovered.second); // take recovered probabilities from second element of "recovered"

        // Post-selection: accept bitstrings whose left/right (alpha/beta) Hamming
        // weights match target electron counts.
        auto result =
            Qiskit::addon::sqd::postselect_bitstrings(
                bs_mat_tmp, probs_arr_tmp,
                Qiskit::addon::sqd::MatchesRightLeftHamming(num_elec_a, num_elec_b));
            
        auto &local_postselected_bitstrings = result.first; // take postselected bitstrings from first element of "result"
        auto &local_postselected_probs      = result.second; // take postselected probabilities from second element of "result"
        int local_N = local_postselected_bitstrings.size();

        // Gather postselected bitstrings and probs from all ranks:
        // (1) convert local bitstrings to char and store them in flat array
        std::vector<char> local_flat_keys(local_N * bitstring_length);  
        for (int i = 0; i < local_N; ++i)
        {
            std::string s;
            boost::to_string(local_postselected_bitstrings[i], s);  // convert to "0101..."
            assert((int)s.size() == bitstring_length);
            memcpy(&local_flat_keys[i * bitstring_length], s.data(), bitstring_length);
        }

        // (2) Probabilities
        // (2.a) Make each rank aware of how much data to expect from other ranks (local_N sizes)
        // (MPI_Allgather: Gathers data from all processes and distributes it to all processes)
        std::vector<int> recv_counts(sqd_data.mpi_size); // store sizes from all ranks in recv_counts array
        MPI_Allgather(&local_N, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, sqd_data.comm);
        
        // (2.b) compute and store the start indices (displacements) for each rank's probability data in the global arrays
        std::vector<int> displs(sqd_data.mpi_size, 0); // starting offset (index) in the global array where rank i’s data should start to be placed
        for (int i = 1; i < sqd_data.mpi_size; ++i)
        {
            displs[i] = displs[i-1] + recv_counts[i-1];
        }
        // (2.c) compute total number of items (postselected bitstrings/probabilities) across all ranks
        // we could sum over elements of recv_counts, but it is more efficient to use the last element of displs
        // (which is already the sum of all elements of recv_counts except the last one) and simply add the last element of recv_counts
        int total_postselect_bitstr = displs.back() + recv_counts.back(); 

        // (2.d) allocate global array to receive all data
        std::vector<double> postselected_probs(total_postselect_bitstr);

        // (2.e) gather all probabilities into postselected_probs
        // (MPI_Allgatherv: Gathers data from all processes and delivers it to all. Each process may contribute a different amount of data)
        MPI_Allgatherv(local_postselected_probs.data(), local_N, MPI_DOUBLE,
                       postselected_probs.data(), recv_counts.data(),
                       displs.data(), MPI_DOUBLE, sqd_data.comm);

        // (3) Bitstrings
        // (3.a) compute and store local amounts of data and displacements also for bitstrings (chars)
        std::vector<int> recv_counts_keys(sqd_data.mpi_size);
        std::vector<int> displs_keys(sqd_data.mpi_size);
        for (int i = 0; i < sqd_data.mpi_size; ++i) 
        {
            recv_counts_keys[i] = recv_counts[i] * bitstring_length;
            displs_keys[i] = displs[i] * bitstring_length;
        }

        // (3.b) allocate global (flat) array to receive all bitstrings
        std::vector<char> global_flat_keys(total_postselect_bitstr * bitstring_length);

        // (3.c) gather all bitstrings into global_flat_keys
        MPI_Allgatherv(local_flat_keys.data(), local_N * bitstring_length, MPI_CHAR,
                       global_flat_keys.data(), recv_counts_keys.data(),
                       displs_keys.data(), MPI_CHAR, sqd_data.comm);

        // (4) Reconstruct postselected bitstrings in correct format from global_flat_keys
        std::vector<boost::dynamic_bitset<>> postselected_bitstrings(total_postselect_bitstr);
        for (int i = 0; i < total_postselect_bitstr; ++i) 
        {
            std::string s(&global_flat_keys[i * bitstring_length], bitstring_length);
            postselected_bitstrings[i] = boost::dynamic_bitset<>(s);
        }

        double end_recovery = MPI_Wtime();
        if (sqd_data.mpi_rank == 0)
        {
            std::cout << "TIME: recovery + post-selection " << (end_recovery - start_recovery) << " seconds" << std::endl;
            //std::cout << "Number of postselected bitstrings (this iteration): " << postselected_bitstrings.size() << std::endl;
        }

        /* OLD VERSION, DOING RECOVERY AND POSTSELECTION ONLY ON RANK 0 THEN BROADCAST
        // flat version of bitstrings for broadcasting (needed because we cannot broadcast vector<string>)
        std::vector<char> flat_bitstrings;
        // Initialize vector of post-selected bitstrings in string format (populated in rank 0, then broadcast to all ranks).
        // because we cannot directly broadcast vector<dynamic_bitset>
        std::vector<std::string> postselected_bitstrings_str;
        // number of postselected bitstrings (after recovery + postselection)
        // this is needed for broadcasting to know the amount of data to send/recv
        int n_postselected = 0;
        
        // RECOVERY & POST-SELECTION of BITSTRINGS and PROBABILITIES
        // (only on rank 0, then broadcast to all ranks)
        if (sqd_data.mpi_rank == 0)
        {
            // Recover physically consistent configurations from observed probabilities
            // + prior occupancies.
            auto recovered = Qiskit::addon::sqd::recover_configurations(
                bitstring_matrix_full, probs_arr_full, latest_occupancies, {num_elec_a, num_elec_b},
                rc_rng);
            bs_mat_tmp = std::move(recovered.first);
            probs_arr_tmp = std::move(recovered.second);
            
            // Post-selection: accept bitstrings whose left/right (alpha/beta) Hamming
            // weights match target electron counts.
            auto result =
                Qiskit::addon::sqd::postselect_bitstrings(
                    bs_mat_tmp, probs_arr_tmp,
                    Qiskit::addon::sqd::MatchesRightLeftHamming(num_elec_a, num_elec_b));
            
            postselected_bitstrings = std::move(result.first);
            postselected_probs      = std::move(result.second);

            // now save the number of postselected bitstrings in n_postselected
            n_postselected = postselected_bitstrings.size();
            log(sqd_data, {"Number of postselected bitstrings (from rank 0): ", std::to_string(n_postselected)});

            // convert postselected bitstrings to flat string format for broadcasting
            std::string tmp;
            // clear is needed for recovery iterations > 0 (to avoid appending to previous data)
            flat_bitstrings.clear();
            flat_bitstrings.reserve(n_postselected * bitstring_length);
            // following loop iterates over all elements of postselected_bitstrings
            for (const auto& bs : postselected_bitstrings) 
            {
                // convert to string
                boost::to_string(bs, tmp);
                // add to flat_bitstrings
                flat_bitstrings.insert(flat_bitstrings.end(), tmp.begin(), tmp.end());
            }
        }

        // Broadcast the number of postselected bitstrings and bitstring length to all ranks
        // (so they know how much data to expect)
        MPI_Bcast(&n_postselected, 1, MPI_INT, 0, sqd_data.comm);
        MPI_Bcast(&bitstring_length, 1, MPI_INT, 0, sqd_data.comm);

        // compute total number of characters to be broadcasted 
        // (it is the sum of all bitstring lengths)
        int total_chars = n_postselected * bitstring_length;
        
        // on ranks != 0, resize flat_bitstrings to receive the data
        if (sqd_data.mpi_rank != 0) 
        {
            flat_bitstrings.resize(total_chars);
        }
        // Broadcast the bitstrings in flat format
        MPI_Bcast(flat_bitstrings.data(), total_chars, MPI_CHAR, 0, sqd_data.comm);

        // Broadcast the probabilities
        // (easier, just a single array of doubles)
        // resize postselected_probs to receive the data (this will have no effect on rank 0)
        postselected_probs.resize(n_postselected);

        MPI_Bcast(postselected_probs.data(), n_postselected, MPI_DOUBLE, 0, sqd_data.comm);
        
        // On non-root ranks, reconstruct the bitstrings
        if (sqd_data.mpi_rank != 0) 
        {
            // received data was sent to flat_bitstrings but we need to put it back to postselected_bitstrings
            // make sure it's empty before filling (needed for recovery iterations > 0)
            postselected_bitstrings.clear();   
            
            // offset will help us to know the position in flat_bitstrings where each bitstring starts
            int offset = 0;
            std::string tmp;
            for (int i = 0; i < n_postselected; ++i) 
            {
                tmp = std::string(flat_bitstrings.data() + offset, bitstring_length);
                postselected_bitstrings.emplace_back(tmp);
                offset += bitstring_length;
            }

            assert(offset == total_chars); // sanity check
        }
        */ // END OLD VERSION

        // ===== BATCHES SPLIT ACROSS MPI NODES =====
        double start_sampling = MPI_Wtime();
        // n_node_batches[i] = number of batches assigned to node i
        // each node initially recieves N_batches/num_nodes batches
        std::vector<int> n_node_batches(num_nodes, n_batches / num_nodes);
        // then, the reminder of N_batches/num_nodes is assigned to the first nodes
        for (int r = 0; r < n_batches % num_nodes; ++r) 
        {
            n_node_batches[r] += 1; 
        }
        // ask only node leaders to print the batch assignment
        if (is_node_leader) 
        {
            log(sqd_data, {"Node ", std::to_string(leaders_rank), 
                           " assigned ", std::to_string(n_node_batches[leaders_rank]), " batches."});
        }
        
        // now syncronize all ranks on same node using node_comm:
        // each rank gets the leaders_rank stored in node_id variable, to be used as index for n_node_batches
        int node_id = leaders_rank;
        MPI_Bcast(&node_id, 1, MPI_INT, 0, node_comm); 

        // initialize vector of batches (size: n_node_batches[rank] each batch is a vector of dynamic_bitset) 
        std::vector<std::vector<boost::dynamic_bitset<>>> batches(n_node_batches[node_id]);

        // Random generator:
        //  - rng    : used for sampling/subsampling (fixed seed for reproducibility).
        std::mt19937 rng(node_id * node_id + 1234); // different seed per MPI node rank
        
        // Subsample batches of bitstrings for SBD input.
        // Each rank populates its own batches vector of size n_node_batches[rank]
        Qiskit::addon::sqd::subsample_multiple_batches(batches, postselected_bitstrings, postselected_probs,
                                      samples_per_batch, n_node_batches[node_id], rng);
        // old version, hardcoded for single batch
        //Qiskit::addon::sqd::subsample(batch, postselected_bitstrings, postselected_probs,
        //                              samples_per_batch, rng);

        double end_sampling = MPI_Wtime();
        if (is_node_leader)
        {
            std::cout << "TIME: batch sampling " << (end_sampling - start_sampling) << " seconds" << std::endl;
        }
        
        // variables to store node-specific energies and occupations
        // (vectors of one element per processed batch)
        std::vector<double> local_energies;
        std::vector<std::vector<double>> local_occs;

        double start_diag;
        double end_diag;

        // iterate over batches assigned to this node
        for (size_t b = 0; b < batches.size(); ++b)
        {
            start_diag = MPI_Wtime();
            // node leader writes alpha-determinants file for this batch
            if (is_node_leader) 
            {
                diag_data.adetfile =
                    write_alphadets_file(sqd_data, norb, num_elec_a, bitsets_to_bitstrings(batches[b]),
                                         sqd_data.samples_per_batch * 2, i_recovery, node_id);
            }

            // node leader broadcasts the adetfile path to all ranks on same node
            int path_length = 0;
            if (is_node_leader) 
            {
                path_length = diag_data.adetfile.size();
            }
            MPI_Bcast(&path_length, 1, MPI_INT, 0, node_comm);
            if (!is_node_leader) 
            {
                diag_data.adetfile.resize(path_length);
            }
            MPI_Bcast(diag_data.adetfile.data(), path_length, MPI_CHAR, 0, node_comm);  

            // Run SBD for this batch
            auto [energy_sci, occs_batch] = sbd_main(node_comm, diag_data);

            // Save results for this batch
            local_energies.push_back(energy_sci);
            local_occs.push_back(occs_batch);

            end_diag = MPI_Wtime();
        
            // log results 
            if (is_node_leader) 
            {
            log(sqd_data, {"Node ", std::to_string(leaders_rank), 
                           ", batch ", std::to_string(b),
                           ", energy: ", std::to_string(energy_sci),
                           " (iteration ", std::to_string(i_recovery), ")"});
            std::cout << "TIME: diagonalization " << (end_diag - start_diag) << " seconds" << std::endl;
            }
        }

        // ===== COLLECT AND POST-PROCESS RESULTS =====

        double start_postprocess = MPI_Wtime();
        
        // (1) compute sum of occupations across all batches on this node (stored in node_sum_occ)
        int occ_size = local_occs.empty() ? 0 : static_cast<int>(local_occs[0].size());   
        std::vector<double> node_sum_occ(occ_size, 0.0);
        for (auto &occ : local_occs)
            for (size_t i = 0; i < occ_size; ++i)
                node_sum_occ[i] += occ[i]; 
        
        // (2) reduce (+) across all nodes (using leaders_comm)
        // (the total sum is stored in avg_occs on node leader 0, then divided by n_batches to get average)
        std::vector<double> avg_occs(occ_size, 0.0);
        if (is_node_leader) 
        {
            // reduce
            MPI_Reduce(node_sum_occ.data(), avg_occs.data(), occ_size, MPI_DOUBLE, MPI_SUM, 0, leaders_comm);
            // Compute average occupations by dividing by total number of batches
            for (size_t i = 0; i < occ_size; ++i)
                avg_occs[i] /= n_batches;
            // Convert interleaved [alpha0, beta0, alpha1, beta1, ...] to { alpha[], beta[] }. 
            // NOTE: assert ensures avg_occs size matches 2 * alpha.size().
            assert(2 * latest_occupancies[0].size() == avg_occs.size());
            for (std::size_t j = 0; j < latest_occupancies[0].size(); ++j)
            {
                latest_occupancies[0][j] = avg_occs[2 * j];     // alpha orbital
                latest_occupancies[1][j] = avg_occs[2 * j + 1]; // beta orbital
            }
        }

        // (3) Broadcast latest occupations to all ranks for the next recovery iteration
        MPI_Bcast(latest_occupancies[0].data(), latest_occupancies[0].size(), MPI_DOUBLE, 0, sqd_data.comm);
        MPI_Bcast(latest_occupancies[1].data(), latest_occupancies[1].size(), MPI_DOUBLE, 0, sqd_data.comm);

        // (4) Each node (through leader) sends its minimum energy to rank 0
        // global minimum is automatically saved in best_E on rank 0 through MPI_Reduce using MPI_MIN as operation
        double node_best_E = std::numeric_limits<double>::max();;
        for (const auto& e : local_energies)
        {
            node_best_E = std::min(node_best_E, e);
        }
        if (is_node_leader)
        {
            MPI_Reduce(&node_best_E, &best_E, 1, MPI_DOUBLE, MPI_MIN, 0, leaders_comm); 
        }

        double end_postprocess = MPI_Wtime();
        if (is_node_leader)
        {
            std::cout << "TIME: collect results and postprocess " << (end_postprocess - start_postprocess) << " seconds" << std::endl;
        }
    }

    // End time and report elapsed time
    auto end = std::chrono::high_resolution_clock::now();   // end timer
    std::chrono::duration<double> elapsed = end - start;
    if (sqd_data.mpi_rank == 0)
    {
        std::cout << "Best energy: " << best_E << std::endl;
        std::cout << "\nTotal SQD-HPC elapsed time: " << elapsed.count() << " seconds" << std::endl;
    }

    // Synchronize and tear down MPI. No MPI calls are allowed beyond this point.
    if (leaders_comm != MPI_COMM_NULL) 
    {
        MPI_Comm_free(&leaders_comm);
    }
    MPI_Comm_free(&node_comm);
    MPI_Finalize();

    return 0;
        
}
