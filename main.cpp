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

    // ===== SBD (diagonalization sub-workflow) configuration =====
    // Build SBD parameters from CLI args. Used by sbd_main for energy evaluation, etc.
    SBD diag_data = generate_sbd_data(argc, argv);

    // ===== SQD (sampling/recovery sub-workflow) configuration =====
    // Holds run_id, backend, shot count, and other metadata for sampling and recovery.
    SQD sqd_data = generate_sqd_data(argc, argv);
    sqd_data.comm = MPI_COMM_WORLD;
    MPI_Comm_rank(sqd_data.comm, &sqd_data.mpi_rank);
    MPI_Comm_size(sqd_data.comm, &sqd_data.mpi_size);
    //std::cout<<"MPI rank/size: " << sqd_data.mpi_rank << "/" << sqd_data.mpi_size << std::endl;
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

    // Random generators:
    //  - rng    : used for sampling/subsampling (fixed seed for reproducibility).
    //  - rc_rng : used for configuration recovery randomness (derived from rng).
    std::mt19937 rng(sqd_data.mpi_rank * sqd_data.mpi_rank + 1234); // different seed per MPI rank
    std::mt19937 rc_rng(rng());

    // Batch sizing for SBD input (alpha-determinant groups).
    uint64_t samples_per_batch = sqd_data.samples_per_batch;
    uint64_t n_batches = sqd_data.n_batches;

    // Read initial parameters (norb, nelec, params for lucj) from JSON.
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
        log(sqd_data,
            {"initial parameters are loaded. param_length=", std::to_string(init_params.size())});
    }
    int node_per_member = sqd_data.mpi_size;

    // Measurement results: (bitstring -> counts). Produced on rank 0, then array-ified
    // later.
    std::unordered_map<std::string, uint64_t> counts;

    auto num_elec_a = nelec.first;
    auto num_elec_b = nelec.second;
    if (sqd_data.mpi_rank == 0)
    {
// ===== Sampling mode switch =====
// a) Mock: generate_counts_uniform (debugging)
// b) Real: build LUCJ circuit -> transpile -> run on backend with Sampler -> get counts
#if USE_RANDOM_SHOTS != 0
        counts = generate_counts_uniform(sqd_data.num_shots, 2 * norb, 1234);
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

    ////// Configuration Recovery, Post Selection, Diagonalization //////

    // Expand counts (map) into (bitstrings[], probs[]).
    auto [bitstring_matrix_full_, probs_arr_full] = counts_to_arrays(counts);
    auto bitstring_matrix_full = bitsets_from_bitstrings(bitstring_matrix_full_);

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
    // ===== Configuration recovery loop (n_recovery iterations) =====
    // Each iter: recover_configurations → postselect → subsample → SBD (diagonalize) →
    // update occupancies.
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

        // ===== BATCHES SPLIT ACROSS MPI RANKS =====
        // new declaration (Flavia): each rank has its own batches
        // n_rank_batches[i] = number of batches assigned to rank i
        // each batch initially recieves N_batches/N_ranks batches
        std::vector<int> n_rank_batches(sqd_data.mpi_size, n_batches / sqd_data.mpi_size);
        // then, the reminder of N_batches/N_ranks is assigned to the first ranks
        for (int r = 0; r < n_batches % sqd_data.mpi_size; ++r) 
        {
            n_rank_batches[r] += 1; 
        }
        log(sqd_data, {"Rank ", std::to_string(sqd_data.mpi_rank), 
                       " processes ", std::to_string(n_rank_batches[sqd_data.mpi_rank]), " batches."});
        
        // initialize vector of batches (size: n_rank_batches[rank] each batch is a vector of dynamic_bitset) 
        std::vector<std::vector<boost::dynamic_bitset<>>> batches(n_rank_batches[sqd_data.mpi_rank]);

        // ===== BITSTRINGS & PROBABILITIES FOR SAMPLING =====
        // Initialize vector of post-selected bitstrings in string format (populated in rank 0, then broadcast to all ranks).
        // because we cannot directly broadcast vector<dynamic_bitset>
        std::vector<std::string> postselected_bitstrings_str;
        // number of postselected bitstrings (after recovery + postselection)
        // this is needed for broadcasting to know the amount of data to send/recv
        int n_postselected = 0;
        int bitstring_length = 2 * norb; // length of each bitstring (same for all)
        
        // vectors for postselected bitstrings and probabilities (will contain the full set on all ranks after broadcasting)
        std::vector<double> postselected_probs;
        std::vector<boost::dynamic_bitset<>> postselected_bitstrings;
        // flat version of bitstrings for broadcasting (needed because we cannot broadcast vector<string>)
        std::vector<char> flat_bitstrings;

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

        
        // Subsample batches of bitstrings for SBD input.
        // Each rank populates its own batches vector of size n_rank_batches[rank]
        Qiskit::addon::sqd::subsample_multiple_batches(batches, postselected_bitstrings, postselected_probs,
                                      samples_per_batch, n_rank_batches[sqd_data.mpi_rank], rng);
        // old version, hardcoded for single batch
        //Qiskit::addon::sqd::subsample(batch, postselected_bitstrings, postselected_probs,
        //                              samples_per_batch, rng);
        
        // variables to store rank-specific energies and occupations
        // (vectors of one element per processed batch)
        std::vector<double> local_energies;
        std::vector<std::vector<double>> local_occs;

        // iterate over batches assigned to this rank
        for (size_t b = 0; b < batches.size(); ++b)
        {
            
            // Write alpha-determinants file for this batch
            diag_data.adetfile =
                write_alphadets_file(sqd_data, norb, num_elec_a, bitsets_to_bitstrings(batches[b]),
                                     sqd_data.samples_per_batch * 2, i_recovery);
            
            // Run SBD for this batch
            auto [energy_sci, occs_batch] = sbd_main(sqd_data.comm, diag_data);
            //double energy_sci = -300.0;
            //std::vector<double> occs_batch(2 * norb, 1.0);

            // Save results for this batch
            local_energies.push_back(energy_sci);
            local_occs.push_back(occs_batch);
        
            // log results 
                log(sqd_data, {"Rank ", std::to_string(sqd_data.mpi_rank), 
                               ", batch ", std::to_string(b),
                               ", energy: ", std::to_string(energy_sci),
                               " (iteration ", std::to_string(i_recovery), ")"});
        }

        // variables for rank 0 to collect final energies from all batches from all ranks
        // int local_n_batches = local_energies.size();
        // recvcounts[i] = number of batches from rank i
        // displs[i] = displacement (offset) in the final array where rank i's data starts
        std::vector<int> recvcounts, displs;
        std::vector<double> all_energies;

        if (sqd_data.mpi_rank == 0) 
        {
            //recvcounts.resize(sqd_data.mpi_size);
            displs.resize(sqd_data.mpi_size);
        }
        /*
        int MPI_Gather(
            const void *sendbuf,  // Starting address of send buffer
            int sendcount,        // Number of elements to send
            MPI_Datatype sendtype,// Data type of send buffer elements
            void *recvbuf,        // Starting address of receive buffer (root only)
            int recvcount,        // Number of elements received from each process (root only)
            MPI_Datatype recvtype,// Data type of receive buffer elements (root only)
            int root,             // Rank of the root process
            MPI_Comm comm         // Communicator
        );

        */
        // collect the number of batches from each rank into recvcounts array on rank 0
        // si può usare n_rank_batches 
        /*
        MPI_Gather(&local_n_batches, 
                   1, 
                   MPI_INT, 
                   recvcounts.data(), 
                   1, 
                   MPI_INT, 
                   0, 
                   sqd_data.comm);*/

        if (sqd_data.mpi_rank == 0) 
        {
            // compute displacements based on n_rank_batches
            int total = 0;
            for (int i = 0; i < sqd_data.mpi_size; ++i) 
            {
                displs[i] = total;
                total += n_rank_batches[i];
            }
            all_energies.resize(total);
        }
        
        /*
        Gartherv allows to send a variable amount of data from each rank.
        int MPI_Gatherv(
            const void *sendbuf,    // Starting address of send buffer
            int sendcount,          // Number of elements to send
            MPI_Datatype sendtype,  // Data type of send buffer elements
            void *recvbuf,          // Starting address of receive buffer (root only)
            const int recvcounts[], // Array specifying the number of elements received from each process
            const int displs[],     // Array specifying displacements in the receive buffer
            MPI_Datatype recvtype,  // Data type of receive buffer elements
            int root,               // Rank of the root process
            MPI_Comm comm           // Communicator
        );
        */
        // collect energies from all ranks into all_energies array on rank 0
        MPI_Gatherv(local_energies.data(), 
                    n_rank_batches[sqd_data.mpi_rank], 
                    MPI_DOUBLE, 
                    all_energies.data(), 
                    n_rank_batches.data(), 
                    displs.data(), 
                    MPI_DOUBLE, 
                    0, 
                    sqd_data.comm);
        
        // Print all energies (for debugging)
        if (sqd_data.mpi_rank == 0) 
        {
            std::cout << "all_energies = [";
            for (size_t i = 0; i < all_energies.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << all_energies[i];
            }
            std::cout << "]\n";
        }

        // On all ranks, flatten local_occs (vector<vector<double>>) to a single vector
        // (flattening is needed for MPI communication)
        int occ_size = local_occs.empty() ? 0 : local_occs[0].size();
        std::vector<double> local_occs_flat;
        for (const auto& occ : local_occs)
            local_occs_flat.insert(local_occs_flat.end(), occ.begin(), occ.end());

        // Gather occupancy sizes
        int local_total_occ = n_rank_batches[sqd_data.mpi_rank] * occ_size;
        // similar to before, but now for occupancies
        // on rank 0, recvcounts_occ[i] = length of local_occs_flat from rank i
        // on rank 0, displs_occ[i] = displacement in final array where rank i's data starts
        std::vector<int> recvcounts_occ, displs_occ;
        // arrays for rank 0 to collect all occupancies from all ranks
        std::vector<double> all_occs_flat;
        std::vector<std::vector<double>> all_occs;
        
        if (sqd_data.mpi_rank == 0) 
        {
            recvcounts_occ.resize(sqd_data.mpi_size);
            displs_occ.resize(sqd_data.mpi_size);
        }
        // gather the sizes of local_occs_flat from all ranks into recvcounts_occ on rank 0
        MPI_Gather(&local_total_occ, 
                   1, 
                   MPI_INT,
                   recvcounts_occ.data(), 
                   1, 
                   MPI_INT, 
                   0, 
                   sqd_data.comm);
        
        if (sqd_data.mpi_rank == 0) 
        {
            // compute displacements based on recvcounts_occ
            int total_occ = 0;
            for (int i = 0; i < sqd_data.mpi_size; ++i) {
                displs_occ[i] = total_occ;
                total_occ += recvcounts_occ[i];
            }
            all_occs_flat.resize(total_occ);
        }
        
        // gather all local_occs_flat into all_occs_flat on rank 0
        MPI_Gatherv(local_occs_flat.data(), 
                    local_total_occ, 
                    MPI_DOUBLE,
                    all_occs_flat.data(), 
                    recvcounts_occ.data(), 
                    displs_occ.data(), 
                    MPI_DOUBLE,
                    0, 
                    sqd_data.comm);
        
        // On rank 0, reconstruct all_occs (vector of vector<double>) from all_occs_flat
        // each batch has occ_size elements, so we can split all_occs_flat accordingly
        if (sqd_data.mpi_rank == 0) 
        {
            int total_batches = all_energies.size();
            all_occs.reserve(total_batches);
            for (int i = 0; i < total_batches; ++i) {
                all_occs.emplace_back(
                    all_occs_flat.begin() + i * occ_size,
                    all_occs_flat.begin() + (i + 1) * occ_size
                );
            }
            // Now all_energies[i] and all_occs[i] correspond to batch i

            assert(!all_occs.empty());

            // initialize vector to hold average occupancies (avg on all batches)
            std::vector<double> avg_occs(occ_size, 0.0);
        
            // Sum over all batches
            for (const auto& occ : all_occs) 
            {
                for (size_t j = 0; j < occ_size; ++j) 
                {
                    avg_occs[j] += occ[j];
                }
            }
            // Divide by number of batches to get the average
            for (size_t j = 0; j < occ_size; ++j) 
            {
                avg_occs[j] /= static_cast<double>(total_batches);
            }
        
            // Print the result (debugging)
            std::cout << "avg_occs = [";
            for (size_t j = 0; j < avg_occs.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << avg_occs[j];
            }
            std::cout << "]\n";

            // Convert interleaved [alpha0, beta0, alpha1, beta1, ...] to { alpha[], beta[]
            // }. NOTE: assert ensures occs_batch size matches 2 * alpha.size().
            assert(2 * latest_occupancies[0].size() == avg_occs.size());
            for (std::size_t j = 0; j < latest_occupancies[0].size(); ++j)
            {
                latest_occupancies[0][j] = avg_occs[2 * j];     // alpha orbital
                latest_occupancies[1][j] = avg_occs[2 * j + 1]; // beta orbital
            }

            /*// again print latest_occupancies for debugging
            std::cout << "latest_occupancies:\n";

            // Alpha
            std::cout << "  alpha: [";
            for (size_t j = 0; j < latest_occupancies[0].size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << latest_occupancies[0][j];
            }
            std::cout << "]\n";

            // Beta
            std::cout << "  beta:  [";
            for (size_t j = 0; j < latest_occupancies[1].size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << latest_occupancies[1][j];
            }
            std::cout << "]\n";*/
        }

        
        
    }

    // Synchronize and tear down MPI. No MPI calls are allowed beyond this point.
    MPI_Finalize();

    return 0;
}
