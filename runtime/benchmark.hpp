#pragma once

#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <fstream>
#include <iostream>

#include "PowerSensor.hpp"
#include "loadable.hpp"
#include "cudla_runtime.hpp"
#include "cudla.h"

#define RUNTIMES 2
#define MIN_BUFFERS 8
// Every batch contains 256 complex samples (and the 15 other historical taps)
#define BATCHES_TO_RUN 40'000
// Cache size in bytes. Consider every loction where the samples can be cached.
#define CACHE_SIZE 4'000'000
// How often we loop through all the files and execute them
#define ITERATIONS 20


class Benchmark {

public:
    /**
     * Construct a Benchmark instance and initialize power sensor and CSV output.
     *
     * @param powerSensor The device path to the power sensor for energy measurements.
     * @param csvFileName The path to the CSV file for writing benchmark results.
     */
    Benchmark(const std::string powerSensor, const std::string csvFileName);
    ~Benchmark() noexcept;

    /**
     * Initialize the benchmark environment by setting up CUDA and creating runtime instances.
     * Writes CSV header and creates RUNTIMES number of CUDLARuntime instances.
     *
     * @return True if initialization succeeded, false otherwise.
     */
    bool init();

    /**
     * Load all .nvdla loadable files from the specified directory.
     *
     * @param dir The directory path containing the .nvdla loadable files to benchmark.
     */
    void load_files(std::string const& dir);

    /**
     * Run the complete benchmark suite on all loaded files.
     * Executes all files on all DLA cores for ITERATIONS iterations and records results to CSV.
     */
    void run();

    /**
     * Run a single loadable file on a specific DLA core with power and timing measurements.
     * Allocates buffers, performs warm-up, executes multiple runs, and records energy/timing data.
     *
     * @param file The path to the .nvdla loadable file to execute.
     * @param dla The DLA core to use for execution (default is 0).
     */
    void run_single_dla(const std::string& file, const int dla = 0);

private:
    std::ofstream csvFile;
    PowerSensor3::PowerSensor ps3;
    PowerSensor3::State start, stop, base_start, base_stop;

    std::vector<std::string> files;
    std::vector<std::shared_ptr<CUDLARuntime>> runtimes;

    /**
     * Calculate the number of buffers and runs needed based on input shape and cache size.
     * Ensures cache doesn't reuse data by allocating enough buffers to exceed cache size.
     *
     * @param inputShape Tuple of (N, C, H, W) representing the input tensor dimensions.
     * @param dlaCount The number of DLA cores being used.
     * @return Tuple of (buffers, runs, samplesPerRun) for the benchmark execution.
     */
    std::tuple<size_t, size_t, size_t> calculateBuffersAndRuns(std::tuple<int, int, int, int> inputShape, int dlaCount) const noexcept;

    /**
     * CUDA host callback function to record power sensor state at benchmark start.
     *
     * @param instance Pointer to the Benchmark instance.
     */
    static void CUDART_CB hostCallbackStart(void* instance);

    /**
     * CUDA host callback function to record power sensor state at benchmark stop.
     *
     * @param instance Pointer to the Benchmark instance.
     */
    static void CUDART_CB hostCallbackStop(void* instance);
};
