// SPDX-License-Identifier: GPL-3.0-only
// QuantaSpindle optional CUDA runner. Binary protocol is documented in docs/cuda.md.
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {
constexpr uint64_t max_scenarios = 1000000;
constexpr uint64_t max_steps = 1000000;
constexpr uint64_t max_updates = 10000000000ULL;
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
uint64_t read_uint(const unsigned char* p, int n) {
    uint64_t value = 0;
    for (int i = 0; i < n; ++i) value |= uint64_t(p[i]) << (8*i);
    return value;
}
double read_double(const unsigned char* p) {
    uint64_t bits = read_uint(p, 8); double value;
    std::memcpy(&value, &bits, 8); return value;
}
void put_uint(unsigned char* p, uint64_t value, int n) {
    for (int i = 0; i < n; ++i) p[i] = static_cast<unsigned char>(value >> (8*i));
}
void put_double(unsigned char* p, double value) {
    uint64_t bits; std::memcpy(&bits, &value, 8); put_uint(p, bits, 8);
}
struct Record { double terminal, total_return, volatility, drawdown; };
static_assert(sizeof(Record) == 32, "unexpected record padding");

__global__ void gbm(Record* output, int* invalid, uint64_t count, uint64_t start,
                    uint64_t seed, uint32_t steps, double initial, double mu,
                    double sigma, double dt) {
    uint64_t i = uint64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= count) return;
    curandStatePhilox4_32_10_t random;
    // Scenario index, never worker/block/batch index, identifies the stream.
    curand_init(seed, start+i, 0, &random);
    double price = initial, peak = initial, drawdown = 0, mean = 0, m2 = 0;
    double drift = (mu-0.5*sigma*sigma)*dt, diffusion = sigma*sqrt(dt);
    for (uint32_t step = 0; step < steps; ++step) {
        double log_return = drift + diffusion*curand_normal_double(&random);
        double next = price*exp(log_return);
        if (!isfinite(next) || next <= 0 || !isfinite(log_return)) {
            atomicExch(invalid, 1); return;
        }
        double delta = log_return-mean;
        mean += delta/double(step+1);
        m2 += delta*(log_return-mean);
        price = next; peak = fmax(peak, price);
        drawdown = fmax(drawdown, (peak-price)/peak);
    }
    double volatility = steps > 1 ? sqrt(fmax(0.0,m2)/double(steps-1)/dt) : 0;
    double total_return = price/initial-1;
    if (!isfinite(volatility) || !isfinite(total_return)) { atomicExch(invalid,1); return; }
    output[i] = {price, total_return, volatility, drawdown};
}
struct Resources {
    Record* records = nullptr; int* invalid = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    size_t capacity = 0;
    void initialize() {
        check(cudaMalloc(reinterpret_cast<void**>(&invalid),sizeof(int)),"status allocation");
        check(cudaEventCreate(&begin),"begin event creation");
        check(cudaEventCreate(&end),"end event creation");
    }
    void reserve(uint64_t count) {
        if (count <= capacity) return;
        if (records) {
            check(cudaFree(records),"record release"); records = nullptr;
        }
        check(cudaMalloc(reinterpret_cast<void**>(&records),size_t(count)*sizeof(Record)),"record allocation");
        capacity = size_t(count);
    }
    ~Resources() {
        if (records) cudaFree(records);
        if (invalid) cudaFree(invalid);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }
};
}

int main(int argc, char** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY); _setmode(_fileno(stdout), _O_BINARY);
#endif
    try {
        bool streaming = argc == 2 && std::strcmp(argv[1], "--stream") == 0;
        if (argc != 1 && !streaming) throw std::runtime_error("usage: quantaspindle-cuda [--stream]");
        if (streaming) {
            std::cout.write("QSPNCS01", 8); std::cout.flush();
            if (!std::cout) throw std::runtime_error("greeting write failed");
        }
        Resources resources;
        std::vector<Record> output;
        bool initialized = false;
        int device = 0, device_count = 0, runtime_version = 0, driver_version = 0;
        cudaDeviceProp properties{};
        for (;;) {
            unsigned char request[72];
            std::cin.read(reinterpret_cast<char*>(request), sizeof(request));
            if (streaming && std::cin.gcount() == 0 && std::cin.eof()) break;
            if (std::cin.gcount() != sizeof(request)) throw std::runtime_error("truncated request");
            auto wall_start = std::chrono::steady_clock::now();
            if (!streaming && std::cin.get() != std::char_traits<char>::eof()) throw std::runtime_error("trailing request bytes");
            if (std::memcmp(request, "QSPNCU01",8) || read_uint(request+8,4) != 1) throw std::runtime_error("unsupported protocol");
            uint32_t steps = uint32_t(read_uint(request+12,4));
            uint64_t count = read_uint(request+16,8), start = read_uint(request+24,8), seed = read_uint(request+32,8);
            double initial = read_double(request+40), mu = read_double(request+48);
            double sigma = read_double(request+56), dt = read_double(request+64);
            if (count == 0 || count > max_scenarios || steps == 0 || steps > max_steps ||
                count > max_updates/steps || start > UINT64_MAX-(count-1) ||
                count > SIZE_MAX/sizeof(Record)) throw std::runtime_error("dimensions outside runner limits");
            if (!std::isfinite(initial) || initial <= 0 || !std::isfinite(mu) ||
                !std::isfinite(sigma) || sigma < 0 || !std::isfinite(dt) || dt <= 0 || !std::isfinite(double(steps)*dt) ||
                !std::isfinite((mu-0.5*sigma*sigma)*dt) || !std::isfinite(sigma*std::sqrt(dt)))
                throw std::runtime_error("invalid numeric inputs");
            if (!initialized) {
                try {
                    check(cudaGetDeviceCount(&device_count),"device enumeration");
                    if (device_count < 1) throw std::runtime_error("no CUDA device");
                    check(cudaSetDevice(device),"device selection");
                    check(cudaGetDeviceProperties(&properties,device),"device properties");
                    check(cudaRuntimeGetVersion(&runtime_version),"runtime version");
                    check(cudaDriverGetVersion(&driver_version),"driver version");
                    resources.initialize();
                    initialized = true;
                } catch (const std::exception& error) {
                    throw std::runtime_error(std::string("CUDA unavailable: ")+error.what());
                }
            }
            resources.reserve(count);
            check(cudaMemset(resources.invalid,0,sizeof(int)),"status initialization");
            check(cudaEventRecord(resources.begin),"begin event");
            gbm<<<unsigned((count+255)/256),256>>>(resources.records,resources.invalid,count,start,seed,steps,initial,mu,sigma,dt);
            check(cudaGetLastError(),"kernel launch");
            check(cudaEventRecord(resources.end),"end event");
            check(cudaEventSynchronize(resources.end),"kernel completion");
            float kernel_ms = 0; check(cudaEventElapsedTime(&kernel_ms,resources.begin,resources.end),"kernel timing");
            int invalid = 0; check(cudaMemcpy(&invalid,resources.invalid,sizeof(int),cudaMemcpyDeviceToHost),"status copy");
            if (invalid) throw std::runtime_error("nonfinite or nonpositive simulated output");
            output.resize(size_t(count));
            check(cudaMemcpy(output.data(),resources.records,size_t(count)*sizeof(Record),cudaMemcpyDeviceToHost),"output copy");
            double host_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall_start).count();
            unsigned char header[176]{};
            std::memcpy(header,"QSPNCR01",8); put_uint(header+8,1,4); put_uint(header+12,0,4);
            put_uint(header+16,count,8); put_double(header+24,kernel_ms); put_double(header+32,host_ms);
            put_uint(header+40,runtime_version,4); put_uint(header+44,driver_version,4);
            std::memcpy(header+48,properties.name,std::min(size_t(127),std::strlen(properties.name)));
            std::cout.write(reinterpret_cast<char*>(header),sizeof(header));
            for (const auto& record : output) {
                unsigned char bytes[32]; put_double(bytes,record.terminal); put_double(bytes+8,record.total_return);
                put_double(bytes+16,record.volatility); put_double(bytes+24,record.drawdown);
                std::cout.write(reinterpret_cast<char*>(bytes),sizeof(bytes));
            }
            std::cout.flush();
            if (!std::cout) throw std::runtime_error("output write failed");
            if (!streaming) break;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "quantaspindle-cuda: " << error.what() << '\n'; return 1;
    }
}
