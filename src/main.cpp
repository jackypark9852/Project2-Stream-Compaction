/**
 * @file      main.cpp
 * @brief     Stream compaction test program with CSV mode + selectable tests + size + repeats
 */

#include <cstdio>
#include <vector>
#include <string>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <ctime>
#include <cmath>

#include <cuda_runtime.h>

#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

 // Runtime sizes (set by --exp)
static int SIZE = 1 << 28;
static int NPOT = (1 << 28) - 3;

// Host buffers (allocated after parsing)
static int* a = nullptr;
static int* b = nullptr;
static int* c = nullptr;

// ---------- Pretty-print helpers (non-CSV mode) ----------
static void print_times_line(const char* label, const std::vector<float>& ts, const char* unit) {
    printf("%s: ", label);
    for (size_t i = 0; i < ts.size(); ++i) {
        printf("%.3f", ts[i]);
        if (i + 1 < ts.size()) printf(" | ");
    }
    printf(" %s\n", unit);
}
static void print_avg_line(const std::vector<float>& ts, const char* unit) {
    float sum = std::accumulate(ts.begin(), ts.end(), 0.0f);
    float avg = (ts.empty() ? 0.0f : sum / ts.size());
    printf("avg: %.3f %s\n\n", avg, unit);
}

// ---------- Args ----------
struct Args {
    int runs = 1;
    int exp = 28;
    bool csv = false;   // CSV mode: emit only RUN/SUMMARY lines, exactly one test

    bool any = true;   // becomes true if any selection flag is used

    // Group flags
    bool all = false;
    bool scans = false;
    bool compacts = false;

    // Individual scan tests
    bool scan_cpu_pot = false, scan_cpu_npot = false;
    bool scan_naive_pot = false, scan_naive_npot = false;
    bool scan_eff_pot = false, scan_eff_npot = false;
    bool scan_thrust_pot = false, scan_thrust_npot = false;

    // Individual compaction tests
    bool compact_cpu_noscan_pot = false, compact_cpu_noscan_npot = false;
    bool compact_cpu_scan = false;
    bool compact_eff_pot = false, compact_eff_npot = false;
};

static bool starts_with(const char* s, const char* pfx) {
    return std::strncmp(s, pfx, std::strlen(pfx)) == 0;
}

static void print_help(const char* prog) {
    printf(
        "Usage: %s [--runs=N] [--exp=K] [--csv] [--all|--scans|--compacts] [individual flags]\n"
        "  --runs=N        Number of repetitions per test (default 5)\n"
        "  --exp=K         Problem size is SIZE = 2^K (default 28), NPOT = SIZE-3\n"
        "  --csv           CSV mode: run exactly one selected test and print RUN/SUMMARY lines only\n"
        "\n"
        "Groups:\n"
        "  --all           Run all tests (default if no flags given)\n"
        "  --scans         Run only scan tests\n"
        "  --compacts      Run only compaction tests\n"
        "\n"
        "Scan flags:\n"
        "  --scan-cpu-pot           --scan-cpu-npot\n"
        "  --scan-naive-pot         --scan-naive-npot\n"
        "  --scan-efficient-pot     --scan-efficient-npot\n"
        "  --scan-thrust-pot        --scan-thrust-npot\n"
        "\n"
        "Compaction flags:\n"
        "  --compact-cpu-noscan-pot --compact-cpu-noscan-npot\n"
        "  --compact-cpu-scan\n"
        "  --compact-efficient-pot  --compact-efficient-npot\n",
        prog
    );
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const char* s = argv[i];

        if (!std::strcmp(s, "--help") || !std::strcmp(s, "-h")) {
            print_help(argv[0]);
            std::exit(0);
        }
        if (starts_with(s, "--runs=")) { a.runs = std::max(1, std::atoi(s + 7)); continue; }
        if (starts_with(s, "--exp=")) { a.exp = std::max(1, std::atoi(s + 6)); continue; }
        if (!std::strcmp(s, "--csv")) { a.csv = true; a.any = true; continue; }

        // groups
        if (!std::strcmp(s, "--all")) { a.all = true; a.any = true; continue; }
        if (!std::strcmp(s, "--scans")) { a.scans = true; a.any = true; continue; }
        if (!std::strcmp(s, "--compacts")) { a.compacts = true; a.any = true; continue; }

        // scan tests
        if (!std::strcmp(s, "--scan-cpu-pot")) { a.scan_cpu_pot = true; a.any = true; continue; }
        if (!std::strcmp(s, "--scan-cpu-npot")) { a.scan_cpu_npot = true; a.any = true; continue; }
        if (!std::strcmp(s, "--scan-naive-pot")) { a.scan_naive_pot = true; a.any = true; continue; }
        if (!std::strcmp(s, "--scan-naive-npot")) { a.scan_naive_npot = true; a.any = true; continue; }
        if (!std::strcmp(s, "--scan-efficient-pot")) { a.scan_eff_pot = true; a.any = true; continue; }
        if (!std::strcmp(s, "--scan-efficient-npot")) { a.scan_eff_npot = true; a.any = true; continue; }
        if (!std::strcmp(s, "--scan-thrust-pot")) { a.scan_thrust_pot = true; a.any = true; continue; }
        if (!std::strcmp(s, "--scan-thrust-npot")) { a.scan_thrust_npot = true; a.any = true; continue; }

        // compact tests
        if (!std::strcmp(s, "--compact-cpu-noscan-pot")) { a.compact_cpu_noscan_pot = true; a.any = true; continue; }
        if (!std::strcmp(s, "--compact-cpu-noscan-npot")) { a.compact_cpu_noscan_npot = true; a.any = true; continue; }
        if (!std::strcmp(s, "--compact-cpu-scan")) { a.compact_cpu_scan = true; a.any = true; continue; }
        if (!std::strcmp(s, "--compact-efficient-pot")) { a.compact_eff_pot = true; a.any = true; continue; }
        if (!std::strcmp(s, "--compact-efficient-npot")) { a.compact_eff_npot = true; a.any = true; continue; }

        printf("Warning: unknown arg '%s'\n", s);
    }
    return a;
}

static bool want_scan_group(const Args& a) { return !a.any || a.all || a.scans; }
static bool want_compact_group(const Args& a) { return !a.any || a.all || a.compacts; }

// ---------- CSV helpers ----------
static std::string iso_utc_now() {
    char buf[32];
    std::time_t t = std::time(nullptr);
#if defined(_WIN32)
    std::tm tm{};
    gmtime_s(&tm, &t);
#else
    std::tm tm{};
    gmtime_r(&t, &tm);
#endif
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buf);
}
static void stats(const std::vector<float>& xs, float& avg, float& stddev, float& mn, float& mx) {
    if (xs.empty()) { avg = stddev = mn = mx = 0.f; return; }
    double sum = 0.0;
    mn = xs[0]; mx = xs[0];
    for (float v : xs) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
    avg = static_cast<float>(sum / xs.size());
    double var = 0.0;
    for (float v : xs) { double d = v - avg; var += d * d; }
    stddev = xs.size() > 1 ? static_cast<float>(std::sqrt(var / xs.size())) : 0.f;
}

enum class TestID {
    ScanCpuPot, ScanCpuNpot,
    ScanNaivePot, ScanNaiveNpot,
    ScanEffPot, ScanEffNpot,
    ScanThrustPot, ScanThrustNpot,
    CompCpuNoScanPot, CompCpuNoScanNpot,
    CompCpuScan,
    CompEffPot, CompEffNpot,
    Unknown
};
struct SelectedTest {
    TestID id = TestID::Unknown;
    const char* test_id_str = "";
    const char* pot_kind = "pot"; // "pot" or "npot"
    bool is_scan = true;
};

static SelectedTest pick_test_csv(const Args& a) {
    int count = 0;
    SelectedTest sel{};
    auto set = [&](TestID id, const char* s, const char* kind, bool is_scan) {
        sel = { id, s, kind, is_scan }; ++count;
    };

    if (a.scan_cpu_pot)        set(TestID::ScanCpuPot, "scan-cpu-pot", "pot", true);
    if (a.scan_cpu_npot)       set(TestID::ScanCpuNpot, "scan-cpu-npot", "npot", true);
    if (a.scan_naive_pot)      set(TestID::ScanNaivePot, "scan-naive-pot", "pot", true);
    if (a.scan_naive_npot)     set(TestID::ScanNaiveNpot, "scan-naive-npot", "npot", true);
    if (a.scan_eff_pot)        set(TestID::ScanEffPot, "scan-efficient-pot", "pot", true);
    if (a.scan_eff_npot)       set(TestID::ScanEffNpot, "scan-efficient-npot", "npot", true);
    if (a.scan_thrust_pot)     set(TestID::ScanThrustPot, "scan-thrust-pot", "pot", true);
    if (a.scan_thrust_npot)    set(TestID::ScanThrustNpot, "scan-thrust-npot", "npot", true);

    if (a.compact_cpu_noscan_pot)  set(TestID::CompCpuNoScanPot, "compact-cpu-noscan-pot", "pot", false);
    if (a.compact_cpu_noscan_npot) set(TestID::CompCpuNoScanNpot, "compact-cpu-noscan-npot", "npot", false);
    if (a.compact_cpu_scan)        set(TestID::CompCpuScan, "compact-cpu-scan", "pot", false);
    if (a.compact_eff_pot)         set(TestID::CompEffPot, "compact-efficient-pot", "pot", false);
    if (a.compact_eff_npot)        set(TestID::CompEffNpot, "compact-efficient-npot", "npot", false);

    if (count != 1) sel.id = TestID::Unknown; // require exactly one
    return sel;
}

// Run ONE iteration and return (ok, time_ms)
static std::pair<bool, float> run_one(const SelectedTest& tsel, int n_pot, int n_npot) {
    // Prepare inputs
    if (tsel.is_scan) { genArray(n_pot - 1, a, 50); a[n_pot - 1] = 0; }
    else { genArray(n_pot - 1, a, 4);  a[n_pot - 1] = 0; }

    int n = (std::strcmp(tsel.pot_kind, "npot") == 0) ? n_npot : n_pot;
    zeroArray(n_pot, c); // clear superset

    cudaError_t cerr = cudaSuccess;
    switch (tsel.id) {
    case TestID::ScanCpuPot:
    case TestID::ScanCpuNpot:
        StreamCompaction::CPU::scan(n, c, a);
        return { true, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() };

    case TestID::ScanNaivePot:
    case TestID::ScanNaiveNpot:
        StreamCompaction::Naive::scan(n, c, a);
        cerr = cudaGetLastError();
        return { cerr == cudaSuccess, StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation() };

    case TestID::ScanEffPot:
    case TestID::ScanEffNpot:
        StreamCompaction::Efficient::scan(n, c, a);
        cerr = cudaGetLastError();
        return { cerr == cudaSuccess, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() };

    case TestID::ScanThrustPot:
    case TestID::ScanThrustNpot:
        StreamCompaction::Thrust::scan(n, c, a);
        cerr = cudaGetLastError();
        return { cerr == cudaSuccess, StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation() };

    case TestID::CompCpuNoScanPot:
    case TestID::CompCpuNoScanNpot: {
        (void)StreamCompaction::CPU::compactWithoutScan(n, c, a);
        return { true, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() };
    }
    case TestID::CompCpuScan: {
        (void)StreamCompaction::CPU::compactWithScan(n, c, a);
        return { true, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() };
    }
    case TestID::CompEffPot:
    case TestID::CompEffNpot: {
        (void)StreamCompaction::Efficient::compact(n, c, a);
        cerr = cudaGetLastError();
        return { cerr == cudaSuccess, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() };
    }
    default:
        return { false, 0.f };
    }
}

int main(int argc, char* argv[]) {
    Args ARGS = parse_args(argc, argv);
    const int RUNS = ARGS.runs;

    // Set sizes and allocate
    SIZE = 1 << ARGS.exp;
    NPOT = SIZE - 3;

    if (ARGS.exp > 30) {
        printf("Warning: --exp=%d -> SIZE=%d may be very large.\n", ARGS.exp, SIZE);
    }

    a = new (std::nothrow) int[SIZE];
    b = new (std::nothrow) int[SIZE];
    c = new (std::nothrow) int[SIZE];
    if (!a || !b || !c) {
        fprintf(stderr, "Host allocation failed. Try a smaller --exp.\n");
        delete[] a; delete[] b; delete[] c;
        return 1;
    }

    // ======================
    // CSV MODE: single test
    // ======================
    if (ARGS.csv) {
        SelectedTest tsel = pick_test_csv(ARGS);
        if (tsel.id == TestID::Unknown) {
            // Only stderr on misuse; stdout must remain clean for CSV redirection.
            fprintf(stderr, "CSV mode requires exactly ONE test flag and --exp=K.\n");
            delete[] a; delete[] b; delete[] c;
            return 2;
        }
        const int n_pot = SIZE;
        const int n_npot = NPOT;
        const int n_eff = (std::strcmp(tsel.pot_kind, "npot") == 0) ? n_npot : n_pot;

        std::vector<float> good; good.reserve(RUNS);

        for (int i = 1; i <= RUNS; ++i) {
            auto [ok, tms] = run_one(tsel, n_pot, n_npot);
            const std::string ts = iso_utc_now();
            if (ok) {
                good.push_back(tms);
                // RUN line
                printf("RUN,%s,%s,%d,%d,%s,%d,%.3f,ok\n",
                    ts.c_str(), tsel.test_id_str, ARGS.exp, n_eff, tsel.pot_kind, i, tms);
            }
            else {
                // RUN line with error status and empty time field
                printf("RUN,%s,%s,%d,%d,%s,%d,,error\n",
                    ts.c_str(), tsel.test_id_str, ARGS.exp, n_eff, tsel.pot_kind, i);
            }
            fflush(stdout);
        }

        float avg = 0, sd = 0, mn = 0, mx = 0;
        stats(good, avg, sd, mn, mx);
        const std::string ts = iso_utc_now();
        const int ok_runs = static_cast<int>(good.size());
        const char* overall = (ok_runs == RUNS) ? "ok" : "partial";

        if (ok_runs > 0) {
            printf("SUMMARY,%s,%s,%d,%d,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%s\n",
                ts.c_str(), tsel.test_id_str, ARGS.exp, n_eff, tsel.pot_kind,
                RUNS, ok_runs, avg, sd, mn, mx, overall);
        }
        else {
            // No successful runs -> blank stats fields
            printf("SUMMARY,%s,%s,%d,%d,%s,%d,%d,,,,,%s\n",
                ts.c_str(), tsel.test_id_str, ARGS.exp, n_eff, tsel.pot_kind,
                RUNS, ok_runs, overall);
        }

        delete[] a; delete[] b; delete[] c;
        return 0;
    }

    // ======================
    // Non-CSV mode (pretty)
    // ======================
    printf("Running with SIZE=2^%d (%d), NPOT=%d, runs=%d\n", ARGS.exp, SIZE, NPOT, RUNS);

    // ---- SCAN TESTS ----
    if (want_scan_group(ARGS) ||
        ARGS.scan_cpu_pot || ARGS.scan_cpu_npot ||
        ARGS.scan_naive_pot || ARGS.scan_naive_npot ||
        ARGS.scan_eff_pot || ARGS.scan_eff_npot ||
        ARGS.scan_thrust_pot || ARGS.scan_thrust_npot)
    {
        printf("\n");
        printf("****************\n");
        printf("** SCAN TESTS **\n");
        printf("****************\n");

        genArray(SIZE - 1, a, 50);  a[SIZE - 1] = 0;
        printArray(SIZE, a, true);

        auto ensure_cpu_baseline_pot = [&]() {
            zeroArray(SIZE, b);
            StreamCompaction::CPU::scan(SIZE, b, a);
        };
        auto ensure_cpu_baseline_npot = [&]() {
            zeroArray(SIZE, b);
            StreamCompaction::CPU::scan(NPOT, b, a);
        };

        if (!ARGS.any || ARGS.all || ARGS.scans || ARGS.scan_cpu_pot) {
            zeroArray(SIZE, b);
            std::vector<float> ts; ts.reserve(RUNS);
            for (int i = 0; i < RUNS; ++i) {
                zeroArray(SIZE, b);
                StreamCompaction::CPU::scan(SIZE, b, a);
                ts.push_back(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
            }
            print_times_line("cpu scan, power-of-two (ms)", ts, "");
            print_avg_line(ts, "ms");
        }

        if (!ARGS.any || ARGS.all || ARGS.scans || ARGS.scan_cpu_npot) {
            zeroArray(SIZE, c);
            std::vector<float> ts; ts.reserve(RUNS);
            for (int i = 0; i < RUNS; ++i) {
                zeroArray(SIZE, c);
                StreamCompaction::CPU::scan(NPOT, c, a);
                ts.push_back(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
            }
            print_times_line("cpu scan, non-power-of-two (ms)", ts, "");
            print_avg_line(ts, "ms");
            printCmpResult(NPOT, b, c);
        }

        if (!ARGS.any || ARGS.all || ARGS.scans || ARGS.scan_naive_pot) {
            ensure_cpu_baseline_pot();
            zeroArray(SIZE, c);
            std::vector<float> ts; ts.reserve(RUNS);
            for (int i = 0; i < RUNS; ++i) {
                zeroArray(SIZE, c);
                StreamCompaction::Naive::scan(SIZE, c, a);
                ts.push_back(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation());
            }
            print_times_line("naive scan, power-of-two (ms)", ts, "");
            print_avg_line(ts, "ms");
            printCmpResult(SIZE, b, c);
        }

        if (!ARGS.any || ARGS.all || ARGS.scans || ARGS.scan_naive_npot) {
            ensure_cpu_baseline_npot();
            zeroArray(SIZE, c);
            std::vector<float> ts; ts.reserve(RUNS);
            for (int i = 0; i < RUNS; ++i) {
                zeroArray(SIZE, c);
                StreamCompaction::Naive::scan(NPOT, c, a);
                ts.push_back(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation());
            }
            print_times_line("naive scan, non-power-of-two (ms)", ts, "");
            print_avg_line(ts, "ms");
            printCmpResult(NPOT, b, c);
        }

        if (!ARGS.any || ARGS.all || ARGS.scans || ARGS.scan_eff_pot) {
            ensure_cpu_baseline_pot();
            zeroArray(SIZE, c);
            std::vector<float> ts; ts.reserve(RUNS);
            for (int i = 0; i < RUNS; ++i) {
                zeroArray(SIZE, c);
                StreamCompaction::Efficient::scan(SIZE, c, a);
                ts.push_back(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
            }
            print_times_line("work-efficient scan, power-of-two (ms)", ts, "");
            print_avg_line(ts, "ms");
            printCmpResult(SIZE, b, c);
        }

        if (!ARGS.any || ARGS.all || ARGS.scans || ARGS.scan_eff_npot) {
            ensure_cpu_baseline_npot();
            zeroArray(SIZE, c);
            std::vector<float> ts; ts.reserve(RUNS);
            for (int i = 0; i < RUNS; ++i) {
                zeroArray(SIZE, c);
                StreamCompaction::Efficient::scan(NPOT, c, a);
                ts.push_back(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
            }
            print_times_line("work-efficient scan, non-power-of-two (ms)", ts, "");
            print_avg_line(ts, "ms");
            printCmpResult(NPOT, b, c);
        }

        if (!ARGS.any || ARGS.all || ARGS.scans || ARGS.scan_thrust_pot) {
            ensure_cpu_baseline_pot();
            zeroArray(SIZE, c);
            std::vector<float> ts; ts.reserve(RUNS);
            for (int i = 0; i < RUNS; ++i) {
                zeroArray(SIZE, c);
                StreamCompaction::Thrust::scan(SIZE, c, a);
                ts.push_back(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());
            }
            print_times_line("thrust scan, power-of-two (ms)", ts, "");
            print_avg_line(ts, "ms");
            printCmpResult(SIZE, b, c);
        }

        if (!ARGS.any || ARGS.all || ARGS.scans || ARGS.scan_thrust_npot) {
            ensure_cpu_baseline_npot();
            zeroArray(SIZE, c);
            std::vector<float> ts; ts.reserve(RUNS);
            for (int i = 0; i < RUNS; ++i) {
                zeroArray(SIZE, c);
                StreamCompaction::Thrust::scan(NPOT, c, a);
                ts.push_back(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());
            }
            print_times_line("thrust scan, non-power-of-two (ms)", ts, "");
            print_avg_line(ts, "ms");
            printCmpResult(NPOT, b, c);
        }
    }

    // ---- STREAM COMPACTION TESTS ----
    if (want_compact_group(ARGS) ||
        ARGS.compact_cpu_noscan_pot || ARGS.compact_cpu_noscan_npot ||
        ARGS.compact_cpu_scan || ARGS.compact_eff_pot || ARGS.compact_eff_npot)
    {
        printf("\n");
        printf("*****************************\n");
        printf("** STREAM COMPACTION TESTS **\n");
        printf("*****************************\n");

        genArray(SIZE - 1, a, 4);  a[SIZE - 1] = 0;
        printArray(SIZE, a, true);

        int expectedCount = -1, expectedNPOT = -1;

        if (!ARGS.any || ARGS.all || ARGS.compacts || ARGS.compact_cpu_noscan_pot) {
            zeroArray(SIZE, b);
            std::vector<float> ts; ts.reserve(RUNS);
            int lastCount = 0;
            for (int i = 0; i < RUNS; ++i) {
                zeroArray(SIZE, b);
                lastCount = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
                ts.push_back(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
                if (i == 0) expectedCount = lastCount;
            }
            print_times_line("cpu compact without scan, power-of-two (ms)", ts, "");
            print_avg_line(ts, "ms");
            printArray(expectedCount, b, true);
            printCmpLenResult(expectedCount, expectedCount, b, b);
        }

        if (!ARGS.any || ARGS.all || ARGS.compacts || ARGS.compact_cpu_noscan_npot) {
            zeroArray(SIZE, c);
            std::vector<float> ts; ts.reserve(RUNS);
            int lastCount = 0;
            for (int i = 0; i < RUNS; ++i) {
                zeroArray(SIZE, c);
                lastCount = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
                ts.push_back(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
                if (i == 0) expectedNPOT = lastCount;
            }
            print_times_line("cpu compact without scan, non-power-of-two (ms)", ts, "");
            print_avg_line(ts, "ms");
            printArray(expectedNPOT, c, true);
            printCmpLenResult(expectedNPOT, expectedNPOT, b, c);
        }

        if (!ARGS.any || ARGS.all || ARGS.compacts || ARGS.compact_cpu_scan) {
            zeroArray(SIZE, c);
            std::vector<float> ts; ts.reserve(RUNS);
            int lastCount = 0;
            for (int i = 0; i < RUNS; ++i) {
                zeroArray(SIZE, c);
                lastCount = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
                ts.push_back(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
            }
            print_times_line("cpu compact with scan (ms)", ts, "");
            print_avg_line(ts, "ms");

            if (expectedCount < 0) expectedCount = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
            printCmpLenResult(lastCount, expectedCount, b, c);
        }

        if (!ARGS.any || ARGS.all || ARGS.compacts || ARGS.compact_eff_pot) {
            if (expectedCount < 0) expectedCount = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
            zeroArray(SIZE, c);
            std::vector<float> ts; ts.reserve(RUNS);
            int lastCount = 0;
            for (int i = 0; i < RUNS; ++i) {
                zeroArray(SIZE, c);
                lastCount = StreamCompaction::Efficient::compact(SIZE, c, a);
                ts.push_back(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
            }
            print_times_line("work-efficient compact, power-of-two (ms)", ts, "");
            print_avg_line(ts, "ms");
            printCmpLenResult(lastCount, expectedCount, b, c);
        }

        if (!ARGS.any || ARGS.all || ARGS.compacts || ARGS.compact_eff_npot) {
            if (expectedNPOT < 0) expectedNPOT = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
            zeroArray(SIZE, c);
            std::vector<float> ts; ts.reserve(RUNS);
            int lastCount = 0;
            for (int i = 0; i < RUNS; ++i) {
                zeroArray(SIZE, c);
                lastCount = StreamCompaction::Efficient::compact(NPOT, c, a);
                ts.push_back(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
            }
            print_times_line("work-efficient compact, non-power-of-two (ms)", ts, "");
            print_avg_line(ts, "ms");
            printCmpLenResult(lastCount, expectedNPOT, b, c);
        }
    }

#ifdef _WIN32
    system("pause");
#endif
    delete[] a; delete[] b; delete[] c;
    return 0;
}
