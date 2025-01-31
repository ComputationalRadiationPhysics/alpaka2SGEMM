#!/usr/bin/env python3

"""
- Mibibytes for matrices/colors 
- profiler GMem with 1000x1000 and 4000x4000 to see what goes wrong
- profile SMem and see where the bottleneck is
- export OMP_NUM_THREADS=16
"""

"""
Usage:
- We're patching the SGEMM_CUDA project to use custom sizes and to disable verification
    - See the patches in the `build_blogbench` function
    - Make sure to set the right CUDA compute capability in line 177
        - This capability should match -DCMAKE_CUDA_ARCHITECTURES=89 in the CMake command in line 34
    - The important part is in line 190, which is similar to 10_bench.cpp line 221
      As long as those loops are in sync, the benchmarks test the same matrix sizes
    - `int repeat_times = 20;` should match `uint64_t bench_repeats = 20;` in 10_bench.cpp
- NOTE: The benchmarks will take a while to run, please don't run other tasks in parallel
        The biggest matrix is roughly 20k x 20k, which is a lot of data to process
- The suite will cache the results in `alpaka_bench.txt` and `blog_bench.txt`
  so if you want to rerun the benchmarks, you can delete these files
  or run `./.venv/bin/python3 measure.py clean all` to remove all build files
- Plots will be saved in the `resources` directory
"""

from typing import List, Dict, Tuple, Callable

import subprocess
import os
import sys

#CMAKE = "cmake .. -Dalpaka_API_OMP=ON -Dalpaka_API_CUDA=ON -Dalpaka_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -G Ninja"
CMAKE = "cmake .. -Dalpaka_EXEC_CpuOmpBlocksAndThreads=OFF -Dalpaka_API_OMP=ON -Dalpaka_API_CUDA=ON -Dalpaka_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -G Ninja"
ALPAKA_BENCH_FILE = os.path.abspath("alpaka_bench.txt")
BLOG_BENCH_FILE = os.path.abspath("blog_bench.txt")

def call_cmd(
    cmd: List,
    cwd: str | None = None,
    env: Dict | None = None,
    print_output: bool = False
) -> subprocess.CompletedProcess[bytes] | Tuple[int, str]:
    if env is None:
        print("[CMD] " + " ".join(cmd))
    else:
        envstr = " ".join([f"{key}={value}" for key, value in env.items()])
        print("[CMD] " + envstr + " " + " ".join(cmd))
    if print_output:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env) as process:
            output = ""
            for line in process.stdout:
                l = line.decode("utf-8")
                print(l, end="")
                output += l
            for line in process.stderr:
                print(line.decode("utf-8"), end="")
            process.wait()
            return process.returncode, output
    else:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env)

def call_and_catch(
    cmd: List,
    err_msg: str,
    cwd: str | None = None,
    env: Dict | None = None,
    print_output: bool = False
) -> str:
    process = call_cmd(cmd, cwd, env, print_output)
    if isinstance(process, subprocess.CompletedProcess):
        if process.returncode != 0:
            print(err_msg)
            print(process.stderr.decode("utf-8"))
            print(process.stdout.decode("utf-8"))
            exit(process.returncode)
        return process.stdout.decode("utf-8")
    else:
        code, output = process
        if code != 0:
            print(err_msg)
            print(output)
            exit(code)
        return output

def MKDIR(path: str):
    if not os.path.isdir(path):
        print(f"[INFO] Creating directory {path}")
        os.mkdir(path)
    else:
        print(f"[INFO] Directory {path} already exists")

def CH(path: str):
    print(f"[INFO] Changing directory to {path}")
    os.chdir(path)

def RM(*paths: str):
    for path in paths:
        call_and_catch(["rm", "-rf", path], "Could not remove path!")

def TOUCH(path: str):
    call_and_catch(["touch", path], "Could not touch file!")

def build_alpakabench():
    # Build project
    MKDIR("build")
    CH("build")
    if not os.path.exists("../alpaka_built.txt"):
        call_and_catch(CMAKE.split(), "CMake failed!")
    else:
        print("[INFO] Alpaka already configured")
    call_and_catch(["ninja"], "Ninja failed!")
    CH("..")
    TOUCH("alpaka_built.txt")

def run_alpakabench():
    build_alpakabench()
    # Run benchmark, skip if already done
    output = ""
    if os.path.exists(ALPAKA_BENCH_FILE):
        with open(ALPAKA_BENCH_FILE, "r") as f:
            output = f.read()
    else:
        output = call_and_catch(
            ["./build/example/SGEMM/10_bench"],
            "Alpaka Execution failed!",
            print_output=True
        )
        with open(ALPAKA_BENCH_FILE, "w") as f:
            f.write(output)
    return output

def bring_to_standard(kid: int, output: str) -> str:
    # Running kernel 0 on device 0.
    # Max size: 4096
    # dimensions(m=n=k) 128, alpha: 0.5, beta: 3
    # Average elapsed time: (0.000503) s, performance: (    8.3) GFLOPS. size: (128).
    # dimensions(m=n=k) 256, alpha: 0.5, beta: 3
    # Average elapsed time: (0.000146) s, performance: (  230.6) GFLOPS. size: (256).
    # ...
    # want: Blog;<kernel>;{m,n};{n,k};{m,k};<time in ns>;<time in ns>;<timeout>
    assert kid < AlgoKind.CudaLike, "Unknown kernel"
    kernel = ["cuBLAS", "Naive", "GMemCoalesced", "SMemCaching"][kid]
    new_output = ""
    lines = output.split("\n")
    descr, lines = lines[0], lines[1:]
    assert descr == f"Running kernel {kid} on device 0."
    size, lines = lines[0], lines[1:]
    assert size.startswith("Max size:")
    _, lines = lines[-1], lines[:-1] # No one cares about empty lines
    for i in range(0, len(lines), 2):
        spec = lines[i].split(" ")
        size = spec[1]
        size = int(size[:-1])
        size_vecs = f"{{{size},{size}}}"
        time = lines[i+1].split(" ")
        elapsed = time[3][1:-1]
        elapsed_ns = float(elapsed) * 1e9
        elapsed_ns = int(elapsed_ns)
        new_output += f"Blog;GpuCuda;{kernel};{size_vecs};{size_vecs};{size_vecs};{elapsed_ns};{elapsed_ns};\n"
    return new_output

def build_blogbench():
    GIT_URL = "https://github.com/siboehm/SGEMM_CUDA.git"
    if not os.path.isdir("SGEMM_CUDA"):
        call_and_catch(["git", "clone", GIT_URL], "Could not clone repository!")
    else:
        print("[INFO] Repository already cloned")
    CH("SGEMM_CUDA")
    MKDIR("build")
    if not os.path.exists("../blog_built.txt"):
        lines = []
        with open("CMakeLists.txt", "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "set(CUDA_COMPUTE_CAPABILITY" in line:
                ccc = 89
                print(f"[INFO] CMakelists.txt:{i}: Patch: Update CMake config with `set(CUDA_COMPUTE_CAPABILITY {ccc})`")
                lines[i] = f"set(CUDA_COMPUTE_CAPABILITY {ccc})\n"
                break
        with open("CMakeLists.txt", "w") as f:
            f.writelines(lines)
        with open("sgemm.cu", "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "std::vector<int> SIZE" in line:
                step = 16
                sizes = ""
                j = 32
                while j <= 20000:
                    if j % 10 == 0:
                        step *= 2
                    sizes += f"{j},"
                    j += step
                sizes = sizes[:-1]
                print(f"[INFO] sgemm.cu:{i}: Patch: Replace with custom sizes `std::vector<int> SIZE = {{<sizes>}};`")
                lines[i] = f"std::vector<int> SIZE = {{{sizes}}};\n"
            if "if (!verify_matrix" in line:
                print(f"[INFO] sgemm.cu:{i}: Patch: Overwrite verification with `if (false) {{`")
                lines[i] = "if (false) {\n"
            if "int repeat_times = 50;" in line:
                print(f"[INFO] sgemm.cu:{i}: Patch: Speed up benchmarking with `int repeat_times = 20;`")
                lines[i] = "int repeat_times = 20;\n"
        print("[NOTE] All patches can be reverted by running `git restore .` in the SGEMM_CUDA directory")
        with open("sgemm.cu", "w") as f:
            f.writelines(lines)
    CH("build")
    if not os.path.exists("../../blog_built.txt"):
        call_and_catch(["cmake", "..", "-G", "Ninja"], "CMake failed!")
    else:
        print("[INFO] Blog already configured")
    call_and_catch(["ninja"], "Ninja failed!")
    CH("..")
    CH("..")
    TOUCH("blog_built.txt")

def run_blogbench():
    build_blogbench()
    output = ""
    if os.path.exists(BLOG_BENCH_FILE):
        with open(BLOG_BENCH_FILE, "r") as f:
            output = f.read()
    else:
        kernels = [0, 1, 2, 3]
        # DEVICE=<device_id> ./sgemm <kernel number>
        for kernel in kernels:
            step = call_and_catch(
                ["./SGEMM_CUDA/build/sgemm", str(kernel)],
                "Blog Execution failed!",
                env={"DEVICE": "0"},
                print_output=True
            )
            output += bring_to_standard(kernel, step)
        with open(BLOG_BENCH_FILE, "w") as f:
            f.write(output)
    return output

from dataclasses import dataclass
from enum import IntEnum

class ImplKind(IntEnum):
    Alpaka = 0
    Blog = 1

class AccKind(IntEnum):
    Cpu = 0
    Omp = 1
    Cuda = 2

class AlgoKind(IntEnum):
    CuBLAS = 0
    Naive = 1
    GMemCoalesced = 2
    SMemCaching = 3
    CudaLike = 4

@dataclass
class Entry:
    impl: ImplKind
    acc: AccKind
    algo: AlgoKind
    in1: Tuple[int, int]
    in2: Tuple[int, int]
    out: Tuple[int, int]
    warmup_time: int
    actual_time: int
    gflops: float

def process(bench: str, output: str) -> List[Entry]:
    print(f"[INFO] Processing output from {bench}")
    lines = output.split("\n")
    entries = []
    for line in lines:
        elements = line.split(";")
        if len(elements) < 8:
            assert len(elements) == 1 and elements[0] == "", f"Unknown line: {line}"
            continue
        impl = ImplKind.Alpaka if elements[0] == "Alpaka" else ImplKind.Blog
        acc = AccKind.Cpu
        if "CpuSingle" in elements[1]:
            acc = AccKind.Cpu
        elif "CpuOmpBlocks" in elements[1]:
            acc = AccKind.Omp
        elif "Cuda" in elements[1]:
            acc = AccKind.Cuda
        else:
            assert False, f"Unknown accelerator: {elements[1]}"
        algo = AlgoKind.Naive
        if "Naive" in elements[2]:
            algo = AlgoKind.Naive
        elif "GMemCoalesced" in elements[2]:
            algo = AlgoKind.GMemCoalesced
        elif "SMemCaching" in elements[2]:
            algo = AlgoKind.SMemCaching
        elif "cuBLAS" in elements[2]:
            algo = AlgoKind.CuBLAS
        elif "CudaLike" in elements[2]:
            algo = AlgoKind.CudaLike
        else:
            assert False, f"Unknown algorithm: {elements[2]}"
        in1 = tuple(map(int, elements[3][1:-1].split(",")))
        in2 = tuple(map(int, elements[4][1:-1].split(",")))
        out = tuple(map(int, elements[5][1:-1].split(",")))
        warmup_time = int(elements[6])
        actual_time = int(elements[7])
        # 1 GFLOPS = 10^9 FLOPS
        # 1 second = 10^9 ns
        # No need to divide by 10^9, since we are already in ns
        flops = in1[0] * in1[1] * in2[1] * 2
        assert actual_time != 0, f"Invalid entry: {line}"
        gflops = flops / actual_time if actual_time > 0 else 0
        entry = Entry(impl, acc, algo, in1, in2, out, warmup_time, actual_time, gflops)
        entries.append(entry)
    return entries

Filter = Callable[[List[Entry]], List[Entry]]
def plot_benchmarks(entries: Dict[str, List[Entry]]):
    import matplotlib.pyplot as plt
    import numpy as np
    def plot_wrapper(path: str, ylabel: str, filter_x: Filter, filter_y: Filter, filter_key=lambda _: True, xscale="linear", yscale="linear"):
        plot_thing(path, ylabel, filter_x, filter_y, filter_key, cublas=False, xscale=xscale, yscale=yscale)
        plot_thing(path, ylabel, filter_x, filter_y, filter_key, cublas=True, xscale=xscale, yscale=yscale)
    def plot_thing(path: str, ylabel: str, filter_x: Filter, filter_y: Filter, filter_key=lambda _: True, cublas: bool = False, xscale: str = "linear", yscale: str = "linear"):
        fig, ax = plt.subplots()
        fig.set_size_inches(40, 20)
        ax.set_yscale(yscale)
        FS = 20
        plt.xticks(fontsize=FS)
        plt.yticks(fontsize=FS)
        ax.set_ylabel(ylabel, fontsize=FS)
        ax.set_xlabel("Matrix Size (MiB)", fontsize=FS)
        ax.set_title(f"{ylabel} vs Matrix Size (MiB)", fontsize=FS)
        colors = plt.cm.tab20(np.linspace(0, 1, len(entries)))
        for i, (key, entry_list) in enumerate(entries.items()):
            if not filter_key(key):
                continue
            if not cublas and "CuBLAS" in key:
                continue
            xs = filter_x([entry for entry in entry_list if entry.actual_time > 0])
            ys = filter_y([entry for entry in entry_list if entry.actual_time > 0])
            xs = [x * x * 4 / 1024 / 1024 for x in xs]
            if len(xs) == 0 or len(ys) == 0:
                continue
            color = colors[i]
            ax.plot(xs, ys, color=color, label=key, marker="o", markersize=4, linewidth=2)
        ax.legend(prop={"size": FS})
        save_path = f"benchmarks_{path}.png"
        if cublas:
            save_path = f"benchmarks_{path}_cublas.png"
        print(f"[INFO] Saving plot to {save_path}")
        plt.savefig(save_path)
    MKDIR("resources")
    CH("resources")
    plot_wrapper("runtime", "Execution Time (s)",
        lambda x: [entry.out[0] for entry in x],
        lambda x: [entry.actual_time / 1e9 for entry in x],
    )
    plot_wrapper("gflops", "GFLOPS",
        lambda x: [entry.out[0] for entry in x],
        lambda x: [entry.gflops for entry in x]
    )
    plot_wrapper("gflops_log", "GFLOPS",
        lambda x: [entry.out[0] for entry in x],
        lambda x: [entry.gflops for entry in x],
        yscale="log"
    )
    plot_wrapper("cuda", "GFLOPS",
        lambda x: [entry.out[0] for entry in x if entry.acc == AccKind.Cuda],
        lambda x: [entry.gflops for entry in x if entry.acc == AccKind.Cuda],
        lambda x: "Cuda" in x
    )
    plot_wrapper("cuda_log", "GFLOPS",
        lambda x: [entry.out[0] for entry in x if entry.acc == AccKind.Cuda],
        lambda x: [entry.gflops for entry in x if entry.acc == AccKind.Cuda],
        lambda x: "Cuda" in x,
        yscale="log"
    )
    plot_wrapper("cuda_runtime", "Execution Time (s)",
        lambda x: [entry.out[0] for entry in x if entry.acc == AccKind.Cuda],
        lambda x: [entry.actual_time / 1e9 for entry in x if entry.acc == AccKind.Cuda],
        lambda x: "Cuda" in x
    )
    plot_thing("naive", "GFLOPS",
        lambda x: [entry.out[0] for entry in x if entry.algo == AlgoKind.Naive],
        lambda x: [entry.gflops for entry in x if entry.algo == AlgoKind.Naive],
    )
    plot_thing("gmem", "GFLOPS",
        lambda x: [entry.out[0] for entry in x if entry.algo == AlgoKind.GMemCoalesced],
        lambda x: [entry.gflops for entry in x if entry.algo == AlgoKind.GMemCoalesced],
    )
    plot_thing("smem", "GFLOPS",
        lambda x: [entry.out[0] for entry in x if entry.algo == AlgoKind.SMemCaching],
        lambda x: [entry.gflops for entry in x if entry.algo == AlgoKind.SMemCaching],
    )
    plot_thing("smem_gmem", "GFLOPS",
        lambda x: [entry.out[0] for entry in x if entry.algo == AlgoKind.SMemCaching or entry.algo == AlgoKind.GMemCoalesced],
        lambda x: [entry.gflops for entry in x if entry.algo == AlgoKind.SMemCaching or entry.algo == AlgoKind.GMemCoalesced],
    )
    plot_thing("cpu", "GFLOPS",
        lambda x: [entry.out[0] for entry in x if entry.acc == AccKind.Cpu],
        lambda x: [entry.gflops for entry in x if entry.acc == AccKind.Cpu],
        lambda x: "Cpu" in x
    )
    plot_thing("cpu_log", "GFLOPS",
        lambda x: [entry.out[0] for entry in x if entry.acc == AccKind.Cpu],
        lambda x: [entry.gflops for entry in x if entry.acc == AccKind.Cpu],
        lambda x: "Cpu" in x,
        yscale="log"
    )
    plot_thing("cpu_omp", "GFLOPS",
        lambda x: [entry.out[0] for entry in x if entry.acc == AccKind.Cpu or entry.acc == AccKind.Omp],
        lambda x: [entry.gflops for entry in x if entry.acc == AccKind.Cpu or entry.acc == AccKind.Omp],
        lambda x: "Cpu" in x or "Omp" in x
    )
    CH("..")

def main():
    if sys.prefix == sys.base_prefix:
        print("\033[93m[WARNING] This script should be run in a virtual environment\033[0m")
        initialize_venv()
        print("[INFO] Please run `./.venv/bin/python3 measure.py` from now on")
        exit(0)
    if len(sys.argv) == 1:
        print("Usage: measure.py <build|run|plot|clean>")
        print("- build:   Build benchmarks")
        print("- run:     Run benchmarks and save output")
        print("- plot:    Plot benchmarks")
        print("- clean:   Clean up all build files")
        exit(1)
    mode = sys.argv[1]
    match mode:
        case "build":
            build_alpakabench()
            build_blogbench()
            exit(0)
        case "run":
            run_alpakabench()
            run_blogbench()
        case "plot":
            alpaka_output = run_alpakabench()
            blog_output = run_blogbench()
            alpaka_entries = process("alpaka", alpaka_output)
            blog_entries = process("blog", blog_output)
            all_entries = alpaka_entries + blog_entries
            all_entries.sort(key=lambda x: x.out)
            entries = {}
            for entry in all_entries:
                key = f"{entry.impl.name}_{entry.acc.name}_{entry.algo.name}"
                if key not in entries:
                    entries[key] = []
                entries[key].append(entry)
            plot_benchmarks(entries)
        case "clean":
            RM(".venv")
            RM("build")
            RM("SGEMM_CUDA")
            RM("alpaka_built.txt", "blog_built.txt")
            if len(sys.argv) > 2 and sys.argv[2] == "all":
                print("Are you sure you want to remove the benchmark files?")
                confirm = input(f"This will remove {ALPAKA_BENCH_FILE} and {BLOG_BENCH_FILE} [y/N] ")
                if confirm.lower() == "y":
                    RM(ALPAKA_BENCH_FILE, BLOG_BENCH_FILE)
            exit(0)
        case _:
            print("Unknown command")
            exit(1)

def initialize_venv():
    if not os.path.isdir(".venv"):
        print("[INFO] Initializing virtual environment")
        call_and_catch(["python3", "-m", "venv", ".venv"], "Could not create virtual environment!")
    else:
        print("[INFO] Virtual environment already exists")
    call_and_catch(["./.venv/bin/python3", "-m", "pip", "install", "matplotlib"], "Could not install matplotlib!", print_output=True)
    call_and_catch(["./.venv/bin/python3", "-m", "pip", "install", "numpy"], "Could not install numpy!", print_output=True)

if __name__ == "__main__":
    main()