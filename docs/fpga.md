# Adding an FPGA backend

There are no supported FPGA boards or tested vendor toolchains in v0.1.0. Physical hardware tests, vendor emulation, functional RTL simulation, and FPGA performance measurements have not run. There is no performance estimator in this release. CPU and CUDA results must never be labeled FPGA results.

Implement the Go Backend interface to add a separately selected adapter. Return an explicit error for unsupported models or unavailable devices. Go goroutines coordinate host tasks. A board runs a compiled hardware design, which may be written in Verilog, SystemVerilog, VHDL, or a vendor-supported C/C++ high-level synthesis flow. No Go compiler or runtime changes are planned.

Before choosing a board, check the vendor's current supported driver, runtime, synthesis tools, operating systems, and license. Only claim support after building and testing on that exact target. A bitstream is specific to its board and interface. Do not distribute third-party bitstreams without permission.

An adapter must define a versioned input and output protocol, asset and step limits, numerical formats, scenario ID and random seed handling, parameter units, errors, and cancellation. The CPU reference is the comparison target. Document exact comparisons for integer formats or justified tolerances for floating point. Check bitstream SHA-256, interface version, board identity, runtime compatibility, and timeouts before loading hardware.

The example [configuration](../examples/fpga-v1.json) is a specification template, not an accepted run configuration. Its unavailable values deliberately prevent anyone mistaking it for working hardware support. A future adapter must validate every field. Never execute or load a user-specified image automatically during ordinary CPU tests.

Provide separate commands for Go unit tests, RTL simulation, vendor hardware emulation, and physical board tests when an adapter exists. Record board and host CPU, tool and driver versions, runtime, bitstream digest, clock, numeric format, batch size, seed, timeout, repeats, and commands. Report setup, transfers, kernel time, and full run time separately. Keep vendor build products under ignored build/report directories. Use containers only where the vendor supports hardware access and licensing.
