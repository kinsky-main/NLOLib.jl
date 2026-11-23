# API reference

Every exported name of `NonlinearOptics`.

## Propagation

```@docs
propagate
PulseSpec
ExecutionOptions
StorageOptions
PropagationResult
StepHistory
StorageResult
ProgressInfo
PropagationAbortedError
NLOLibError
```

## Library and runtime limits

```@docs
load
loaded_library_path
query_runtime_limits
RuntimeLimits
storage_is_available
tensor_record_view
```

## Logging and progress

```@docs
set_log_file
set_log_buffer
clear_log_buffer
read_log_buffer
set_log_level
set_progress_options
set_progress_stream
```

## Performance counters

```@docs
perf_profile_set_enabled
perf_profile_is_enabled
perf_profile_reset
perf_profile_read
PerfProfileSnapshot
```
