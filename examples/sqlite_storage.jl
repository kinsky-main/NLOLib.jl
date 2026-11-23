# Stream records to a SQLite database instead of returning them.
#
# The one shared example that sets return_records = false: the records go to
# the snapshot store in chunks of chunk_records and are counted back from it.

using NonlinearOpticsExamples
pushfirst!(LOAD_PATH, package_root_from(@__FILE__))

using DBInterface
using NonlinearOptics
using SQLite

# nlolib-example-parameters: begin
nt = 128
dt = 0.0078125
pulse_period = 1.0
z_final = 0.2
num_records = 24
chunk_records = 6
sqlite_max_bytes = 53687091200
# nlolib-example-parameters: end

count_rows(db, table) = first(DBInterface.execute(db, "SELECT COUNT(*) AS n FROM $(table)")).n

function main(argv = ARGS)
    args = parse_example_args("sqlite_storage", "SQLite snapshot chunking demo with DB-backed run/replot.", argv)
    db = ExampleRunDB(args[:db_path])
    example_name = "sqlite_storage"
    case_key = "default"
    storage_is_available() ||
        error("This nlolib build does not include SQLite storage support. Reconfigure/build with SQLite available.")

    t = [(i - nt / 2.0) * dt * 8.0 for i in 0:nt - 1]
    pulse = PulseSpec(
        samples = ComplexF64.(exp.(-(t .^ 2))),
        delta_time = dt,
        pulse_period = pulse_period,
        frequency_grid = ComplexF64.(0:nt - 1),
    )

    if args[:replot]
        run_group = resolve_replot_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        run_id = load_case(db; example_name = example_name, run_group = run_group, case_key = case_key).run_id
    else
        run_group = begin_group(db, example_name, isempty(args[:run_group]) ? nothing : args[:run_group])
        storage = storage_kwargs(db; example_name = example_name, run_group = run_group, case_key = case_key,
            chunk_records = chunk_records, sqlite_max_bytes = sqlite_max_bytes)
        result = propagate(
            pulse,
            (A, w) -> 0.0,
            (A, I) -> 0.0;
            propagation_distance = z_final,
            num_records = num_records,
            preset = "balanced",
            return_records = false,
            exec_options = ExecutionOptions(backend_type = "auto"),
            storage = StorageOptions(; storage...),
        )
        stored = result.storage_result
        run_id = stored.run_id
        save_case!(db; example_name = example_name, run_group = run_group, case_key = case_key, run_id = run_id,
            meta = Dict("num_records" => num_records))
        println("storage summary: captured=$(stored.records_captured) spilled=$(stored.records_spilled) " *
                "chunks=$(stored.chunks_written) truncated=$(Int(stored.truncated)) db_size_bytes=$(stored.db_size_bytes)")
        report_metric("records captured", stored.records_captured)
        report_metric("chunks written", stored.chunks_written)
        report_metric("db size bytes", stored.db_size_bytes)
    end
    println("sqlite db: $(db.db_path)")
    println("run_id: $(run_id)")
    println("run_group: $(run_group)")

    conn = SQLite.DB(db.db_path)
    try
        println("db rows: runs=$(count_rows(conn, "io_runs")) chunks=$(count_rows(conn, "io_record_chunks")) " *
                "system_configs=$(count_rows(conn, "io_system_config")) sim_configs=$(count_rows(conn, "io_sim_config"))")
        for row in DBInterface.execute(conn,
                "SELECT records_written, chunks_written, truncated FROM io_runs WHERE run_id=?;", (run_id,))
            println("run row: records_written=$(row.records_written) chunks_written=$(row.chunks_written) truncated=$(row.truncated)")
        end
    finally
        DBInterface.close!(conn)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
