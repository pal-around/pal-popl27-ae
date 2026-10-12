# Tuning a random list generator with Dice (Alea.jl): learn one stop probability per
# position so the length distribution becomes uniform over 0..N.
#
#   alea-jl natlist.jl [--n 100] [--epochs 1000] [--lr 0.3] [--output-csv path] [--show-dist]
#
# The loss is the exact KL divergence, computed in log space, and training is gradient
# descent.
# Times reported separately: building the BDD and expanding its weighted model count
# happen once, before training, and the first epoch includes Julia's own JIT.
using Dates
using Dice

# Define inductive datatype for nat lists
@inductive DistNatList DistNil() DistCons(DistUInt32, DistNatList)

function length(l::DistNatList)
    @match l [
        DistNil() -> DistUInt32(0),
        DistCons(x, xs) -> DistUInt32(1) + length(xs),
    ]
end

# Our generator for nat lists: at each position, stop with a trainable probability.
function gen_list(size, elem)
    size == 0 && return DistNil()

    @dice_ite if flip(sigmoid(Var(size)))
        DistNil()
    else
        # The flips used in `uniform` have concrete weights, so their probabilities are
        # not learned, and the element is not read by `length`.
        DistCons(uniform(DistUInt32, 0, elem), gen_list(size-1, elem))
    end
end

EPOCHS = 1000
LR = 0.3
INIT_SIZE_DEFAULT = 100
ELEM_DEFAULT = 100

function parse_args(args)
    output_csv = nothing
    epochs = EPOCHS
    lr = LR
    n = INIT_SIZE_DEFAULT
    elem = ELEM_DEFAULT
    show_dist = false
    i = 1
    while i <= Base.length(args)
        if args[i] == "--output-csv"
            i == Base.length(args) && error("missing value for --output-csv")
            output_csv = args[i + 1]
            i += 2
        elseif args[i] == "--epochs"
            i == Base.length(args) && error("missing value for --epochs")
            epochs = parse(Int, args[i + 1])
            i += 2
        elseif args[i] == "--lr"
            i == Base.length(args) && error("missing value for --lr")
            lr = parse(Float64, args[i + 1])
            i += 2
        elseif args[i] == "--n"
            i == Base.length(args) && error("missing value for --n")
            n = parse(Int, args[i + 1])
            i += 2
        elseif args[i] == "--elem"
            i == Base.length(args) && error("missing value for --elem")
            elem = parse(Int, args[i + 1])
            i += 2
        elseif args[i] == "--show-dist"
            show_dist = true
            i += 1
        else
            error("unknown argument: $(args[i])")
        end
    end
    return (output_csv=output_csv, epochs=epochs, lr=lr, n=n, elem=elem, show_dist=show_dist)
end

function default_output_path(filename)
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    output_dir = joinpath(@__DIR__, timestamp)
    mkpath(output_dir)
    return joinpath(output_dir, filename)
end

function log_device_info()
    println("Julia device info:")
    println("  Device: CPU == $(Sys.CPU_NAME)")
    println("  Julia threads: $(Threads.nthreads()) / $(Sys.CPU_THREADS)")
    println("  Machine: $(Sys.MACHINE)")
    println("  Kernel: $(Sys.KERNEL)")
    println("  Julia version: $(VERSION)")
    println()
end

# Compiling the BDD and expanding its weighted model count into an arithmetic DAG
# happens once; only the epochs that follow are training.
function compile_loss(loss::ADNode)
    expander = Dice.LogPrExpander(Dice.WMC(Dice.BDDCompiler(Dice.bool_roots([loss]))))
    return Dice.expand_logprs(expander, loss)
end

function timed_train!(
    var_vals::Valuation,
    expanded_loss::ADNode;
    epochs::Integer,
    learning_rate::Real,
    append_last_loss=true,
    stop_if_inf_or_nan=true,
)
    curve = Any[]
    times = Float64[]
    train_start = time()
    prev_epoch_end = 0.0

    for _ in 1:epochs
        vals, derivs = Dice.differentiate(var_vals, Dice.Derivs(expanded_loss => learning_rate))
        push!(curve, vals[expanded_loss])
        push!(times, prev_epoch_end)

        if stop_if_inf_or_nan && (isinf(vals[expanded_loss]) || isnan(vals[expanded_loss]))
            return curve, times
        end

        for (adnode, d) in derivs
            if adnode isa Var
                var_vals[adnode] -= d
            end
        end

        prev_epoch_end = time() - train_start
    end

    if append_last_loss
        push!(curve, Dice.compute(var_vals, expanded_loss))
        push!(times, prev_epoch_end)
    end

    return curve, times
end

parsed = parse_args(ARGS)
csv_path = isnothing(parsed.output_csv) ? default_output_path("dice.csv") : parsed.output_csv
mkpath(dirname(csv_path))
log_device_info()
println("n: $(parsed.n)  elem: $(parsed.elem)  epochs: $(parsed.epochs)  lr: $(parsed.lr)")

list = gen_list(parsed.n, parsed.elem)
list_len = length(list)
var_vals = Valuation(Var(size) => 0 for size in 1:parsed.n)
desired_dist = [DistUInt32(i) => 1/(parsed.n + 1) for i in 0:parsed.n]

if parsed.show_dist
    println("Distribution before training:")
    display(pr_mixed(var_vals)(list_len))
    println()
end

t_compile_start = time()
expanded_loss = compile_loss(kl_divergence(desired_dist, list_len))
t_compile = time() - t_compile_start
println("BDD compile: $(t_compile)s")

t_train_start = time()
curve, times = timed_train!(var_vals, expanded_loss, epochs=parsed.epochs, learning_rate=parsed.lr)
t_train = time() - t_train_start

if parsed.show_dist
    println("Distribution over lengths after training:")
    display(pr_mixed(var_vals)(list_len))
    println()
end

println("Training: $(t_train)s  ($(t_train/parsed.epochs*1000)ms per epoch)")
println("Final loss: $(curve[end])")

# Save loss curve to CSV
open(csv_path, "w") do f
    println(f, "epoch,loss,time")
    for (i, (l, t)) in enumerate(zip(curve, times))
        println(f, "$(i-1),$l,$t")
    end
end
println("Saved loss curve to $csv_path")
