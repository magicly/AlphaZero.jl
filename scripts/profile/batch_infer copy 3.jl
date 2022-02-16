using CUDA
using Flux
using Revise, BenchmarkTools, OhMyREPL
using ThreadPools
using Base.Threads
using AlphaZero


function infer_server(net; num_workers, batch_size=1024)
    println("in infer_server: $num_workers, $batch_size")
    channel = Channel(batch_size * 2) # Channel buffer大一点？没影响

    old_batch_size = batch_size

    ThreadPools.@tspawnat 1 begin
        println("in infer server $(Threads.threadid())")

        num_active = num_workers
        pending = []

        wait_batch_time = 0
        infer_time = 0
        put_time = 0
        total_time = 0

        t0 = time()
        total = 0
        while true
            req, t = @timed take!(channel)
            if t > 1e-5
                # println("take! cost too much: $(t22 - t11)")
            end
            if req.query == :done
                # println("game $(req.idx) is done!")
                num_active -= 1
                if num_active < batch_size
                    batch_size = num_active
                end
                if batch_size < old_batch_size
                    # println("$old_batch_size just remain $batch_size $(length(pending))!")
                end
            else
                push!(pending, req)
                # println("pendings: $(length(pending))")
            end

            @assert length(pending) <= num_active
            @assert batch_size <= num_active

            if batch_size <= 0 && length(pending) == 0
                println("infer over! restart")
                break;
                num_active = num_workers
                batch_size = old_batch_size
            end

            if length(pending) >= batch_size && length(pending) > 0
                if batch_size < old_batch_size
                    # println("========$batch_size====$(length(pending))")
                end
                t1 = time()
                wait_batch_time += t1 - t0

                t2 = time()
                batch = [p.query for p in pending]
                results = Network.evaluate_batch(net, batch)
                t3 = time()
                infer_time += t3 - t2

                for i in eachindex(pending)
                    # put!(pending[i].ans_channel, results[i]) # TODO put!占用10%
                    put!(pending[i].ans_channel, 1) # TODO put!占用10%
                end
                empty!(pending)
                put_time += time() - t3

                total_time += time() - t0
                t0 = time()
                total += 1
                if total % 100 == 0
                    println("batch: $total, $batch_size, wait_batch_time: $wait_batch_time, infer_time: $infer_time, total_time: $total_time, put_time: $put_time")
                    println("infer ratio: $(infer_time / total_time)")
                    println("==========total avg infer time: $(infer_time / total / batch_size)")
                end
            end
        end
    end
    channel
end
    
function one_game(idx, steps, sims, infer_channel)
    ans_channel = Channel()

    exp = Examples.experiments["connect-four"];
    obs = GI.current_state(GI.init(exp.gspec))

    for i in 1:steps 
        for _ in 1:sims # one step needs multiple mcts simulations
            t1 = time()
            # obs = step((7, 6, 3), timeout=1e-4) # TODO timeout长了跑不慢cpu？！！ 太短的话threads不能太多， 2就够了, 1e-6, 4就太大了
            # obs = GI.current_state(GI.init(exp.gspec))
            while time() - t1 < 1e-6 # TODO 太快了反而不行， infer ratio低， wait_time高?! data race?
            end # TODO gpu 占用不影响avg time！ 这才是对的
            t2 = time()
            # println("====obs cost: $(t2 - t1)") # <1e-6
            req = (query = obs, answer_channel = ans_channel)
            t1 = time()
            put!(infer_channel, req)
            t2 = time()
            # println("put cost: $(t2 - t1)")
            t3 = time()
            y = take!(ans_channel)
            t4 = time()
            # if i % 10 == 0
            # println("play total time: $(t4 - t1), think time: $(t4 - t2), think ratio: $((t4 - t2) / (t4 - t1))")
            # end
            # println("result: $idx, $y")
        end
    end
    return idx
end
    

function workers2(num_workers, N_GAMES, infer_channel)
    println("in workers2, $num_workers, $N_GAMES")
    return Util.mapreduce(1:N_GAMES, num_workers, vcat, []) do
        function simulate_game(idx)
            return one_game(idx, STEPS, SIMS, infer_channel)
        end
        function done()
            # put!(infer_channel, (query = :done, idx = 1))
            put!(infer_channel, :done)
        end
        return (process = simulate_game, terminate = done)
    end
end
function workers(num_workers, N_GAMES, infer_channel)
    println("in workers, $num_workers, $N_GAMES")
    next = 1
    lock = ReentrantLock()

    tasks = []
    results = []
    nbg = Threads.nthreads() - 1
    nbg = 2
    for i in 1:num_workers
        tid = 2 + (i - 1) % nbg
        task = ThreadPools.@tspawnat tid begin
            # println("start new task at thread: $tid")
            while true
                Base.lock(lock)
                if next > N_GAMES
                    Base.unlock(lock)
                    break
                end
                idx = next
                next += 1
                Base.unlock(lock)

                r = one_game(idx, STEPS, SIMS, infer_channel)
                Base.lock(lock)
                push!(results, r)
                Base.unlock(lock)
            end
            # put!(infer_channel, (query = :done, idx = i))
            put!(infer_channel, :done)
        end
        push!(tasks, task)
    end
    wait.(tasks)
    println("$next games done!  $N_GAMES")
    results
end


batch_size = 64 * 2
STEPS = 20
SIMS = 60

function go(ngames=500, worker_idx=1)# TODO 500时候把cpu跑不慢，task调度问题？！
    batch_size = 64 * 8
    num_workers = batch_size * 2
    # num_workers = round(Int, batch_size * 1.2)

    # num_workers = ngames # 太多task会不会造成并发影响， 太多了wait反而高！ 但是不影响infer时间
    # 如果play时间特别短， 没必要开大量thread， 造成并发竞争， infer ratio反而低， cpu占用还高
    # play太长， infer跟不上， avg infer time要长一些， 1.8 -> 2.5
    # task多不怕， thread少控制住就行？ 还是有一定影响~20%

    # rand timeout 长， 需要开更多thread， gpu占用波动很大， 但infer ratio挺高的？！ avg infer time也稍高， 1.3 vs 1.9
    # cpu占用也有波动， 所以还是task调度问题？

    exp = Examples.experiments["connect-four"];
    params = exp.netparams
    net = exp.mknet(exp.gspec, params)
    net = Network.copy(net; on_gpu=true, test_mode=true)
    state = GI.current_state(GI.init(exp.gspec))
    batch = [state for _ in 1:batch_size]
    Network.evaluate_batch(net, batch) # To compile everything


    # infer_channel = infer_server(net, num_workers=num_workers, batch_size=batch_size)
    # infer_channel = Batchifier.launch_server(batch -> Network.evaluate_batch(net, batch), num_workers, batch_size)
    infer_channel = Batchifier.launch_server(; num_workers, batch_size) do batch
        Network.evaluate_batch(net, batch)
    end
    if worker_idx == 1
        workers(num_workers, ngames, infer_channel)
    else
        workers2(num_workers, ngames, infer_channel)
    end
end

t1 = time()
# @time go(1000)
println("cost: $(time() - t1)")