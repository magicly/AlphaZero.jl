using CUDA
using Flux
using Revise, BenchmarkTools, OhMyREPL
using ThreadPools
using Base.Threads
using AlphaZero
using Setfield

function one_game(idx, steps, sims, infer_channel)
    ans_channel = Channel()

    exp = Examples.experiments["connect-four"];
    obs = GI.current_state(GI.init(exp.gspec))

    for i in 1:steps 
        for _ in 1:sims # one step needs multiple mcts simulations
            t1 = time()
            # obs = step((7, 6, 3), timeout=1e-4) # TODO timeout长了跑不慢cpu？！！ 太短的话threads不能太多， 2就够了, 1e-6, 4就太大了
            # obs = GI.current_state(GI.init(exp.gspec))
            while time() - t1 < 1e-5 # TODO 太快了反而不行， infer ratio低， wait_time高?! data race?
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
function one_game2(idx, steps, sims, player)
    # ans_channel = Channel()

    exp = Examples.experiments["connect-four"];
    obs = GI.current_state(GI.init(exp.gspec))


    # trace = play_game(exp.gspec, player, flip_probability=p.flip_probability)
    trace = play_game(exp.gspec, player, flip_probability=0.)
    return trace

    for i in 1:steps 
        for _ in 1:sims # one step needs multiple mcts simulations
            t1 = time()
            # obs = step((7, 6, 3), timeout=1e-4) # TODO timeout长了跑不慢cpu？！！ 太短的话threads不能太多， 2就够了, 1e-6, 4就太大了
            # obs = GI.current_state(GI.init(exp.gspec))
            while time() - t1 < 1e-5 # TODO 太快了反而不行， infer ratio低， wait_time高?! data race?
            end # TODO gpu 占用不影响avg time！ 这才是对的
            t2 = time()
            # println("====obs cost: $(t2 - t1)") # <1e-6
            game = GI.init(exp.gspec)
            actions = GI.available_actions(game)
            state = GI.current_state(game)
            π, _ = player.network(state)

            # 随机不影响infer
	        # obs = (board=SMatrix{7,6}(rand(UInt8, (7, 6))), curplayer=1)
            # player.network(obs)
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



# Evaluate a single neural network for a one-player game (params::ArenaParams)
function evaluate_network(gspec, net, params)
  make_oracles() = Network.copy(net, on_gpu=params.sim.use_gpu, test_mode=true)
  simulator = Simulator(make_oracles, record_trace) do oracle
    # MctsPlayer(gspec, oracle, params.mcts)
    NetworkPlayer(oracle)
  end
  samples = simulate(
    simulator, gspec, params.sim,
    # game_simulated=(() -> println("game_simulated!")))
    game_simulated=(() -> ()))
  return sum(length(s.trace.rewards) for s in samples)
#   return length(samples)
end

function workers3(gspec, net, params, num_workers, N_GAMES, infer_channel)
    # reqc = launch_inference_server(o; kwargs...) # infer channel
    reqc = infer_channel
    make() = Batchifier.BatchedOracle(reqc)
    send_done!(reqc) = () -> Batchifier.client_done!(reqc)
    done_n = Atomic{Int}()

    println("in workers3, $num_workers, $N_GAMES")
    samples = Util.mapreduce(1:N_GAMES, num_workers, vcat, []) do
        oracle = make()
        function simulate_game(idx)
            # player = NetworkPlayer(oracle)
            player = MctsPlayer(gspec, oracle, params.mcts)
            return one_game2(idx, STEPS, SIMS, player)
        end
        function done()
            atomic_add!(done_n, 1)
            println("=============done! $done_n")
            send_done!(reqc)
            # put!(infer_channel, :done)
        end
        # return (process = simulate_game, terminate = done)
        return (process = simulate_game, terminate = send_done!(reqc))
    end
    1
    # return sum(length(s.rewards) for s in samples)
end

batch_size = 64 * 2
STEPS = 20
SIMS = 60

function go(ngames=500, worker_idx=1)# TODO 500时候把cpu跑不慢，task调度问题？！
    batch_size = 64 * 8
    num_workers = batch_size * 2

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


    infer_channel = Batchifier.launch_server(; num_workers, batch_size) do batch
        Network.evaluate_batch(net, batch)
    end


    if worker_idx == 1
        workers(num_workers, ngames, infer_channel)
    elseif worker_idx == 2
        workers2(num_workers, ngames, infer_channel)
    elseif worker_idx == 3
        params = exp.params.self_play
        params = @set params.sim.num_games = ngames
        params = @set params.mcts.num_iters_per_turn = SIMS
        workers3(exp.gspec, net, params, num_workers, ngames, infer_channel)
    else
        params = exp.params.self_play
        params = @set params.sim.num_games = ngames
        params = @set params.mcts.num_iters_per_turn = SIMS
        evaluate_network(exp.gspec, net, params)
    end
end

t1 = time()
# @time go(1000)
println("cost: $(time() - t1)")