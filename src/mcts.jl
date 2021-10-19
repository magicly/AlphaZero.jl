"""
A generic, standalone implementation of Monte Carlo Tree Search.
It can be used on any game that implements `GameInterface`
and with any external oracle.

## Oracle Interface

An oracle can be any function or callable object.
  
   oracle(state)

evaluates a single state from the current player's perspective and returns 
a pair `(P, V)` where:

  - `P` is a probability vector on `GI.available_actions(GI.init(gspec, state))`
  - `V` is a scalar estimating the value or win probability for white.
"""
module MCTS

using Distributions: Categorical, Dirichlet

using ..AlphaZero: GI, Util

#####
##### Standard Oracles
#####

"""
    MCTS.RolloutOracle(game_spec::AbstractGameSpec, γ=1.) <: Function

This oracle estimates the value of a position by simulating a random game
from it (a rollout). Moreover, it puts a uniform prior on available actions.
Therefore, it can be used to implement the "vanilla" MCTS algorithm.
"""
struct RolloutOracle{GameSpec} <: Function
  gspec :: GameSpec
  gamma :: Float64
  RolloutOracle(gspec, γ=1.) = new{typeof(gspec)}(gspec, γ)
end

function rollout!(game, γ=1.)
  action = rand(GI.available_actions(game))
  GI.play!(game, action)
  wr = GI.white_reward(game)
  if GI.game_terminated(game)
    return wr
  else
    return wr + γ * rollout!(game, γ)
  end
end

function (r::RolloutOracle)(state)
  g = GI.init(r.gspec, state)
  wp = GI.white_playing(g)
  n = length(GI.available_actions(g))
  P = ones(n) ./ n
  wr = rollout!(g, r.gamma)
  V = wp ? wr : -wr
  return P, V
end

struct RandomOracle{GameSpec}
  gspec :: GameSpec
end

function (r::RandomOracle)(state)
  g = GI.init(r.gspec, state)
  n = length(GI.available_actions(g))
  P = ones(n) ./ n
  V = 0.
  return P, V
end

#####
##### State Statistics
#####

struct ActionStats
  P :: Float32 # Prior probability as given by the oracle
  W :: Float64 # Cumulated Q-value for the action (Q = W/N)
  N :: Int # Number of times the action has been visited
end

struct StateInfo
  stats :: Vector{ActionStats}
  Vest  :: Float32 # Value estimate given by the oracle
end

Ntot(b::StateInfo) = sum(s.N for s in b.stats)

#####
##### MCTS Environment
#####

"""
    MCTS.Env(game_spec::AbstractGameSpec, oracle; <keyword args>)

Create and initialize an MCTS environment with a given `oracle`.

## Keyword Arguments

  - `gamma=1.`: the reward discount factor
  - `cpuct=1.`: exploration constant in the UCT formula
  - `noise_ϵ=0., noise_α=1.`: parameters for the dirichlet exploration noise
     (see below)
  - `prior_temperature=1.`: temperature to apply to the oracle's output
     to get the prior probability vector used by MCTS.

## Dirichlet Noise

A naive way to ensure exploration during training is to adopt an ϵ-greedy
policy, playing a random move at every turn instead of using the policy
prescribed by [`MCTS.policy`](@ref) with probability ϵ.
The problem with this naive strategy is that it may lead the player to make
terrible moves at critical moments, thereby biasing the policy evaluation
mechanism.

A superior alternative is to add a random bias to the neural prior for the root
node during MCTS exploration: instead of considering the policy ``p`` output
by the neural network in the UCT formula, one uses ``(1-ϵ)p + ϵη`` where ``η``
is drawn once per call to [`MCTS.explore!`](@ref) from a Dirichlet distribution
of parameter ``α``.
"""
mutable struct Env{State, Oracle}
  # Store (nonterminal) state statistics assuming the white player is to play
  tree :: Dict{State, StateInfo}
  # External oracle to evaluate positions
  oracle :: Oracle
  # Parameters
  gamma :: Float64 # Discount factor
  cpuct :: Float64
  noise_ϵ :: Float64
  noise_α :: Float64
  prior_temperature :: Float64
  # Performance statistics
  total_simulations :: Int64
  total_nodes_traversed :: Int64
  # Game specification
  gspec :: GI.AbstractGameSpec

  function Env(gspec, oracle;
      gamma=1., cpuct=1., noise_ϵ=0., noise_α=1., prior_temperature=1.)
    S = GI.state_type(gspec)
    tree = Dict{S, StateInfo}()
    total_simulations = 0
    total_nodes_traversed = 0
    new{S, typeof(oracle)}(
      tree, oracle, gamma, cpuct, noise_ϵ, noise_α, prior_temperature,
      total_simulations, total_nodes_traversed, gspec)
  end
end

#####
##### Access and initialize state information
#####

function init_state_info(P, V, prior_temperature)
  P = Util.apply_temperature(P, prior_temperature)
  stats = [ActionStats(p, 0, 0) for p in P]
  return StateInfo(stats, V)
end

# Returns statistics for the current player, along with a boolean indicating
# whether or not a new node has been created.
function state_info(env, state)
  if haskey(env.tree, state)
    return (env.tree[state], false)
  else
    (P, V) = env.oracle(state)
    info = init_state_info(P, V, env.prior_temperature)
    env.tree[state] = info
    return (info, true)
  end
end

#####
##### Main algorithm
#####

function uct_scores(info::StateInfo, cpuct, ϵ, η)
  @assert iszero(ϵ) || length(η) == length(info.stats)
  sqrtNtot = sqrt(Ntot(info))
  return map(enumerate(info.stats)) do (i, a)
    Q = a.W / max(a.N, 1)
    P = iszero(ϵ) ? a.P : (1-ϵ) * a.P + ϵ * η[i]
    Q + cpuct * P * sqrtNtot / (a.N + 1)
  end
end

function update_state_info!(env, state, action_id, q)
  stats = env.tree[state].stats
  astats = stats[action_id]
  stats[action_id] = ActionStats(astats.P, astats.W + q, astats.N + 1)
end

# Run a single MCTS simulation, updating the statistics of all traversed states.
# Return the estimated Q-value for the current player.
# Modifies the state of the game environment.
function run_simulation!(env::Env, game; η, root=true)
  if GI.game_terminated(game)
    return 0.
  else
    state = GI.current_state(game)
    actions = GI.available_actions(game)
    info, new_node = state_info(env, state)
    if new_node
      return info.Vest
    else
      ϵ = root ? env.noise_ϵ : 0.
      scores = uct_scores(info, env.cpuct, ϵ, η)
      action_id = argmax(scores)
      action = actions[action_id]
      wp = GI.white_playing(game)
      GI.play!(game, action)
      wr = GI.white_reward(game)
      r = wp ? wr : -wr
      pswitch = wp != GI.white_playing(game)
      qnext = run_simulation!(env, game, η=η, root=false)
      qnext = pswitch ? -qnext : qnext
      q = r + env.gamma * qnext
      update_state_info!(env, state, action_id, q)
      env.total_nodes_traversed += 1
      return q
    end
  end
end

function dirichlet_noise(game, α)
  actions = GI.available_actions(game)
  n = length(actions)
  return rand(Dirichlet(n, α))
end

"""
    MCTS.explore!(env, game, nsims)

Run `nsims` MCTS simulations from the current state.
"""
function explore0!(env::Env, game, nsims)
  η = dirichlet_noise(game, env.noise_α)
  for i in 1:nsims
    env.total_simulations += 1
    run_simulation!(env, GI.clone(game), η=η)
  end
end

"""
    MCTS.policy(env, game)

Return the recommended stochastic policy on the current state.

A call to this function must always be preceded by
a call to [`MCTS.explore!`](@ref).
"""
function policy(env::Env, game)
  actions = GI.available_actions(game)
  state = GI.current_state(game)
  info =
    try env.tree[state]
    catch e
      if isa(e, KeyError)
        error("MCTS.explore! must be called before MCTS.policy")
      else
        rethrow(e)
      end
    end
  Ntot = sum(a.N for a in info.stats)
  π = [a.N / Ntot for a in info.stats]
  π ./= sum(π)
  return actions, π
end

mutable struct Node{Action}
  parent::Union{Node,Nothing}
  children::Vector{Node}

  action::Action
  n::Int
  P::Float32 # Prior probability as given by the oracle
  reward::Float32

  function Node(parent, action, P=1.0, reward=0.)
    new{typeof(action)}(parent, [], action, 0, P, reward)
  end
end
function expand!(env::Env, node::Node, game)
    state = GI.current_state(game)
    P0, V = env.oracle(state)
    P = Util.apply_temperature(P0, env.prior_temperature)
    actions = GI.available_actions(game)

    for (action, p) in zip(actions, P)
      child_node = Node(node, action, p)
      push!(node.children,  child_node)
    end
    V
end
function backup!(node::Node, reward)
    if node.parent !== nothing
        backup!(node.parent, -reward)
    end

    node.n += 1
    node.reward += reward
end

function select(node::Node, cpuct, ϵ, η)
  scores = map(enumerate(node.children)) do  (i, a)
    q = a.reward / max(a.n, 1)
    prob = iszero(ϵ) ? a.P : (1 - ϵ) * a.P + ϵ * η[i]
    explore = cpuct * prob * √(node.n) / (1 + a.n)
    q + explore
  end
  action_id = argmax(scores)
  node.children[action_id]
end

function isLeaf(node::Node)
  return length(node.children) == 0
end

function simulate!(env::Env, game, root::Node; η)
    node = root
    # select
    while true
        if isLeaf(node)
          break
        end
        ϵ = node == root ? env.noise_ϵ : 0.
        node = select(node, env.cpuct, ϵ, η)
        GI.play!(game, node.action)
        env.total_nodes_traversed += 1
    end

    leaf_value = 0.
    # expand
    if GI.game_terminated(game)
        leaf_value = GI.white_reward(game)
        wp = GI.white_playing(game)
        leaf_value = wp ? leaf_value : -leaf_value
    else
        leaf_value = expand!(env, node, game)
        env.total_nodes_traversed += 1
    end

    # backup
    backup!(node, -leaf_value)
end

        
function explore!(env::Env, game, nsims, episode=1)
  η = dirichlet_noise(game, env.noise_α)
  root = Node(nothing, 0)
  for i in 1:nsims
    env.total_simulations += 1
    simulate!(env, GI.clone(game), root, η=η) 
  end

  # to keep the API same
  state = GI.current_state(game)
  info, new_node = state_info(env, state)
  states = [ActionStats(0, 0, c.n) for c in root.children]
  env.tree[state] = StateInfo(states, 0.)
end
"""
    MCTS.reset!(env)

Empty the MCTS tree.
"""
function reset!(env)
  empty!(env.tree)
  #GC.gc(true)
end

#####
##### Profiling Utilities
#####

"""
    MCTS.average_exploration_depth(env)

Return the average number of nodes that are traversed during an
MCTS simulation, not counting the root.
"""
function average_exploration_depth(env)
  env.total_simulations == 0 && (return 0)
  return env.total_nodes_traversed / env.total_simulations
end

"""
    MCTS.memory_footprint_per_node(gspec)

Return an estimate of the memory footprint of a single MCTS node
for the given game (in bytes).
"""
function memory_footprint_per_node(gspec)
  # The hashtable is at most twice the number of stored elements
  # For every element, a state and a pointer are stored
  size_key = 2 * (GI.state_memsize(gspec) + sizeof(Int))
  dummy_stats = StateInfo([
    ActionStats(0, 0, 0) for i in 1:GI.num_actions(gspec)], 0)
  size_stats = Base.summarysize(dummy_stats)
  return size_key + size_stats
end

"""
    MCTS.approximate_memory_footprint(env)

Return an estimate of the memory footprint of the MCTS tree (in bytes).
"""
function approximate_memory_footprint(env::Env)
  return memory_footprint_per_node(env.gspec) * length(env.tree)
end

# Possibly very slow for large trees
memory_footprint(env::Env) = Base.summarysize(env.tree)

end
