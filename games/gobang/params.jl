Network = NetLib.ResNet

netparams = NetLib.ResNetHP(
  num_filters=128,
  num_blocks=5,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=32,
  num_value_head_filters=32,
  batch_norm_momentum=0.1)

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=256 * 4,
    num_workers=64*4,
    batch_size=32*4,
    use_gpu=true,
    reset_every=2,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=800,
    cpuct=2.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  sim=SimParams(
    num_games=128,
    num_workers=128,
    batch_size=64,
    use_gpu=true,
    reset_every=2,
    flip_probability=0.,
    alternate_colors=true),
  mcts=MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.2),
    dirichlet_noise_ϵ=0.05),
  update_threshold=0.05)

learning = LearningParams(
  use_gpu=true,
  samples_weighing_policy=LOG_WEIGHT,
  l2_regularization=1e-4,
  optimiser=CyclicNesterov(
    lr_base=1e-3,
    lr_high=1e-2,
    lr_low=1e-3,
    momentum_high=0.9,
    momentum_low=0.8),
  batch_size=32,
  loss_computation_batch_size=2048,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=0,
  max_batches_per_checkpoint=5_000,
  num_checkpoints=1)

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=15,
  memory_analysis=MemAnalysisParams(
    num_game_stages=4),
  ternary_rewards=true,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(80_000))

benchmark_sim = SimParams(
  arena.sim;
  num_games=10,
  num_workers=100,
  batch_size=100)

benchmark = [
  # Benchmark.Duel(
  #   Benchmark.Full(self_play.mcts),
  #   Benchmark.MctsRollouts(self_play.mcts),
  #   benchmark_sim),
  Benchmark.Duel(
    Benchmark.NetworkOnly(),
    Benchmark.MctsRollouts(self_play.mcts),
    benchmark_sim)]

experiment = Experiment(
  "gobang", GameSpec(), params, Network, netparams, benchmark)