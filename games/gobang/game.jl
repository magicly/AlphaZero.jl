import AlphaZero.GI
using StaticArrays

const BOARD_SIDE = 15
const N_IN_ROW = 5
const NUM_POSITIONS = BOARD_SIDE^2

const Player = UInt8
const BLACK = 0x01
const WHITE = 0x02

other(p::Player) = 0x03 - p

const Cell = Player
const EMPTY = 0x00
const Board = SMatrix{BOARD_SIDE,BOARD_SIDE,Cell,NUM_POSITIONS}

const INITIAL_BOARD = @SMatrix zeros(Cell, BOARD_SIDE, BOARD_SIDE)
const INITIAL_STATE = (board = INITIAL_BOARD, curplayer = WHITE)


#####
##### Game environments and game specifications
#####

struct GameSpec <: GI.AbstractGameSpec end

mutable struct GameEnv <: GI.AbstractGameEnv
  board::Board
  curplayer::Player
  finished::Bool
  winner::Player
  amask::Vector{Bool} # actions mask
  last_pos::Int
end

function GameEnv(g::GameEnv)
  GameEnv(copy(g.board), g.curplayer, g.finished, g.winner, copy(g.amask), g.last_pos)
end

function GI.init(::GameSpec)
  board = INITIAL_STATE.board
  curplayer = INITIAL_STATE.curplayer
  finished = false
  winner = EMPTY
  # amask = trues(NUM_POSITIONS) # 96, 但是g.amask == 265
  amask = ones(Bool, NUM_POSITIONS)
  # println("amask size: $(Base.summarysize(amask))")
  g = GameEnv(board, curplayer, finished, winner, amask, 0)
  # println(11111, Base.summarysize(g))
  g
end

GI.spec(::GameEnv) = GameSpec()

#####
##### Queries on specs
#####

GI.two_players(::GameSpec) = true

const ACTIONS = collect(1:NUM_POSITIONS)
GI.actions(::GameSpec) = ACTIONS
# GI.actions(::GameSpec) = Array{Int32}(ACTIONS)


# board1: current player
# board2: opponent player
function GI.vectorize_state(::GameSpec, state)
  board = state.board
  player = state.curplayer
  player2 = other(player)

  # obsv = zeros(Float32, BOARD_SIDE, BOARD_SIDE, 5)
  # obsv[:, :, 1] = board .== EMPTY
  # obsv[:, :, 2] = board .== BLACK
  # obsv[:, :, 3] = board .== WHITE
  # obsv[:, :, 4] .= player
  # if state.last_pos != 0
  #   obsv[:, :, 5][state.last_pos] = 1
  # end
  # return obsv




  boards = zeros(Float32, BOARD_SIDE, BOARD_SIDE, 4)
  for j = 1:BOARD_SIDE
    for i = 1:BOARD_SIDE
      if board[i, j] == player
        boards[i, j, 1] = 1
      elseif board[i, j] == player2
        boards[i, j, 2] = 1
      end
    end
  end
  boards[:, :, 3] .= player
  if state.last_pos != 0
    boards[:, :, 4][state.last_pos] = 1
  end
  return boards
end


#####
##### Operations on envs
#####


function GI.set_state!(g::GameEnv, state)
  g.board = state.board
  g.curplayer = state.curplayer
  g.finished = state.finished
  g.winner = state.winner
  g.amask = copy(state.amask) # TODO why
  g.last_pos = state.last_pos
end

# GI.current_state(g::GameEnv) = (board = copy(g.board), curplayer = g.curplayer, finished = g.finished, winner = g.winner, amask = copy(g.amask))
GI.current_state(g::GameEnv) = (board = g.board, curplayer = g.curplayer, finished = g.finished, winner = g.winner, amask = copy(g.amask), last_pos=g.last_pos)

GI.game_terminated(g::GameEnv) = g.finished

GI.white_playing(g::GameEnv) = g.curplayer == WHITE

GI.actions_mask(g::GameEnv) = g.amask

function GI.play!(g::GameEnv, pos)
  g.board = setindex(g.board, g.curplayer, pos)
  # g.board[pos] = g.curplayer # setindex! is not defined
  g.amask[pos] = false
  g.last_pos = pos
  if winning_pattern_at(g.board, g.curplayer, pos)
    g.winner = g.curplayer
    g.finished = true
  else
    g.finished = !any(g.amask)
  end
  g.curplayer = other(g.curplayer)
end

isvalid_pos(x) = x >= 1 && x <= BOARD_SIDE

function winning_pattern_at(board, player, pos)
  col, row = divrem(pos - 1, BOARD_SIDE) .+ 1
  # row
  n = 1
  for i = -1:-1:-4 # -> left
    if isvalid_pos(col + i) && board[row, col + i] == player
      n += 1
    else
      break
    end
  end
  for i = 1:4 # -> right
    if isvalid_pos(col + i) && board[row, col + i] == player
      n += 1
    else
      break
    end
  end
  if n >= N_IN_ROW
    return true
  end
  # col
  n = 1
  for i = -1:-1:-4 # -> up
    if isvalid_pos(row + i) && board[row + i, col] == player
      n += 1
    else
      break
    end
  end
  for i = 1:4 # -> down
    if isvalid_pos(row + i) && board[row + i, col] == player
      n += 1
    else
      break
    end
  end
  if n >= N_IN_ROW
    return true
  end
  # \
  n = 1
  for i = -1:-1:-4 # -> left up
    if isvalid_pos(row + i) && isvalid_pos(col + i) && board[row + i, col + i] == player
      n += 1
    else
      break
    end
  end
  for i = 1:4 # -> right down
    if isvalid_pos(row + i) && isvalid_pos(col + i) && board[row + i, col + i] == player
      n += 1
    else
      break
    end
  end
  if n >= N_IN_ROW
    return true
  end
  # /
  n = 1
  for i = -1:-1:-4 # -> right up
    if isvalid_pos(row + i) && isvalid_pos(col - i) && board[row + i, col - i] == player
      n += 1
    else
      break
    end
  end
  for i = 1:4 # -> left down
    if isvalid_pos(row + i) && isvalid_pos(col - i) && board[row + i, col - i] == player
      n += 1
    else
      break
    end
  end
  if n >= N_IN_ROW
    return true
  end
  false
end


function GI.white_reward(g::GameEnv)
     g.winner == WHITE && return 1.
     g.winner == BLACK && return -1.
     return 0.
end

function GI.heuristic_value(g::GameEnv)
  0.
end


#####
##### Symmetries
#####

# TODO


#####
##### Interface for interactive exploratory tools
#####

function GI.action_string(::GameSpec, a)
  string(Char(Int('A') + a - 1))
end

function GI.parse_action(::GameSpec, str)
  length(str) == 1 || (return nothing)
  x = Int(uppercase(str[1])) - Int('A')
  (0 <= x < NUM_POSITIONS) ? x + 1 : nothing
end

function read_board(::GameSpec)
  n = BOARD_SIDE
  str = reduce(*, ((readline() * "   ")[1:n] for i in 1:n))
  white = ['w', 'r', 'o']
  black = ['b', 'b', 'x']
  function cell(i)
    if (str[i] ∈ white) WHITE
    elseif (str[i] ∈ black) BLACK
    else nothing end
  end
  @SVector [cell(i) for i in 1:NUM_POSITIONS]
end

function GI.read_state(::GameSpec)
  b = read_board(GameSpec())
  nw = count(==(WHITE), b)
  nb = count(==(BLACK), b)
  if nw == nb
    return (board = b, curplayer = WHITE)
  elseif nw == nb + 1
    return (board = b, curplayer = BLACK)
  else
    return nothing
  end
end

using Crayons

player_color(p) = p == WHITE ? crayon"light_red" : crayon"light_blue"
player_name(p)  = p == WHITE ? "Red" : "Blue"
player_mark(p)  = p == WHITE ? "o" : "x"

function GI.render(g::GameEnv; with_position_names=true, botmargin=true)
  pname = player_name(g.curplayer)
  pcol = player_color(g.curplayer)
  print(pcol, pname, " plays:", crayon"reset", "\n\n")
  for y in 1:BOARD_SIDE
    for x in 1:BOARD_SIDE
      pos = pos_of_xy((x, y))
      c = g.board[pos]
      if isnothing(c)
        print(" ")
      else
            print(player_color(c), player_mark(c), crayon"reset")
      end
      print(" ")
    end
    if with_position_names
      print(" | ")
      for x in 1:BOARD_SIDE
        print(GI.action_string(GI.spec(g), pos_of_xy((x, y))), " ")
      end
    end
    print("\n")
  end
  botmargin && print("\n")
end
