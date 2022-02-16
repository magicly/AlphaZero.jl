using HTTP
using JSON

using AlphaZero

const BOARD_SIDE = 12
experiment = Examples.experiments["gobang"];
session = Session(experiment, dir="sessions/go-1230-1");

player = AlphaZeroPlayer(session)

function getAction(req::HTTP.Request)
    body = HTTP.payload(req) |> String |> JSON.parse
		board_json = body["board"]
		curplayer = body["curplayer"] == "w" ? 1 : 2
		last_action = body["last_action"]
		println("body: $board_json, $curplayer, $last_action")
		board = zeros(UInt8, BOARD_SIDE, BOARD_SIDE)
		for row in 1:length(board_json)
			for col in 1:length(board_json[row])
				if board_json[row][col] == ""
					board[row, col] = 0
				elseif board_json[row][col] == "w"
					board[row, col] = 1
				elseif board_json[row][col] == "b"
					board[row, col] = 2
				end
			end
		end
		# pos = last_action[2] * BOARD_SIDE + last_action[1] + 1
		# println(board)
		# println("pos: $pos")

    t1 = time()
		game = GI.init(experiment.gspec)
		# state = GI.current_state(game)
  	amask = ones(Bool, BOARD_SIDE, BOARD_SIDE)
		amask[board .!= 0] .= false
		amask = reshape(amask, BOARD_SIDE^2)
		GI.set_state!(game, (board = board, curplayer = curplayer, finished = false, winner = 0, amask = amask))

		action = select_move(player, game, 1)
		col, row = divrem(action - 1, BOARD_SIDE)
		action_json = [row, col]
		println("action: $action, $action_json")
    r = Dict("action" => action_json)

    t2 = time()
    println("cost: ", t2 - t1)
    headers = ["Content-Type" => "application/json"]

    return HTTP.Response(200, headers, body=JSON.json(r))
end


ROUTER = HTTP.Router()
HTTP.@register(ROUTER, "POST", "/", getAction)

println("start...")
HTTP.serve(ROUTER, "0.0.0.0", 8081)