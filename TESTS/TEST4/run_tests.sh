#!/bin/bash

data_name="$1"  # Get the data name from the first argument

# Function to run a command with the data name substituted
run_command() {
    local window_file="$1"
    local params="$2"
    ./foo.ex "$data_name" "./data/$data_name" "./data/$window_file" $params >> "${data_name}.out"
}

echo "Running tests for $data_name" > "${data_name}.out"

# Execute commands with the provided data name
run_command "${data_name}.g.window" "-1 -1 0 0 0 0 0 10 10 10 4"
run_command "${data_name}.m.window" "-1 -1 1 0 0 0 0 10 10 10 4"
run_command "${data_name}.g.window" "-1 -1 0 0 0 0 500 10 10 10 4"
run_command "${data_name}.m.window" "-1 -1 1 0 0 0 500 10 10 10 4"