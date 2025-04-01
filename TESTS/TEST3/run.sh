# gaussian
./foo1.ex foo ./data/foo -1 -1 0 0 0 0 500 10 10 10 1 1 > foo.out
./foo2.ex foo ./data/foo ./data/foo.window -1 -1 0 0 0 0 500 10 10 10 1 1 >> foo.out
./foo3.ex foo ./data/foo ./data/foo.window -1 -1 0 0 0 0 500 10 10 10 1 1 >> foo.out
# matern
./foo1.ex foo ./data/foo -1 -1 1 0 0 0 500 10 10 10 1 1 >> foo.out
./foo2.ex foo ./data/foo ./data/foo.window -1 -1 1 0 0 0 500 10 10 10 1 1 >> foo.out
./foo3.ex foo ./data/foo ./data/foo.window -1 -1 1 0 0 0 500 10 10 10 1 1 >> foo.out
# see results without training
# gaussian
./foo1.ex foo ./data/foo -1 -1 0 0 0 0 0 10 10 10 1 1 > foo.out
./foo2.ex foo ./data/foo ./data/foo.window -1 -1 0 0 0 0 0 10 10 10 1 1 >> foo.out
./foo3.ex foo ./data/foo ./data/foo.window -1 -1 0 0 0 0 0 10 10 10 1 1 >> foo.out
# matern
./foo1.ex foo ./data/foo -1 -1 1 0 0 0 0 10 10 10 1 1 >> foo.out
./foo2.ex foo ./data/foo ./data/foo.window -1 -1 1 0 0 0 0 10 10 10 1 1 >> foo.out
./foo3.ex foo ./data/foo ./data/foo.window -1 -1 1 0 0 0 0 10 10 10 1 1 >> foo.out