python AutoEncode.py --net ibert --seq_type fib --log true --exp 1
python AutoEncode.py --net xlnet --seq_type fib --log true --exp 2
python AutoEncode.py --net lstm --seq_type fib --log true --exp 4
python AutoEncode.py --net tf --seq_type fib --log true --exp 5
python AutoEncode.py --net ibert --seq_type arith --log true  --exp 6
python AutoEncode.py --net xlnet --seq_type arith --log true --exp 7
python AutoEncode.py --net lstm --seq_type arith --log true --exp 9
python AutoEncode.py --net tf --seq_type arith --log true --exp 10
python AutoEncode.py --net ibert --seq_type copy --log true  --exp 11
python AutoEncode.py --net xlnet --seq_type copy --log true --exp 12
python AutoEncode.py --net lstm --seq_type copy  --log true --exp 14
python AutoEncode.py --net tf --seq_type copy  --log true --exp 15
python AutoEncode.py --net ibert --seq_type palin --log true  --exp 16
python AutoEncode.py --net xlnet --seq_type palin --log true --exp 17
python AutoEncode.py --net lstm --seq_type palin --log true --exp 19
python AutoEncode.py --net tf --seq_type palin --log true --exp 20
python AutoEncode.py --net ibert --seq_type fib --digits 5 --log true  --exp 21
python AutoEncode.py --net ibertpos --seq_type fib --digits 5 --log true  --exp 22
python AutoEncode.py --net ibert2 --seq_type fib --digits 5 --log true --exp 23
python AutoEncode.py --net ibert --seq_type fib --digits 8 --log true --exp 24
python AutoEncode.py --net ibertpos --seq_type fib --digits 8 --log true --exp 25
python AutoEncode.py --net ibert2 --seq_type fib --digits 8 --log true  --exp 26
python AutoEncode.py --net ibert  --seq_type ptbc --lr 1e-4 --epochs 100 --log true  --exp 27 --batch_size 16
python AutoEncode.py --net xlnet --seq_type ptbc --lr 1e-4 --epochs 100 --log true --exp 28 --batch_size 16
python AutoEncode.py --net lstm --seq_type ptbc --lr 1e-4 --epochs 100  --log true --exp 30 --batch_size 16
python AutoEncode.py --net tf  --seq_type ptbc --lr 1e-4 --epochs 100  --log true --exp 31 --batch_size 16
python AutoEncode.py --net ibert  --seq_type ptbw --lr 1e-4 --epochs 100 --log true  --exp 32
python AutoEncode.py --net xlnet --seq_type ptbw --lr 1e-4 --epochs 100 --log true  --exp 33
python AutoEncode.py --net lstm --seq_type ptbw --lr 1e-4 --epochs 100  --log true --exp 35
python AutoEncode.py --net tf  --seq_type ptbw --lr 1e-4 --epochs 100  --log true  --exp 36
python AutoEncode.py --net ibertpos --seq_type fib --digits 12 --log true --exp 37
python AutoEncode.py --net ibert2 --seq_type fib --digits 12 --log true  --exp 38
