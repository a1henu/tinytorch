echo "=== Running all tests... ==="
echo "=============================="

echo "=== Cleaning build directory... ==="
bash clean.sh
echo "=== Cleaning finished. ==="
echo "=============================="

bash test.sh --cpu
echo "=== CPU Tests Finished ==="
echo "=============================="

echo "=== Cleaning build directory... ==="
bash clean.sh
echo "=== Cleaning finished. ==="
echo "=============================="

bash test.sh --gpu
echo "=== GPU Tests Finished ==="
echo "=============================="