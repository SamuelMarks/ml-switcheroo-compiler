with open("tests/backends/tensorflow/test_tensorflow_generator.py") as f:
    c = f.read()
c = c.replace('assert "Zeros" in ops', 'assert "Beta" in ops')
with open("tests/backends/tensorflow/test_tensorflow_generator.py", "w") as f:
    f.write(c)
