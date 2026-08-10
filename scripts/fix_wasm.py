with open("tests/backends/test_edge.py") as f:
    c = f.read()
c = c.replace('assert "Fallback for unsupported" in code', 'assert "_scalar_unknownop" in code')
with open("tests/backends/test_edge.py", "w") as f:
    f.write(c)
