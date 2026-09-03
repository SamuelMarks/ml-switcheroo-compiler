from ml_switcheroo_compiler.export.pb_utils import encode_varint


def test_encode_varint_coverage():
    """Test function."""
    # Negative number
    res1 = encode_varint(-1)
    # Large positive number
    res2 = encode_varint(128)
