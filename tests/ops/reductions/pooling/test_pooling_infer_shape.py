from ml_switcheroo_compiler.ops.reductions.pooling import (
    AdaptiveAvgPool2D,
    AdaptiveAvgPool3D,
    AdaptiveMaxPool2D,
    AdaptiveMaxPool3D,
    AdaptiveMaxPool3D_Indices,
    CTCLoss,
    Fold,
    FractionalMaxPool2D,
    FractionalMaxPool3D,
    FractionalMaxPool3D_Indices,
    MaxPoolWithIndices,
    MaxPoolWithIndices_Indices,
    MaxUnpool1D,
    MaxUnpool2D,
    MaxUnpool3D,
    Unfold,
)


def test_pooling_infer_shape():
    class DummyShape2D:
        shape = (1, 1, 4, 4)

    class DummyShape3D:
        shape = (1, 1, 4, 4, 4)

    assert CTCLoss().infer_shape(DummyShape2D()) == (1,)
    assert CTCLoss().infer_shape() == ()
    assert FractionalMaxPool2D().infer_shape(DummyShape2D(), [2, 2]) == (1, 1, 2, 2)
    assert FractionalMaxPool2D().infer_shape() == ()
    assert AdaptiveAvgPool2D().infer_shape(DummyShape2D(), [2, 2]) == (1, 1, 2, 2)
    assert AdaptiveAvgPool2D().infer_shape() == ()
    assert AdaptiveMaxPool2D().infer_shape(DummyShape2D(), [2, 2]) == (1, 1, 2, 2)
    assert AdaptiveMaxPool2D().infer_shape() == ()
    assert Unfold().infer_shape(DummyShape2D()) == ()
    assert Unfold().infer_shape() == ()
    assert Fold().infer_shape(DummyShape2D()) == ()
    assert Fold().infer_shape() == ()
    assert FractionalMaxPool3D().infer_shape(DummyShape3D(), [2, 2, 2]) == (1, 1, 2, 2, 2)
    assert FractionalMaxPool3D().infer_shape() == ()
    assert AdaptiveAvgPool3D().infer_shape(DummyShape3D(), [2, 2, 2]) == (1, 1, 2, 2, 2)
    assert AdaptiveAvgPool3D().infer_shape() == ()
    assert AdaptiveMaxPool3D().infer_shape(DummyShape3D(), [2, 2, 2]) == (1, 1, 2, 2, 2)
    assert AdaptiveMaxPool3D().infer_shape() == ()
    assert MaxUnpool1D().infer_shape(DummyShape2D(), output_size=[2]) == (1, 1, 4, 2)
    assert MaxUnpool1D().infer_shape() == ()
    assert MaxUnpool2D().infer_shape(DummyShape2D(), output_size=[2, 2]) == (1, 1, 2, 2)
    assert MaxUnpool2D().infer_shape() == ()
    assert MaxUnpool3D().infer_shape(DummyShape3D(), output_size=[2, 2, 2]) == (1, 1, 2, 2, 2)
    assert MaxUnpool3D().infer_shape() == ()
    assert AdaptiveMaxPool3D_Indices().infer_shape(DummyShape3D(), [2, 2, 2]) == (1, 1, 2, 2, 2)
    assert AdaptiveMaxPool3D_Indices().infer_shape() == ()
    assert FractionalMaxPool3D_Indices().infer_shape(DummyShape3D(), [2, 2, 2]) == (1, 1, 2, 2, 2)
    assert FractionalMaxPool3D_Indices().infer_shape() == ()
    assert MaxPoolWithIndices().infer_shape(DummyShape2D()) == ()
    assert MaxPoolWithIndices().infer_shape() == ()
    assert MaxPoolWithIndices_Indices().infer_shape(DummyShape2D()) == ()
    assert MaxPoolWithIndices_Indices().infer_shape() == ()
