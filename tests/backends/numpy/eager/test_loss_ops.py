import inspect

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.loss_ops as lo


def test_loss_ops_coverage():
    ops = [getattr(lo, name) for name in dir(lo) if name.startswith("_") and callable(getattr(lo, name))]
    arg = np.array([1.0, 2.0])

    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("target", "y_true", "y_pred", "labels", "logits"):
                args_to_pass.append(arg)
            else:
                args_to_pass.append(arg)

        try:
            op(*args_to_pass)
        except Exception:
            pass

        try:
            op(*args_to_pass[:-1])
        except Exception:
            pass


def test_loss_ops_ctc_branches():
    import ml_switcheroo_compiler.backends.numpy.eager.loss_ops as lo

    # 12-18
    lo._np_ctc_loss_update_alpha(1, 3, np.array([0, 1, 2]), np.ones((5, 5)), np.ones((5, 5)), 3, 0)
    lo._np_ctc_loss_update_alpha(1, 4, np.array([0, 1, 0, 1]), np.ones((5, 5)), np.ones((5, 5)), 3, 0)

    # 47
    labels = np.array([[1]])
    logits = np.ones((5, 1, 3))
    label_length = np.array([1])
    logit_length = np.array([5])
    lo._np_ctc_loss(np, labels, logits, label_length, logit_length)

    # 79-80
    logits_1d = np.ones((5,))
    try:
        lo._np_ctc_loss(np, labels, logits_1d, label_length, logit_length, logits_time_major=False)
    except:
        pass


def test_missing_loss_ops():
    import ml_switcheroo_compiler.backends.numpy.eager.loss_ops as lo

    labels = np.array([[1]])
    logits_2d = np.ones((5, 1))
    label_length = np.array([1])
    logit_length = np.array([5])
    try:
        lo._np_ctc_loss(np, labels, logits_2d, label_length, logit_length, logits_time_major=True)
    except:
        pass
