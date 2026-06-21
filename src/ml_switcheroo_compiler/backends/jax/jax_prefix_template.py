"""Prefix template code."""

JAX_PREFIX_TEMPLATE = """def jax_elastic_transform(images, displacement, interpolation, fill_value, data_format):
    has_batch = images.ndim == 4
    if not has_batch:
        images = images[None, ...]
        displacement = displacement[None, ...]
    if data_format == "channels_first":
        images = jnp.transpose(images, (0, 2, 3, 1))
    B_sz, H_dim, W_dim, C_dim = images.shape
    y_grid, x_grid = jnp.meshgrid(jnp.arange(H_dim), jnp.arange(W_dim), indexing='ij')
    y_grid, x_grid = y_grid.astype(jnp.float32), x_grid.astype(jnp.float32)
    import jax
    import jax.scipy.ndimage as ndimage
    def process_batch(img, disp):
        src_y = y_grid + disp[..., 0]
        src_x = x_grid + disp[..., 1]
        def process_channel(c_img):
            order = 1 if interpolation == "bilinear" else 0
            return ndimage.map_coordinates(c_img, [src_y, src_x], order=order, mode='constant', cval=fill_value)
        return jax.vmap(process_channel, in_axes=-1, out_axes=-1)(img)
    out = jax.vmap(process_batch)(images, displacement)
    if data_format == "channels_first":
        out = jnp.transpose(out, (0, 3, 1, 2))
    if not has_batch:
        out = out[0]
    return out
def jax_gaussian_blur(images, kernel_size, sigma, padding, data_format):
    has_batch = images.ndim == 4
    if not has_batch:
        images = images[None, ...]
    if data_format == "channels_first":
        images = jnp.transpose(images, (0, 2, 3, 1))
    B, H, W, C = images.shape
    ky, kx = kernel_size
    sy, sx = sigma
    y = jnp.arange(-ky // 2 + 1, ky // 2 + 1, dtype=images.dtype)
    x = jnp.arange(-kx // 2 + 1, kx // 2 + 1, dtype=images.dtype)
    yy, xx = jnp.meshgrid(y, x, indexing='ij')
    kernel = jnp.exp(-(yy**2 / (2.0 * sy**2) + xx**2 / (2.0 * sx**2)))
    kernel = kernel / jnp.sum(kernel)
    kernel = kernel.reshape(ky, kx, 1, 1)
    kernel = jnp.broadcast_to(kernel, (ky, kx, C, 1))
    import jax.lax as lax
    dn = lax.conv_dimension_numbers(images.shape, kernel.shape, ('NHWC', 'HWIO', 'NHWC'))
    out = lax.conv_general_dilated(images, kernel, window_strides=(1, 1), padding=padding.upper(), dimension_numbers=dn, feature_group_count=C)
    if data_format == "channels_first":
        out = jnp.transpose(out, (0, 3, 1, 2))
    if not has_batch:
        out = out[0]
    return out
def jax_median_filter(images, kernel_size, padding, data_format):
    has_batch = images.ndim == 4
    if not has_batch:
        images = images[None, ...]
    if data_format == "channels_first":
        images = jnp.transpose(images, (0, 2, 3, 1))
    import jax.lax as lax
    B, H, W, C = images.shape
    ky, kx = kernel_size
    if padding == 'same':
        pad_y, pad_x = ky // 2, kx // 2
        images = jnp.pad(images, ((0, 0), (pad_y, pad_y), (pad_x, pad_x), (0, 0)))
        H, W = images.shape[1], images.shape[2]
    out_H, out_W = H - ky + 1, W - kx + 1
    patches = jax.lax.conv_general_dilated_patches(images, (ky, kx), (1, 1), 'VALID', dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    patches = patches.reshape(B, out_H, out_W, ky * kx, C)
    out = jnp.median(patches, axis=3)
    if data_format == "channels_first":
        out = jnp.transpose(out, (0, 3, 1, 2))
    if not has_batch:
        out = out[0]
    return out
def jax_extract_bounding_boxes(images, boxes, box_indices, crop_size, interpolation, extrapolation_value, data_format):
    import jax
    import jax.numpy as jnp
    if data_format == "channels_first":
        images = jnp.transpose(images, (0, 2, 3, 1))
    import jax.image as jimg
    # JAX doesn't have a direct crop_and_resize equivalent natively exposed like tf.image.crop_and_resize
    # We'll use the eager fallback mapped into JAX.
    from ml_switcheroo_compiler.backends.eager_utils import extract_bounding_boxes_eager
    return extract_bounding_boxes_eager(jnp, images, boxes, box_indices, crop_size, interpolation, extrapolation_value, data_format)
def jax_iou(boxes1, boxes2, bounding_box_format):
    from ml_switcheroo_compiler.backends.eager_utils import iou_eager
    import jax.numpy as jnp
    return iou_eager(jnp, boxes1, boxes2, bounding_box_format)
def jax_nms(boxes, scores, max_output_size, iou_threshold, score_threshold):
    from ml_switcheroo_compiler.backends.eager_utils import nms_eager
    import jax.numpy as jnp
    return nms_eager(jnp, boxes, scores, max_output_size, iou_threshold, score_threshold)
def jax_resize(images, size, interpolation, align_corners):
    import jax.image as jimg
    method = 'lanczos3' if interpolation == 'lanczos3' else 'bicubic'
    return jimg.resize(images, (images.shape[0], size[0], size[1], images.shape[3]), method)
def jax_istft(stft_tensor, frame_length, frame_step, fft_length, window, center):
    from ml_switcheroo_compiler.backends.eager_utils import istft_eager
    import jax.numpy as jnp
    return istft_eager(jnp, stft_tensor, frame_length, frame_step, fft_length, window, center)
def jax_mel_filterbank(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz):
    from ml_switcheroo_compiler.backends.eager_utils import mel_filterbank_eager
    import jax.numpy as jnp
    return mel_filterbank_eager(jnp, None, num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
def jax_mfcc(spectrogram, sample_rate, num_mel_bins, lower_edge_hertz, upper_edge_hertz, num_mfccs):
    from ml_switcheroo_compiler.backends.eager_utils import mfcc_eager
    import jax.numpy as jnp
    return mfcc_eager(jnp, spectrogram, sample_rate, num_mel_bins, lower_edge_hertz, upper_edge_hertz, num_mfccs)
def jax_perspective_transform(images, start_points, end_points, interpolation, fill_value, data_format):
    def get_h(src, dst):
        A = jnp.zeros((*dst.shape[:-2], 8, 8), dtype=jnp.float32)
        B = jnp.zeros((*dst.shape[:-2], 8), dtype=jnp.float32)
        for i in range(4):
            u, v = dst[..., i, 0], dst[..., i, 1]
            x, y = src[..., i, 0], src[..., i, 1]
            A = A.at[..., i*2, 0].set(u)
            A = A.at[..., i*2, 1].set(v)
            A = A.at[..., i*2, 2].set(1.0)
            A = A.at[..., i*2, 6].set(-x * u)
            A = A.at[..., i*2, 7].set(-x * v)
            A = A.at[..., i*2+1, 3].set(u)
            A = A.at[..., i*2+1, 4].set(v)
            A = A.at[..., i*2+1, 5].set(1.0)
            A = A.at[..., i*2+1, 6].set(-y * u)
            A = A.at[..., i*2+1, 7].set(-y * v)
            B = B.at[..., i*2].set(x)
            B = B.at[..., i*2+1].set(y)
        h = jnp.linalg.solve(A, B)
        return jnp.concatenate([h, jnp.ones((*dst.shape[:-2], 1), dtype=jnp.float32)], axis=-1).reshape((*dst.shape[:-2], 3, 3))
    has_batch = images.ndim == 4
    if not has_batch:
        images = images[None, ...]
        start_points = start_points[None, ...]
        end_points = end_points[None, ...]
    if data_format == "channels_first":
        images = jnp.transpose(images, (0, 2, 3, 1))
    H_mat = get_h(start_points, end_points)
    B_sz, H_dim, W_dim, C_dim = images.shape
    y_grid, x_grid = jnp.meshgrid(jnp.arange(H_dim), jnp.arange(W_dim), indexing='ij')
    y_grid = y_grid.astype(jnp.float32)
    x_grid = x_grid.astype(jnp.float32)
    coords = jnp.stack([x_grid, y_grid, jnp.ones_like(x_grid)], axis=-1)
    import jax
    import jax.scipy.ndimage as ndimage
    def process_batch(img, h_mat):
        t_coords = coords @ h_mat.T
        t_coords = t_coords / t_coords[..., 2:3]
        src_x = t_coords[..., 0]
        src_y = t_coords[..., 1]
        def process_channel(c_img):
            order = 1 if interpolation == "bilinear" else 0
            return ndimage.map_coordinates(c_img, [src_y, src_x], order=order, mode='constant', cval=fill_value)
        return jax.vmap(process_channel, in_axes=-1, out_axes=-1)(img)
    out = jax.vmap(process_batch)(images, H_mat)
    if data_format == "channels_first":
        out = jnp.transpose(out, (0, 3, 1, 2))
    if not has_batch:
        out = out[0]
    return out

def jax_power_iteration(w, num_iters, u=None):
    import jax
    import jax.numpy as jnp
    if u is None:
        u = jnp.ones(w.shape[:-2] + (w.shape[-2], 1), dtype=w.dtype)
    def cond_fun(val):
        return val[0] < num_iters
    def body_fun(val):
        i, u_curr, _ = val
        w_t = jnp.swapaxes(w, -1, -2)
        v_next = jnp.matmul(w_t, u_curr)
        v_next = v_next / (jnp.linalg.norm(v_next, axis=-2, keepdims=True) + 1e-12)
        u_next = jnp.matmul(w, v_next)
        u_next = u_next / (jnp.linalg.norm(u_next, axis=-2, keepdims=True) + 1e-12)
        return i + 1, u_next, v_next
    init_v = jnp.zeros(w.shape[:-2] + (w.shape[-1], 1), dtype=w.dtype)
    _, u_final, v_final = jax.lax.while_loop(cond_fun, body_fun, (0, u, init_v))
    sigma = jnp.matmul(jnp.swapaxes(u_final, -1, -2), jnp.matmul(w, v_final))
    return jnp.squeeze(v_final, -1), jnp.squeeze(u_final, -1), jnp.squeeze(jnp.squeeze(sigma, -1), -1)
"""
