import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

plt.style.use('dark_background')

# Skala pikseli
pixel_scale = 40

# ----------------------------------------------------
# 1. Dane przykładowe
# ----------------------------------------------------
image_rgb = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
n_rows, n_cols, _ = image_rgb.shape

k_rows, k_cols = 3, 3

filter1 = np.zeros((k_rows, k_cols, 3), dtype=int)
filter1[:, :, 0] = np.array([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]])
filter1[:, :, 1] = np.array([[1, 0, 1],
                             [0, 1, 0],
                             [1, 0, 1]])
filter1[:, :, 2] = np.array([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]])

filter2 = np.zeros((k_rows, k_cols, 3), dtype=int)
filter2[:, :, 0] = np.array([[1, 1, 1],
                             [0, 0, 0],
                             [1, 1, 1]])
filter2[:, :, 1] = np.array([[0, 0, 0],
                             [1, 1, 1],
                             [0, 0, 0]])
filter2[:, :, 2] = np.array([[1, 0, 1],
                             [0, 1, 0],
                             [1, 0, 1]])

filter3 = np.zeros((k_rows, k_cols, 3), dtype=int)
filter3[:, :, 0] = np.array([[0, 0, 0],
                             [1, 1, 1],
                             [0, 0, 0]])
filter3[:, :, 1] = np.array([[1, 1, 1],
                             [0, 0, 0],
                             [1, 1, 1]])
filter3[:, :, 2] = np.array([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 0]])

max_conv1 = np.sum(filter1) * 255
max_conv2 = np.sum(filter2) * 255
max_conv3 = np.sum(filter3) * 255

out_rows = n_rows - k_rows + 1
out_cols = n_cols - k_cols + 1

feat_R = np.zeros((out_rows, out_cols), dtype=int)
feat_G = np.zeros((out_rows, out_cols), dtype=int)
feat_B = np.zeros((out_rows, out_cols), dtype=int)

# ----------------------------------------------------
# 2. Układ subplotów (Gridspec)
# ----------------------------------------------------
fig = plt.figure(figsize=(24, 28))

main_gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.4, hspace=0.3)

ax_in_R = fig.add_subplot(main_gs[0, 0])
ax_in_G = fig.add_subplot(main_gs[1, 0])
ax_in_B = fig.add_subplot(main_gs[2, 0])

ax_patch_R = fig.add_subplot(main_gs[0, 1])
ax_patch_G = fig.add_subplot(main_gs[1, 1])
ax_patch_B = fig.add_subplot(main_gs[2, 1])

ax_feat_R = fig.add_subplot(main_gs[0, 2])
ax_feat_G = fig.add_subplot(main_gs[1, 2])
ax_feat_B = fig.add_subplot(main_gs[2, 2])

right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[:, 3], hspace=0.3)
ax_orig = fig.add_subplot(right_gs[0, 0])
ax_combined = fig.add_subplot(right_gs[1, 0])

# Słowniki do rysowania tekstu
text_kwargs = dict(ha='center', va='center', fontsize=6, color='white')
text_kwargs_orig = dict(ha='center', va='center', fontsize=3, color='white')

axes_list = [
    ax_in_R, ax_in_G, ax_in_B,
    ax_patch_R, ax_patch_G, ax_patch_B,
    ax_feat_R, ax_feat_G, ax_feat_B,
    ax_orig, ax_combined
]
for ax in axes_list:
    ax.set_xticks([])
    ax.set_yticks([])

# ----------------------------------------------------
# 2A. Kanały wejściowe R/G/B z wartościami
# ----------------------------------------------------
im_in_R = ax_in_R.imshow(
    image_rgb[:, :, 0],
    cmap='Reds',
    origin='upper',
    extent=[0, n_cols * pixel_scale, n_rows * pixel_scale, 0],
    interpolation='nearest',
    vmin=0, vmax=255
)
ax_in_R.set_title("Kanał R", color='magenta', fontsize=12)
for r in range(n_rows):
    for c in range(n_cols):
        ax_in_R.text(
            c * pixel_scale + pixel_scale / 2,
            r * pixel_scale + pixel_scale / 2,
            f"{image_rgb[r, c, 0]:03d}",
            **text_kwargs
        )

im_in_G = ax_in_G.imshow(
    image_rgb[:, :, 1],
    cmap='Greens',
    origin='upper',
    extent=[0, n_cols * pixel_scale, n_rows * pixel_scale, 0],
    interpolation='nearest',
    vmin=0, vmax=255
)
ax_in_G.set_title("Kanał G", color='magenta', fontsize=12)
for r in range(n_rows):
    for c in range(n_cols):
        ax_in_G.text(
            c * pixel_scale + pixel_scale / 2,
            r * pixel_scale + pixel_scale / 2,
            f"{image_rgb[r, c, 1]:03d}",
            **text_kwargs
        )

im_in_B = ax_in_B.imshow(
    image_rgb[:, :, 2],
    cmap='Blues',
    origin='upper',
    extent=[0, n_cols * pixel_scale, n_rows * pixel_scale, 0],
    interpolation='nearest',
    vmin=0, vmax=255
)
ax_in_B.set_title("Kanał B", color='magenta', fontsize=12)
for r in range(n_rows):
    for c in range(n_cols):
        ax_in_B.text(
            c * pixel_scale + pixel_scale / 2,
            r * pixel_scale + pixel_scale / 2,
            f"{image_rgb[r, c, 2]:03d}",
            **text_kwargs
        )

# ----------------------------------------------------
# 2B. Złączony obraz RGB (Wejściowy)
# ----------------------------------------------------
im_orig = ax_orig.imshow(
    image_rgb,
    origin='upper',
    extent=[0, n_cols * pixel_scale, n_rows * pixel_scale, 0],
    interpolation='nearest'
)
ax_orig.set_title("Złączony Obraz RGB (Wejściowy)", color='magenta', fontsize=12)

texts_orig = []
for r in range(n_rows):
    row_texts = []
    for c in range(n_cols):
        val_r = image_rgb[r, c, 0]
        val_g = image_rgb[r, c, 1]
        val_b = image_rgb[r, c, 2]
        color_hex = f"#{val_r:02X}{val_g:02X}{val_b:02X}"
        txt = ax_orig.text(
            c * pixel_scale + pixel_scale / 2,
            r * pixel_scale + pixel_scale / 2,
            color_hex,
            **text_kwargs_orig
        )
        row_texts.append(txt)
    texts_orig.append(row_texts)

rect = patches.Rectangle((0, 0), k_cols * pixel_scale, k_rows * pixel_scale,
                         edgecolor='magenta', facecolor='none', lw=2)
ax_orig.add_patch(rect)
ax_orig.set_xlim(0, n_cols * pixel_scale)
ax_orig.set_ylim(n_rows * pixel_scale, 0)

# ----------------------------------------------------
# 2C. Patch’e (R/G/B)
# ----------------------------------------------------
im_patch_R = ax_patch_R.imshow(
    np.zeros((k_rows, k_cols)),
    cmap='Reds',
    origin='upper',
    extent=[0, k_cols * pixel_scale, k_rows * pixel_scale, 0],
    interpolation='nearest',
    vmin=0, vmax=255
)
ax_patch_R.set_title("Patch R", color='magenta', fontsize=12)

im_patch_G = ax_patch_G.imshow(
    np.zeros((k_rows, k_cols)),
    cmap='Greens',
    origin='upper',
    extent=[0, k_cols * pixel_scale, k_rows * pixel_scale, 0],
    interpolation='nearest',
    vmin=0, vmax=255
)
ax_patch_G.set_title("Patch G", color='magenta', fontsize=12)

im_patch_B = ax_patch_B.imshow(
    np.zeros((k_rows, k_cols)),
    cmap='Blues',
    origin='upper',
    extent=[0, k_cols * pixel_scale, k_rows * pixel_scale, 0],
    interpolation='nearest',
    vmin=0, vmax=255
)
ax_patch_B.set_title("Patch B", color='magenta', fontsize=12)

texts_patch_R = [
    [ax_patch_R.text(c * pixel_scale + pixel_scale / 2,
                     r * pixel_scale + pixel_scale / 2,
                     '0', **text_kwargs) for c in range(k_cols)]
    for r in range(k_rows)
]
texts_patch_G = [
    [ax_patch_G.text(c * pixel_scale + pixel_scale / 2,
                     r * pixel_scale + pixel_scale / 2,
                     '0', **text_kwargs) for c in range(k_cols)]
    for r in range(k_rows)
]
texts_patch_B = [
    [ax_patch_B.text(c * pixel_scale + pixel_scale / 2,
                     r * pixel_scale + pixel_scale / 2,
                     '0', **text_kwargs) for c in range(k_cols)]
    for r in range(k_rows)
]

# ----------------------------------------------------
# 2D. Feature Maps (3 filtry)
# ----------------------------------------------------
im_feat_R = ax_feat_R.imshow(
    np.zeros((out_rows, out_cols)),
    cmap='Reds',
    origin='upper',
    extent=[0, out_cols, out_rows, 0],
    interpolation='nearest',
    vmin=0, vmax=max_conv1
)
ax_feat_R.set_title("Feature Map Filter 1", color='red', fontsize=12)

im_feat_G = ax_feat_G.imshow(
    np.zeros((out_rows, out_cols)),
    cmap='Greens',
    origin='upper',
    extent=[0, out_cols, out_rows, 0],
    interpolation='nearest',
    vmin=0, vmax=max_conv2
)
ax_feat_G.set_title("Feature Map Filter 2", color='green', fontsize=12)

im_feat_B = ax_feat_B.imshow(
    np.zeros((out_rows, out_cols)),
    cmap='Blues',
    origin='upper',
    extent=[0, out_cols, out_rows, 0],
    interpolation='nearest',
    vmin=0, vmax=max_conv3
)
ax_feat_B.set_title("Feature Map Filter 3", color='blue', fontsize=12)

texts_feat_R = []
texts_feat_G = []
texts_feat_B = []
for rr in range(out_rows):
    row_text_r = []
    row_text_g = []
    row_text_b = []
    for cc in range(out_cols):
        txt_r = ax_feat_R.text(cc + 0.5, rr + 0.5, "0", **text_kwargs)
        txt_g = ax_feat_G.text(cc + 0.5, rr + 0.5, "0", **text_kwargs)
        txt_b = ax_feat_B.text(cc + 0.5, rr + 0.5, "0", **text_kwargs)
        row_text_r.append(txt_r)
        row_text_g.append(txt_g)
        row_text_b.append(txt_b)
    texts_feat_R.append(row_text_r)
    texts_feat_G.append(row_text_g)
    texts_feat_B.append(row_text_b)

# ----------------------------------------------------
# 2E. Finalne Połączenie (Output RGB)
# ----------------------------------------------------
# Najważniejsze jest tu 'extent=[0, out_cols, out_rows, 0]' i 'origin="upper"',
# aby x = c + 0.5, y = r + 0.5 w tekście pasowały do środka pikseli.
im_combined = ax_combined.imshow(
    np.zeros((out_rows, out_cols, 3), dtype=np.uint8),
    interpolation='nearest',
    origin='upper',
    extent=[0, out_cols, out_rows, 0]
)
ax_combined.set_title("Finalne Połączenie (Output RGB)", color='magenta', fontsize=12)

# Ustawiamy granice osi:
ax_combined.set_xlim(0, out_cols)
ax_combined.set_ylim(out_rows, 0)

texts_combined = []
for rr in range(out_rows):
    row_texts = []
    for cc in range(out_cols):
        txt = ax_combined.text(
            cc + 0.5,  # x
            rr + 0.5,  # y
            "#000000",
            ha='center', va='center', color='white', fontsize=3
        )
        row_texts.append(txt)
    texts_combined.append(row_texts)

# ----------------------------------------------------
# 3. Animacja
# ----------------------------------------------------
total_frames = out_rows * out_cols

def init_anim():
    rect.set_xy((0, 0))
    return [
        rect, im_patch_R, im_patch_G, im_patch_B,
        im_feat_R, im_feat_G, im_feat_B, im_combined
    ]

def update(frame):
    i = frame // out_cols
    j = frame % out_cols

    # Przesunięcie ramki na oryginalnym obrazie
    for artist in ax_orig.patches:
        artist.remove()
    new_rect = patches.Rectangle(
        (j * pixel_scale, i * pixel_scale),
        k_cols * pixel_scale,
        k_rows * pixel_scale,
        edgecolor='magenta',
        facecolor='none', lw=2
    )
    ax_orig.add_patch(new_rect)

    # Wycinanie paczek dla patchy
    patch_R = image_rgb[i : i + k_rows, j : j + k_cols, 0]
    patch_G = image_rgb[i : i + k_rows, j : j + k_cols, 1]
    patch_B = image_rgb[i : i + k_rows, j : j + k_cols, 2]

    im_patch_R.set_data(patch_R)
    im_patch_G.set_data(patch_G)
    im_patch_B.set_data(patch_B)

    for rr in range(k_rows):
        for cc in range(k_cols):
            texts_patch_R[rr][cc].set_text(str(patch_R[rr, cc]))
            texts_patch_G[rr][cc].set_text(str(patch_G[rr, cc]))
            texts_patch_B[rr][cc].set_text(str(patch_B[rr, cc]))

    # Obliczenie konwolucji (surowe)
    conv1 = (patch_R * filter1[:, :, 0]).sum() + \
            (patch_G * filter1[:, :, 1]).sum() + \
            (patch_B * filter1[:, :, 2]).sum()

    conv2 = (patch_R * filter2[:, :, 0]).sum() + \
            (patch_G * filter2[:, :, 1]).sum() + \
            (patch_B * filter2[:, :, 2]).sum()

    conv3 = (patch_R * filter3[:, :, 0]).sum() + \
            (patch_G * filter3[:, :, 1]).sum() + \
            (patch_B * filter3[:, :, 2]).sum()

    feat_R[i, j] = conv1
    feat_G[i, j] = conv2
    feat_B[i, j] = conv3

    im_feat_R.set_data(feat_R)
    im_feat_G.set_data(feat_G)
    im_feat_B.set_data(feat_B)

    texts_feat_R[i][j].set_text(str(conv1))
    texts_feat_G[i][j].set_text(str(conv2))
    texts_feat_B[i][j].set_text(str(conv3))

    # Normalizacja i złożenie w RGB
    norm_R = np.clip(feat_R / max_conv1, 0, 1)
    norm_G = np.clip(feat_G / max_conv2, 0, 1)
    norm_B = np.clip(feat_B / max_conv3, 0, 1)

    combined = np.stack([norm_R, norm_G, norm_B], axis=-1)
    disp_combined = (combined * 255).astype(np.uint8)
    im_combined.set_data(disp_combined)

    # Tekst (#RRGGBB) w finalnym połączeniu
    cr = disp_combined[i, j, 0]
    cg = disp_combined[i, j, 1]
    cb = disp_combined[i, j, 2]
    color_hex = f"#{cr:02X}{cg:02X}{cb:02X}"
    texts_combined[i][j].set_text(color_hex)

    ax_combined.set_title(
        f"Finalne Połączenie (krok {frame + 1}/{total_frames})",
        color='magenta', fontsize=12
    )

    return (
        [new_rect, im_patch_R, im_patch_G, im_patch_B,
         im_feat_R, im_feat_G, im_feat_B, im_combined]
        + [txt for row in texts_patch_R for txt in row]
        + [txt for row in texts_patch_G for txt in row]
        + [txt for row in texts_patch_B for txt in row]
        + [txt for row in texts_feat_R for txt in row]
        + [txt for row in texts_feat_G for txt in row]
        + [txt for row in texts_feat_B for txt in row]
        + [txt for row in texts_combined for txt in row]
    )

ani = animation.FuncAnimation(
    fig,
    update,
    frames=total_frames,
    init_func=init_anim,
    interval=500,
    blit=False,
    repeat=False
)
#ani.save(
    #r'C:\Users\topgu\PycharmProjects\obrazowanie\media\videos\rgbcnn.mp4',
    #writer='ffmpeg',
   # fps=2,
   # dpi=150
#)
plt.tight_layout()
plt.show()