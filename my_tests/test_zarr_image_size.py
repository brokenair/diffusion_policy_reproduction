import zarr
root = zarr.open('data/pusht_real_v1.zarr', mode='r')
img_shape = root['data']['img_15000px'].shape
print(f"实际图像形状: {img_shape}")