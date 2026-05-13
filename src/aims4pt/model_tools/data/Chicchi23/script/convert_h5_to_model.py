import tensorflow as tf
import os

def convert_h5_to_savedmodel(h5_path):
    original_path = h5_path
    model = tf.keras.models.load_model(h5_path, compile=False)
    save_path = h5_path.replace('.h5', '_saved')
    model.save(save_path)
    print(f"✅ Converted: {h5_path} → {save_path}")
    # remove original .h5 file
    os.remove(original_path)

# 示例：遍历当前文件夹中所有 .h5 文件
for file in os.listdir('.'):
    if file.endswith('.h5'):
        try:
            convert_h5_to_savedmodel(file)
        except Exception as e:
            print(f"❌ Failed: {file} - {e}")
