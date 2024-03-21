import os
from PIL import Image
import subprocess

def compress_tga_rle(data):
    compressed = bytearray()
    index = 0

    while index < len(data):
        # 查找重复或非重复的像素序列
        sequence_start = index
        is_repeating = False
        max_sequence_length = 128  # RLE块的最大长度

        while index - sequence_start < max_sequence_length and index < len(data) - 1:
            if data[index] == data[index + 1]:
                if index - sequence_start > 1 and not is_repeating:
                    break
                is_repeating = True
            elif is_repeating:
                break
            index += 1

        sequence_length = index - sequence_start + 1

        # 编码RLE块
        if is_repeating:
            compressed.append(0x80 | (sequence_length - 1))  # 设置RLE位并加入长度
            compressed.extend(data[sequence_start:sequence_start + 1])
        else:
            compressed.append(sequence_length - 1)
            compressed.extend(data[sequence_start:sequence_start + sequence_length])

        index += 1

    return compressed


def convert_png_to_tga(png_file_path, tga_file_path):
    # 打开PNG文件
    with Image.open(png_file_path) as img:
        # 保存为TGA格式
        img.save(tga_file_path, format='TGA')

def process_tga_files(folder_path, output_folder, sub_quality, resize_factor=None):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.tga'):
            file_path = os.path.join(folder_path, filename)
            output_png_path = os.path.join(output_folder, filename[:-4] + '.png')


            # 打开图像
            with Image.open(file_path) as img:
                # 如果需要，调整图像大小
                if resize_factor:
                    new_size = tuple([int(x * resize_factor) for x in img.size])
                    # 转换为P-模式（带调色板的模式），这可能会减少颜色的数量
                    img = img.convert('P', palette=Image.ADAPTIVE)
                    # 调整大小
                    img = img.resize(new_size, Image.LANCZOS)

                # 保存为PNG格式
                img.save(output_png_path, 'PNG')

                try:
                    # 构建 pngquant 命令
                    command = ['pngquant', '--force', '--quality', sub_quality, '--output', output_png_path, output_png_path]

                    # 执行命令
                    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"Image compressed successfully: {output_png_path}")
                    # convert_png_to_tga(output_pngq_path, output_tga_path)
                except subprocess.CalledProcessError as e:
                    print(f"Error during compression: {e.stderr.decode().strip()}")
                    # convert_png_to_tga(output_png_path, output_tga_path)



if __name__ == '__main__':
    # 使用此函数处理文件夹中的文件
    process_tga_files('../data/debug/tga', '../data/result_compress', sub_quality='40-60', resize_factor=1)
    # with open('../data/debug/tga/320682001000JJ000030066.tga', 'rb') as file:
    #     tga_data = file.read()
    #
    # # 压缩TGA数据
    # compressed_data = compress_tga_rle(tga_data)
    #
    # # 将压缩后的数据写入新文件
    # with open('../data/result_compress/320682001000JJ000030066.tga', 'wb') as file:
    #     file.write(compressed_data)



