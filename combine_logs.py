import os

def combine_markdown_files(directory_path, output_file="combined_debug_logs.txt"):
    # 确保目录存在
    if not os.path.exists(directory_path):
        print(f"错误：找不到目录 {directory_path}")
        return

    combined_content = []

    # 获取目录下所有 md 文件并排序（按文件名排序方便阅读）
    files = sorted([f for f in os.listdir(directory_path) if f.endswith('.md')])

    if not files:
        print("该目录下没有找到 .md 文件。")
        return

    for filename in files:
        file_path = os.path.join(directory_path, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # 构建组合格式
                combined_content.append(f"--- START OF FILE: {filename} ---")
                combined_content.append(content)
                combined_content.append(f"--- END OF FILE: {filename} ---\n")
        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")

    # 将结果写入新文件
    output_path = os.path.join(directory_path, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(combined_content))

    print(f"成功！所有内容已合并至：\n{output_path}")
