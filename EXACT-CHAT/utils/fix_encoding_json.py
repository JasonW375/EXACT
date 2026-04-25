
import json
import codecs
import os

def fix_encoding(file_path, output_path=None):
    """修复JSON文件的多重编码问题"""
    if output_path is None:
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_fixed{ext}"
    
    try:
        # 尝试直接读取文件内容作为文本
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        # 方法1：尝试直接加载JSON
        try:
            data = json.loads(raw_content)
            print("直接JSON解析成功")
        except:
            # 方法2：尝试修复可能的多重编码问题
            try:
                # 将原始字符串解码为Unicode，再作为JSON解析
                fixed_content = raw_content.encode('latin1').decode('utf-8')
                data = json.loads(fixed_content)
                print("使用latin1->utf-8转换修复成功")
            except:
                # 方法3：尝试其他编码组合
                try:
                    fixed_content = codecs.decode(raw_content, 'unicode_escape')
                    data = json.loads(fixed_content)
                    print("使用unicode_escape解码修复成功")
                except:
                    # 方法4：更复杂的修复尝试
                    try:
                        fixed_content = raw_content.encode('utf-8').decode('unicode_escape')
                        data = json.loads(fixed_content)
                        print("使用utf-8->unicode_escape修复成功")
                    except Exception as e:
                        raise Exception(f"无法解析JSON: {e}")
        
        # 将修复后的数据保存为UTF-8编码的JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"文件已修复并保存到: {output_path}")
        
        # 打印一个示例条目
        if isinstance(data, list) and len(data) > 0:
            print("\n示例输出（第一个条目）:")
            print(json.dumps(data[0], ensure_ascii=False, indent=4))
        
        return data
    
    except Exception as e:
        print(f"处理文件时出错: {e}")
        
        # 尝试更直接的二进制方式读取和修复
        try:
            print("尝试二进制方式修复...")
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # 尝试不同的解码方式
            decode_methods = [
                ('utf-8', None),
                ('utf-8', 'unicode_escape'),
                ('latin1', 'utf-8'),
                ('latin1', None)
            ]
            
            for input_encoding, second_encoding in decode_methods:
                try:
                    decoded = content.decode(input_encoding)
                    if second_encoding:
                        decoded = decoded.encode('utf-8').decode(second_encoding)
                    
                    # 尝试解析JSON
                    data = json.loads(decoded)
                    
                    # 成功解析，保存结果
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    
                    print(f"使用 {input_encoding} -> {second_encoding} 修复成功")
                    print(f"文件已保存到: {output_path}")
                    
                    # 打印示例
                    if isinstance(data, list) and len(data) > 0:
                        print("\n示例输出（第一个条目）:")
                        print(json.dumps(data[0], ensure_ascii=False, indent=4))
                    
                    return data
                except Exception:
                    continue
            
            print("所有修复方法都失败了")
            return None
        except Exception as e:
            print(f"二进制修复也失败了: {e}")
            return None

# 使用该函数修复您的文件
input_file = '/path/to/CT-CHAT2/VQA_dataset/output_validation_vicuna.json'
output_file = '/path/to/CT-CHAT2/VQA_dataset/output_validation_vicuna_fixed.json'

fix_encoding(input_file, output_file)
