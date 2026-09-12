
import json
import codecs
import os

def fix_encoding(file_path, output_path=None):
    """Repair a JSON file that was written with a doubly-encoded string."""
    if output_path is None:
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_fixed{ext}"
    
    try:
        # Read the file as text first
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        # Strategy 1: parse the JSON as-is
        try:
            data = json.loads(raw_content)
            print("Parsed directly as JSON")
        except:
            # Strategy 2: undo a possible double encoding
            try:
                # Decode the raw string to Unicode, then parse it as JSON
                fixed_content = raw_content.encode('latin1').decode('utf-8')
                data = json.loads(fixed_content)
                print("Repaired via latin1 -> utf-8")
            except:
                # Strategy 3: other encoding combinations
                try:
                    fixed_content = codecs.decode(raw_content, 'unicode_escape')
                    data = json.loads(fixed_content)
                    print("Repaired via unicode_escape")
                except:
                    # Strategy 4: last-resort attempt
                    try:
                        fixed_content = raw_content.encode('utf-8').decode('unicode_escape')
                        data = json.loads(fixed_content)
                        print("Repaired via utf-8 -> unicode_escape")
                    except Exception as e:
                        raise Exception(f"Cannot parse JSON: {e}")
        
        # Write the repaired data back as UTF-8 JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"File repaired and saved to: {output_path}")
        
        # Print one example entry
        if isinstance(data, list) and len(data) > 0:
            print("\nExample output (first entry):")
            print(json.dumps(data[0], ensure_ascii=False, indent=4))
        
        return data
    
    except Exception as e:
        print(f"Error while processing the file: {e}")
        
        # Fall back to reading the file in binary mode
        try:
            print("Trying the binary repair path...")
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Try several decoding combinations
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
                    
                    # Attempt to parse the JSON
                    data = json.loads(decoded)
                    
                    # Parsed successfully: save the result
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=4)
                    
                    print(f"Repaired with {input_encoding} -> {second_encoding}")
                    print(f"File saved to: {output_path}")
                    
                    # Print an example
                    if isinstance(data, list) and len(data) > 0:
                        print("\nExample output (first entry):")
                        print(json.dumps(data[0], ensure_ascii=False, indent=4))
                    
                    return data
                except Exception:
                    continue
            
            print("All repair strategies failed")
            return None
        except Exception as e:
            print(f"Binary repair also failed: {e}")
            return None

# Repair the target file
input_file = '/path/to/CT-CHAT2/VQA_dataset/output_validation_vicuna.json'
output_file = '/path/to/CT-CHAT2/VQA_dataset/output_validation_vicuna_fixed.json'

fix_encoding(input_file, output_file)
