#!/usr/bin/env python3
import argparse
import sys
import os
from PIL import Image
import numpy as np
import cv2
import wave
import struct
import hashlib
import base64
import json
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class XillenSteganography:
    def __init__(self):
        self.supported_formats = {
            'image': ['.png', '.bmp', '.jpg', '.jpeg', '.tiff'],
            'audio': ['.wav', '.wave'],
            'video': ['.avi', '.mp4', '.mov', '.mkv']
        }
        self.results = {}
        
    def hide_in_image_lsb(self, carrier_path, secret_data, output_path, bits_per_channel=2):
        """Скрытие данных в изображении методом LSB"""
        print(f"[+] Hiding data in image using LSB method ({bits_per_channel} bits per channel)")
        
        try:
            img = Image.open(carrier_path)
            img_array = np.array(img)
            
            if len(img_array.shape) == 3:
                height, width, channels = img_array.shape
            else:
                height, width = img_array.shape
                channels = 1
                img_array = img_array.reshape(height, width, 1)
            
            if isinstance(secret_data, str):
                secret_data = secret_data.encode('utf-8')
            
            data_length = len(secret_data)
            max_capacity = (height * width * channels * bits_per_channel) // 8
            
            if data_length > max_capacity:
                raise ValueError(f"Data too large. Max capacity: {max_capacity} bytes, data size: {data_length} bytes")
            
            print(f"[+] Carrier image: {width}x{height}, {channels} channels")
            print(f"[+] Data size: {data_length} bytes")
            print(f"[+] Max capacity: {max_capacity} bytes")
            print(f"[+] Using {bits_per_channel} bits per channel")
            
            binary_data = ''.join(format(byte, '08b') for byte in secret_data)
            data_length_binary = format(data_length, '032b')
            
            full_binary = data_length_binary + binary_data
            data_index = 0
            
            for y in range(height):
                for x in range(width):
                    for c in range(channels):
                        if data_index < len(full_binary):
                            pixel_value = img_array[y, x, c]
                            new_value = self.modify_lsb(pixel_value, full_binary[data_index:data_index + bits_per_channel], bits_per_channel)
                            img_array[y, x, c] = new_value
                            data_index += bits_per_channel
                        else:
                            break
                    if data_index >= len(full_binary):
                        break
                if data_index >= len(full_binary):
                    break
            
            result_img = Image.fromarray(img_array)
            result_img.save(output_path)
            
            print(f"[+] Data hidden successfully in {output_path}")
            self.results['hidden_data'] = {
                'carrier': carrier_path,
                'output': output_path,
                'data_size': data_length,
                'method': f'LSB_{bits_per_channel}bit',
                'capacity_used': (data_length / max_capacity) * 100
            }
            
        except Exception as e:
            print(f"[-] Error hiding data: {e}")
            raise
    
    def extract_from_image_lsb(self, stego_path, output_path=None, bits_per_channel=2):
        """Извлечение данных из изображения методом LSB"""
        print(f"[+] Extracting data from image using LSB method ({bits_per_channel} bits per channel)")
        
        try:
            img = Image.open(stego_path)
            img_array = np.array(img)
            
            if len(img_array.shape) == 3:
                height, width, channels = img_array.shape
            else:
                height, width = img_array.shape
                channels = 1
                img_array = img_array.reshape(height, width, 1)
            
            print(f"[+] Stego image: {width}x{height}, {channels} channels")
            
            binary_data = ""
            data_length = 0
            data_length_binary = ""
            
            for y in range(height):
                for x in range(width):
                    for c in range(channels):
                        pixel_value = img_array[y, x, c]
                        lsb_bits = self.extract_lsb(pixel_value, bits_per_channel)
                        binary_data += lsb_bits
                        
                        if len(data_length_binary) < 32:
                            data_length_binary += lsb_bits
                            if len(data_length_binary) == 32:
                                data_length = int(data_length_binary, 2)
                                print(f"[+] Detected data length: {data_length} bytes")
                        
                        if len(binary_data) >= (32 + data_length * 8):
                            break
                    if len(binary_data) >= (32 + data_length * 8):
                        break
                if len(binary_data) >= (32 + data_length * 8):
                    break
            
            data_binary = binary_data[32:32 + data_length * 8]
            extracted_data = bytes(int(data_binary[i:i+8], 2) for i in range(0, len(data_binary), 8))
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(extracted_data)
                print(f"[+] Data extracted to {output_path}")
            else:
                try:
                    text_data = extracted_data.decode('utf-8')
                    print(f"[+] Extracted text: {text_data}")
                except:
                    print(f"[+] Extracted binary data ({len(extracted_data)} bytes)")
                    print(f"[+] Hex: {extracted_data.hex()}")
            
            self.results['extracted_data'] = {
                'stego_file': stego_path,
                'data_size': data_length,
                'method': f'LSB_{bits_per_channel}bit'
            }
            
            return extracted_data
            
        except Exception as e:
            print(f"[-] Error extracting data: {e}")
            raise
    
    def hide_in_audio_lsb(self, carrier_path, secret_data, output_path, bits_per_sample=2):
        """Скрытие данных в аудиофайле методом LSB"""
        print(f"[+] Hiding data in audio using LSB method ({bits_per_sample} bits per sample)")
        
        try:
            with wave.open(carrier_path, 'rb') as audio:
                params = audio.getparams()
                frames = audio.readframes(audio.getnframes())
                
                if isinstance(secret_data, str):
                    secret_data = secret_data.encode('utf-8')
                
                data_length = len(secret_data)
                max_capacity = (len(frames) * bits_per_sample) // 8
                
                if data_length > max_capacity:
                    raise ValueError(f"Data too large. Max capacity: {max_capacity} bytes, data size: {data_length} bytes")
                
                print(f"[+] Carrier audio: {params.nchannels} channels, {params.sampwidth} bytes per sample")
                print(f"[+] Data size: {data_length} bytes")
                print(f"[+] Max capacity: {max_capacity} bytes")
                
                binary_data = ''.join(format(byte, '08b') for byte in secret_data)
                data_length_binary = format(data_length, '032b')
                
                full_binary = data_length_binary + binary_data
                data_index = 0
                
                frames_array = list(frames)
                
                for i in range(len(frames_array)):
                    if data_index < len(full_binary):
                        sample_value = frames_array[i]
                        new_value = self.modify_lsb(sample_value, full_binary[data_index:data_index + bits_per_sample], bits_per_sample)
                        frames_array[i] = new_value
                        data_index += bits_per_sample
                    else:
                        break
                
                modified_frames = bytes(frames_array)
                
                with wave.open(output_path, 'wb') as output_audio:
                    output_audio.setparams(params)
                    output_audio.writeframes(modified_frames)
                
                print(f"[+] Data hidden successfully in {output_path}")
                self.results['hidden_audio'] = {
                    'carrier': carrier_path,
                    'output': output_path,
                    'data_size': data_length,
                    'method': f'LSB_{bits_per_sample}bit'
                }
                
        except Exception as e:
            print(f"[-] Error hiding data in audio: {e}")
            raise
    
    def hide_in_video_lsb(self, carrier_path, secret_data, output_path, frame_interval=1):
        """Скрытие данных в видеофайле методом LSB"""
        print(f"[+] Hiding data in video using LSB method (frame interval: {frame_interval})")
        
        try:
            cap = cv2.VideoCapture(carrier_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if isinstance(secret_data, str):
                secret_data = secret_data.encode('utf-8')
            
            data_length = len(secret_data)
            max_capacity = (total_frames // frame_interval) * width * height * 3 // 8
            
            if data_length > max_capacity:
                raise ValueError(f"Data too large. Max capacity: {max_capacity} bytes, data size: {data_length} bytes")
            
            print(f"[+] Carrier video: {width}x{height}, {fps} FPS, {total_frames} frames")
            print(f"[+] Data size: {data_length} bytes")
            print(f"[+] Max capacity: {max_capacity} bytes")
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            binary_data = ''.join(format(byte, '08b') for byte in secret_data)
            data_length_binary = format(data_length, '032b')
            
            full_binary = data_length_binary + binary_data
            data_index = 0
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0 and data_index < len(full_binary):
                    frame = self.hide_data_in_frame(frame, full_binary, data_index)
                    data_index += width * height * 3
                
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            print(f"[+] Data hidden successfully in {output_path}")
            self.results['hidden_video'] = {
                'carrier': carrier_path,
                'output': output_path,
                'data_size': data_length,
                'method': f'LSB_frame_interval_{frame_interval}'
            }
            
        except Exception as e:
            print(f"[-] Error hiding data in video: {e}")
            raise
    
    def hide_data_in_frame(self, frame, binary_data, start_index):
        """Скрытие данных в одном кадре"""
        height, width, channels = frame.shape
        data_index = start_index
        
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    if data_index < len(binary_data):
                        pixel_value = frame[y, x, c]
                        new_value = self.modify_lsb(pixel_value, binary_data[data_index:data_index + 1], 1)
                        frame[y, x, c] = new_value
                        data_index += 1
                    else:
                        break
                if data_index >= len(binary_data):
                    break
            if data_index >= len(binary_data):
                break
        
        return frame
    
    def modify_lsb(self, value, binary_bits, bits_count):
        """Модификация младших значащих битов"""
        if len(binary_bits) < bits_count:
            binary_bits = binary_bits.ljust(bits_count, '0')
        
        mask = (1 << bits_count) - 1
        cleared_value = value & ~mask
        new_bits = int(binary_bits, 2)
        
        return cleared_value | new_bits
    
    def extract_lsb(self, value, bits_count):
        """Извлечение младших значащих битов"""
        mask = (1 << bits_count) - 1
        extracted_bits = value & mask
        return format(extracted_bits, f'0{bits_count}b')
    
    def hide_in_text(self, carrier_text, secret_data, method='whitespace'):
        """Скрытие данных в тексте"""
        print(f"[+] Hiding data in text using {method} method")
        
        try:
            if isinstance(secret_data, str):
                secret_data = secret_data.encode('utf-8')
            
            if method == 'whitespace':
                binary_data = ''.join(format(byte, '08b') for byte in secret_data)
                words = carrier_text.split()
                
                if len(binary_data) > len(words):
                    raise ValueError("Text too short for hiding data")
                
                result_words = []
                for i, word in enumerate(words):
                    if i < len(binary_data):
                        if binary_data[i] == '1':
                            result_words.append(word + ' ')
                        else:
                            result_words.append(word + '  ')
                    else:
                        result_words.append(word)
                
                result_text = ''.join(result_words)
                
            elif method == 'case':
                binary_data = ''.join(format(byte, '08b') for byte in secret_data)
                result_text = ""
                data_index = 0
                
                for char in carrier_text:
                    if char.isalpha() and data_index < len(binary_data):
                        if binary_data[data_index] == '1':
                            result_text += char.upper()
                        else:
                            result_text += char.lower()
                        data_index += 1
                    else:
                        result_text += char
                
                if data_index < len(binary_data):
                    raise ValueError("Text too short for hiding data")
            
            print(f"[+] Data hidden successfully in text")
            self.results['hidden_text'] = {
                'method': method,
                'data_size': len(secret_data)
            }
            
            return result_text
            
        except Exception as e:
            print(f"[-] Error hiding data in text: {e}")
            raise
    
    def extract_from_text(self, stego_text, method='whitespace'):
        """Извлечение данных из текста"""
        print(f"[+] Extracting data from text using {method} method")
        
        try:
            if method == 'whitespace':
                words = stego_text.split()
                binary_data = ""
                
                for word in words:
                    if word.endswith('  '):
                        binary_data += '0'
                    elif word.endswith(' '):
                        binary_data += '1'
                    else:
                        binary_data += '0'
                
            elif method == 'case':
                binary_data = ""
                
                for char in stego_text:
                    if char.isalpha():
                        if char.isupper():
                            binary_data += '1'
                        else:
                            binary_data += '0'
            
            data_bytes = bytes(int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8))
            
            try:
                text_data = data_bytes.decode('utf-8')
                print(f"[+] Extracted text: {text_data}")
            except:
                print(f"[+] Extracted binary data ({len(data_bytes)} bytes)")
            
            self.results['extracted_text'] = {
                'method': method,
                'data_size': len(data_bytes)
            }
            
            return data_bytes
            
        except Exception as e:
            print(f"[-] Error extracting data from text: {e}")
            raise
    
    def analyze_stego_file(self, file_path):
        """Анализ файла на наличие скрытых данных"""
        print(f"[+] Analyzing file for steganography: {file_path}")
        
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in self.supported_formats['image']:
                self.analyze_image_stego(file_path)
            elif file_ext in self.supported_formats['audio']:
                self.analyze_audio_stego(file_path)
            elif file_ext in self.supported_formats['video']:
                self.analyze_video_stego(file_path)
            else:
                print(f"[-] Unsupported file format: {file_ext}")
                
        except Exception as e:
            print(f"[-] Error analyzing file: {e}")
    
    def analyze_image_stego(self, file_path):
        """Анализ изображения на наличие скрытых данных"""
        try:
            img = Image.open(file_path)
            img_array = np.array(img)
            
            if len(img_array.shape) == 3:
                height, width, channels = img_array.shape
            else:
                height, width = img_array.shape
                channels = 1
            
            print(f"[+] Image analysis: {width}x{height}, {channels} channels")
            
            lsb_analysis = self.analyze_lsb_pattern(img_array)
            
            print(f"[+] LSB Analysis:")
            print(f"    - Randomness score: {lsb_analysis['randomness']:.4f}")
            print(f"    - Entropy: {lsb_analysis['entropy']:.4f}")
            print(f"    - Potential data: {lsb_analysis['potential_data']} bytes")
            
            self.results['image_analysis'] = lsb_analysis
            
        except Exception as e:
            print(f"[-] Error analyzing image: {e}")
    
    def analyze_lsb_pattern(self, img_array):
        """Анализ паттернов LSB"""
        if len(img_array.shape) == 3:
            height, width, channels = img_array.shape
            lsb_values = []
            
            for y in range(min(height, 100)):
                for x in range(min(width, 100)):
                    for c in range(channels):
                        lsb_values.append(img_array[y, x, c] & 1)
            
            lsb_array = np.array(lsb_values)
            
            randomness = np.mean(lsb_array)
            entropy = -np.sum(np.bincount(lsb_array) / len(lsb_array) * np.log2(np.bincount(lsb_array) / len(lsb_array) + 1e-10))
            
            potential_data = int(height * width * channels / 8)
            
            return {
                'randomness': randomness,
                'entropy': entropy,
                'potential_data': potential_data
            }
        
        return {'randomness': 0, 'entropy': 0, 'potential_data': 0}
    
    def save_results(self, output_file):
        """Сохранение результатов анализа"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"[+] Results saved to {output_file}")
        except Exception as e:
            print(f"[-] Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='XILLEN Advanced Steganography Tool')
    parser.add_argument('action', choices=['hide', 'extract', 'analyze'], help='Action to perform')
    parser.add_argument('--carrier', help='Carrier file path')
    parser.add_argument('--secret', help='Secret data or file path')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--method', default='lsb', choices=['lsb', 'whitespace', 'case'], help='Steganography method')
    parser.add_argument('--bits', type=int, default=2, help='Bits per channel for LSB')
    parser.add_argument('--results', help='Results output file')
    
    args = parser.parse_args()
    
    stego = XillenSteganography()
    
    try:
        if args.action == 'hide':
            if not args.carrier or not args.secret or not args.output:
                print("[-] Error: hide action requires --carrier, --secret, and --output")
                sys.exit(1)
            
            if os.path.isfile(args.secret):
                with open(args.secret, 'rb') as f:
                    secret_data = f.read()
            else:
                secret_data = args.secret
            
            file_ext = os.path.splitext(args.carrier)[1].lower()
            
            if file_ext in stego.supported_formats['image']:
                stego.hide_in_image_lsb(args.carrier, secret_data, args.output, args.bits)
            elif file_ext in stego.supported_formats['audio']:
                stego.hide_in_audio_lsb(args.carrier, secret_data, args.output, args.bits)
            elif file_ext in stego.supported_formats['video']:
                stego.hide_in_video_lsb(args.carrier, secret_data, args.output)
            else:
                print(f"[-] Unsupported carrier format: {file_ext}")
        
        elif args.action == 'extract':
            if not args.carrier:
                print("[-] Error: extract action requires --carrier")
                sys.exit(1)
            
            file_ext = os.path.splitext(args.carrier)[1].lower()
            
            if file_ext in stego.supported_formats['image']:
                stego.extract_from_image_lsb(args.carrier, args.output, args.bits)
            elif file_ext in stego.supported_formats['audio']:
                print("[-] Audio extraction not implemented yet")
            elif file_ext in stego.supported_formats['video']:
                print("[-] Video extraction not implemented yet")
            else:
                print(f"[-] Unsupported file format: {file_ext}")
        
        elif args.action == 'analyze':
            if not args.carrier:
                print("[-] Error: analyze action requires --carrier")
                sys.exit(1)
            
            stego.analyze_stego_file(args.carrier)
        
        if args.results:
            stego.save_results(args.results)
            
    except KeyboardInterrupt:
        print("\n[!] Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"[-] Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
