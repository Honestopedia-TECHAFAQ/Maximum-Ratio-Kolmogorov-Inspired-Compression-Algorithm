import streamlit as st
import hashlib
from collections import defaultdict
import base64
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import ffmpeg

class KISCCompressor:
    def __init__(self):
        self.pattern_dict = {}
        self.reverse_dict = {}
        self.next_symbol = 0

    def generate_symbol(self):
        symbol = f'#{self.next_symbol:02x}'
        self.next_symbol += 1
        return symbol

    def find_patterns_fast(self, data, min_length=3):
        data_length = len(data)
        hash_table = {}
        pattern_candidates = set()
        for i in range(data_length - min_length + 1):
            substr = data[i:i + min_length]
            hash_value = hashlib.md5(substr.encode()).hexdigest()
            if hash_value in hash_table:
                pattern_candidates.add(substr)
            else:
                hash_table[hash_value] = i
        compressed_data = data
        for pattern in sorted(pattern_candidates, key=len, reverse=True):
            if pattern not in self.pattern_dict:
                symbol = self.generate_symbol()
                self.pattern_dict[pattern] = symbol
                self.reverse_dict[symbol] = pattern
            compressed_data = compressed_data.replace(pattern, self.pattern_dict[pattern])
        return compressed_data

    def run_length_encoding(self, data):
        encoded_data = []
        count = 1
        for i in range(1, len(data)):
            if data[i] == data[i - 1]:
                count += 1
            else:
                encoded_data.append(f"{data[i - 1]}{count if count > 1 else ''}")
                count = 1
        encoded_data.append(f"{data[-1]}{count if count > 1 else ''}")
        return ''.join(encoded_data)

    def entropy_based_compression(self, data):
        char_frequency = defaultdict(int)
        for char in data:
            char_frequency[char] += 1
        sorted_chars = sorted(char_frequency.items(), key=lambda x: -x[1])
        char_mapping = {char: f'@{i}' for i, (char, _) in enumerate(sorted_chars)}
        compressed_data = ''.join(char_mapping[char] if char in char_mapping else char for char in data)
        for char, mapped in char_mapping.items():
            self.reverse_dict[mapped] = char
        return compressed_data

    def combine_compression_strategies(self, input_data):
        step1 = self.find_patterns_fast(input_data)
        step2 = self.run_length_encoding(step1)
        step3 = self.entropy_based_compression(step2)
        return step3

    def compress(self, input_data):
        start_time = time.time()
        compressed_data = self.combine_compression_strategies(input_data)
        elapsed_time = time.time() - start_time
        st.write(f"Compression completed in {elapsed_time:.4f} seconds")
        return compressed_data

    def compression_metrics(self, original_size, compressed_data):
        compressed_size = len(compressed_data)
        compression_ratio = (1 - compressed_size / original_size) * 100
        return original_size, compressed_size, compression_ratio, len(self.pattern_dict)

    def plot_compression(self, original_size, compressed_size):
        labels = ['Original Size', 'Compressed Size']
        sizes = [original_size, compressed_size]
        fig, ax = plt.subplots()
        ax.bar(labels, sizes, color=['blue', 'orange'])
        ax.set_ylabel('Size in Bytes')
        ax.set_title('Compression Results')
        plt.tight_layout()
        st.pyplot(fig)

    def compress_image(self, image):
        if image.mode == 'RGBA':
            image = image.convert('RGB')  

        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', optimize=True, quality=85)  
        compressed_image = buffer.getvalue()
        compressed_image_b64 = base64.b64encode(compressed_image).decode('utf-8')

        return compressed_image, compressed_image_b64

    def compress_audio(self, audio_file):
        input_file = 'temp_audio.wav'
        output_file = 'compressed_audio.mp3'
        with open(input_file, 'wb') as f:
            f.write(audio_file.read())
        ffmpeg.input(input_file).output(output_file, format='mp3', audio_bitrate='128k').run(overwrite_output=True)
        with open(output_file, 'rb') as f:
            compressed_audio = f.read()
        return compressed_audio

    def compress_video(self, video_file):
        input_file = 'temp_video.mp4'
        output_file = 'compressed_video.mp4'
        with open(input_file, 'wb') as f:
            f.write(video_file.read())
        ffmpeg.input(input_file).output(output_file, vcodec='libx264', crf=23).run(overwrite_output=True)
        with open(output_file, 'rb') as f:
            compressed_video = f.read()
        return compressed_video

def download_link(data, filename, file_label='File'):
    if isinstance(data, str):
        data = data.encode()  
    b64 = base64.b64encode(data).decode()  
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{file_label}</a>'
    return href

def main():
    st.title("Enhanced Kolmogorov-Inspired Compression Algorithm")
    st.write("Upload a file to compress using the custom algorithm.")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "json", "png",])

    if uploaded_file is not None:
        file_data = uploaded_file.read()
        file_type = uploaded_file.type
        compressor = KISCCompressor()  

        if file_type in ["text/plain", "application/json"]:
            file_text = file_data.decode(errors="ignore")
            st.write("### Original File Content (First 1000 characters):")
            st.text(file_text[:1000])

            with st.spinner("Compressing..."):
                compressed_data = compressor.compress(file_text)

            st.write("### Compressed Data (First 1000 characters):")
            st.text(compressed_data[:1000])

            compressed_filename = "compressed_output.txt"
            st.markdown(download_link(compressed_data, compressed_filename, "Download Compressed File"), unsafe_allow_html=True)

            original_size = len(file_text)
            compressed_size = len(compressed_data)
            original_size, compressed_size, compression_ratio, unique_patterns = compressor.compression_metrics(original_size, compressed_data)

            st.write("### Compression Summary:")
            st.write(f"Original Size: {original_size} bytes")
            st.write(f"Compressed Size: {compressed_size} bytes")
            st.write(f"Compression Ratio: {compression_ratio:.2f}%")
            st.write(f"Unique Patterns Found: {unique_patterns}")
            compressor.plot_compression(original_size, compressed_size)

        elif file_type == "image/png":
            image = Image.open(io.BytesIO(file_data))
            st.image(image, caption='Original Image', use_column_width=True)

            with st.spinner("Compressing image..."):
                compressed_image, compressed_image_b64 = compressor.compress_image(image)

            st.write("### Compressed Image Size:")
            st.write(f"Compressed Image Size: {len(compressed_image)} bytes")
            st.markdown(download_link(compressed_image, "compressed_image.jpg", "Download Compressed Image"), unsafe_allow_html=True)
            compressed_image_txt_filename = "compressed_image_data.txt"
            st.markdown(download_link(compressed_image_b64, compressed_image_txt_filename, "Download Compressed Image Data"), unsafe_allow_html=True)

        elif file_type == "audio/wav":
            st.write("### Original Audio File:")
            st.audio(file_data, format="audio/wav")

            with st.spinner("Compressing audio..."):
                compressed_audio = compressor.compress_audio(uploaded_file)

            st.write("### Compressed Audio File Size:")
            st.write(f"Compressed Audio Size: {len(compressed_audio)} bytes")
            st.markdown(download_link(compressed_audio, "compressed_audio.mp3", "Download Compressed Audio"), unsafe_allow_html=True)

        elif file_type == "video/mp4":
            st.write("### Original Video File:")
            st.video(file_data)

            with st.spinner("Compressing video..."):
                compressed_video = compressor.compress_video(uploaded_file)

            st.write("### Compressed Video File Size:")
            st.write(f"Compressed Video Size: {len(compressed_video)} bytes")
            st.markdown(download_link(compressed_video, "compressed_video.mp4", "Download Compressed Video"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
